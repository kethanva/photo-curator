"""
Unit tests for src/privacy.py — screenshot detection, home filtering, document CLIP check.
The CLIP-dependent functions are tested with mocks; pure-Python functions are tested directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.privacy import (
    _SCREEN_RESOLUTIONS,
    _TEXT_HEAVY_PROMPTS,
    _haversine_km,
    _is_private_from_probs,
    assess,
    is_document_from_embedding,
    is_home_private,
    is_reshared_filename,
    is_screenshot,
    is_text_heavy_from_embedding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, color=(100, 100, 100))


# ---------------------------------------------------------------------------
# is_screenshot tests
# ---------------------------------------------------------------------------

class TestIsScreenshot:
    def test_gps_photo_not_screenshot(self):
        img = _img((1080, 1920))
        assert is_screenshot(img, camera_model="", has_gps=True) is False

    def test_camera_model_not_screenshot(self):
        img = _img((1080, 1920))
        assert is_screenshot(img, camera_model="iPhone 15", has_gps=False) is False

    def test_known_screen_resolution_is_screenshot(self):
        w, h = next(iter(_SCREEN_RESOLUTIONS))
        img = _img((w, h))
        assert is_screenshot(img, camera_model="", has_gps=False) is True

    def test_known_resolution_landscape_is_screenshot(self):
        """Landscape orientation of a known screen size also counts."""
        w, h = next(iter(_SCREEN_RESOLUTIONS))
        img = _img((h, w))  # swapped
        assert is_screenshot(img, camera_model="", has_gps=False) is True

    def test_non_screen_resolution_not_screenshot(self):
        img = _img((1234, 5678))  # unlikely screen resolution
        assert is_screenshot(img, camera_model="", has_gps=False) is False


# ---------------------------------------------------------------------------
# _haversine_km tests
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine_km(37.77, -122.41, 37.77, -122.41) == pytest.approx(0.0, abs=0.001)

    def test_known_distance_sf_to_la(self):
        """SF to LA is roughly 560 km."""
        dist = _haversine_km(37.77, -122.41, 34.05, -118.24)
        assert 540 <= dist <= 600

    def test_north_south_pole(self):
        """Two poles are ~20000 km apart (half circumference)."""
        dist = _haversine_km(90.0, 0.0, -90.0, 0.0)
        assert 19000 <= dist <= 21000


# ---------------------------------------------------------------------------
# is_home_private tests
# ---------------------------------------------------------------------------

class TestIsHomePrivate:
    def test_no_home_returns_false(self):
        assert is_home_private(37.77, -122.41, face_count=0, home=None) is False

    def test_no_gps_returns_false(self):
        home = (37.77, -122.41)
        assert is_home_private(0.0, 0.0, face_count=0, home=home) is False

    def test_at_home_solo_returns_true(self):
        home = (37.7749, -122.4194)
        assert is_home_private(37.7749, -122.4194, face_count=0, home=home) is True

    def test_at_home_with_people_returns_false(self):
        home = (37.7749, -122.4194)
        assert is_home_private(37.7749, -122.4194, face_count=2, home=home) is False

    def test_far_from_home_returns_false(self):
        home = (37.7749, -122.4194)
        assert is_home_private(48.8566, 2.3522, face_count=0, home=home) is False

    def test_custom_radius(self):
        home = (37.77, -122.41)
        # ~2 km away
        result_tight = is_home_private(37.79, -122.41, face_count=0, home=home, radius_km=1.0)
        result_loose = is_home_private(37.79, -122.41, face_count=0, home=home, radius_km=5.0)
        assert result_tight is False
        assert result_loose is True


# ---------------------------------------------------------------------------
# assess tests (CLIP path mocked)
# ---------------------------------------------------------------------------

class TestAssess:
    def _img(self):
        return _img((640, 480))

    def test_screenshot_excluded(self):
        """If is_screenshot returns True, assess returns True."""
        img = _img((1080, 1920))  # known screen resolution
        result = assess(
            img=img,
            camera_model="", has_gps=False,
            lat=0.0, lon=0.0, face_count=0,
            home=None, home_radius_km=0.5,
            filter_screenshots=True,
            filter_documents=False,
            filter_home_private=False,
        )
        assert result is True

    def test_screenshot_filter_off(self):
        """With filter_screenshots=False, screenshots are not excluded."""
        img = _img((1080, 1920))
        result = assess(
            img=img,
            camera_model="", has_gps=False,
            lat=0.0, lon=0.0, face_count=0,
            home=None, home_radius_km=0.5,
            filter_screenshots=False,
            filter_documents=False,
            filter_home_private=False,
        )
        assert result is False

    def test_home_private_excluded(self):
        home = (37.7749, -122.4194)
        img = self._img()
        result = assess(
            img=img,
            camera_model="iPhone", has_gps=True,
            lat=37.7749, lon=-122.4194, face_count=0,
            home=home, home_radius_km=0.5,
            filter_screenshots=False,
            filter_documents=False,
            filter_home_private=True,
        )
        assert result is True

    def test_document_clip_mock(self):
        """CLIP document check mocked to return True → assess returns True."""
        img = self._img()
        with patch("src.privacy.is_document_clip", return_value=True):
            result = assess(
                img=img,
                camera_model="iPhone", has_gps=True,
                lat=37.0, lon=-122.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is True

    def test_document_clip_mock_false(self):
        """CLIP document check mocked to return False → assess returns False."""
        img = self._img()
        with patch("src.privacy.is_document_clip", return_value=False):
            result = assess(
                img=img,
                camera_model="iPhone", has_gps=True,
                lat=0.0, lon=0.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is False

    def test_normal_photo_not_excluded(self):
        """Photo with camera model, GPS, and CLIP not doc → should not be excluded."""
        img = self._img()
        with patch("src.privacy.is_document_clip", return_value=False):
            result = assess(
                img=img,
                camera_model="Sony A7", has_gps=True,
                lat=48.8, lon=2.3, face_count=2,
                home=None, home_radius_km=0.5,
                filter_screenshots=True,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is False

    def test_document_embedding_fast_path_used_when_provided(self):
        """When a clip_emb is supplied, assess routes to is_document_from_embedding,
        not is_document_clip (which would otherwise re-run CLIP on the image)."""
        img = self._img()
        emb = np.ones(512, dtype=np.float32)
        with patch("src.privacy.is_document_from_embedding", return_value=True) as emb_check, \
             patch("src.privacy.is_document_clip", return_value=False) as img_check:
            result = assess(
                img=img,
                camera_model="iPhone", has_gps=True,
                lat=0.0, lon=0.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
                clip_emb=emb,
            )
        assert result is True
        emb_check.assert_called_once()
        img_check.assert_not_called()

    def test_document_image_path_used_when_no_embedding(self):
        """Without clip_emb, assess falls back to is_document_clip (original path)."""
        img = self._img()
        with patch("src.privacy.is_document_from_embedding", return_value=False) as emb_check, \
             patch("src.privacy.is_document_clip", return_value=True) as img_check:
            result = assess(
                img=img,
                camera_model="iPhone", has_gps=True,
                lat=0.0, lon=0.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is True
        img_check.assert_called_once()
        emb_check.assert_not_called()


# ---------------------------------------------------------------------------
# Reshared-filename tests
# ---------------------------------------------------------------------------

class TestIsPrivateFromProbs:
    """Locks in the (threshold, ratio) gate that prevents over-flagging."""

    def _probs(self, best_private: float, normal: float) -> np.ndarray:
        # 10 private prompts + 1 normal = 11 slots; fill rest with uniform
        # residual so the array sums to 1.0 (as real softmax output would).
        n_priv = 10
        residual = max(0.0, 1.0 - best_private - normal) / (n_priv - 1)
        arr = np.full(n_priv + 1, residual, dtype=float)
        arr[0] = best_private
        arr[-1] = normal
        return arr

    def test_clear_document_flagged(self):
        # Genuine document: 0.80 private vs 0.05 normal → 16× ratio
        assert _is_private_from_probs(self._probs(0.80, 0.05)) is True

    def test_borderline_vacation_photo_not_flagged(self):
        # Vacation photo that drifts up on "social media screenshot"
        # but normal prompt is still close → caught by ratio gate.
        assert _is_private_from_probs(self._probs(0.47, 0.30)) is False

    def test_absolute_threshold_enforced(self):
        # Even with infinite ratio (normal ≈ 0), below threshold = pass.
        assert _is_private_from_probs(self._probs(0.40, 0.001)) is False

    def test_ratio_gate_enforced(self):
        # Above threshold but normal is comparable → pass (not dominant)
        assert _is_private_from_probs(self._probs(0.52, 0.40)) is False

    def test_custom_threshold_and_ratio(self):
        probs = self._probs(0.55, 0.20)
        assert _is_private_from_probs(probs, threshold=0.50, ratio=1.8) is True
        assert _is_private_from_probs(probs, threshold=0.60, ratio=1.8) is False
        assert _is_private_from_probs(probs, threshold=0.50, ratio=3.0) is False


class TestDocumentEmbedding:
    def test_default_document_gate_catches_clear_documents(self):
        """Default document filtering MUST catch clear text/document hits.

        The previous defaults (threshold=0.95, ratio=50) were unreachable
        because softmax over 11 prompts dilutes each probability to 0.30–0.50,
        so almost no documents were filtered. Defaults are now 0.40 / 3.0.
        """
        emb = np.ones(512, dtype=np.float32)
        probs = np.zeros(11, dtype=float)
        probs[0] = 0.80    # strong document signal
        probs[-1] = 0.05   # weak normal signal
        with patch("src.privacy._private_vs_normal_probs", return_value=probs):
            assert is_document_from_embedding(emb) is True

    def test_default_document_gate_does_not_overflag_normal_photos(self):
        """A normal photo with weak document signals must still pass."""
        emb = np.ones(512, dtype=np.float32)
        probs = np.zeros(11, dtype=float)
        probs[0] = 0.20    # weak document signal
        probs[-1] = 0.40   # strong normal signal
        with patch("src.privacy._private_vs_normal_probs", return_value=probs):
            assert is_document_from_embedding(emb) is False


class TestIsResharedFilename:
    @pytest.mark.parametrize("name", [
        "FB_IMG_1558616482459.jpg",
        "fb_img_1234.jpg",
        "IMG-WA0001.jpg",
        "IMG-WA20230101-WA0007.jpg",
        "WhatsApp Image 2024-01-01 at 10.00.00.jpeg",
        "received_1234567890.jpeg",
        "Screenshot_20240101-103000.png",
        "Screen Shot 2024-01-01 at 10.00.00 AM.png",
        "insta_save_123.jpg",
        "telegram_photo.jpg",
    ])
    def test_reshared_names_flagged(self, name):
        assert is_reshared_filename(name) is True

    @pytest.mark.parametrize("name", [
        "1568579433476.jpg",         # WhatsApp iOS — 13-digit ms epoch
        "1568579433.jpg",            # 10-digit second epoch
        "1568579433476-2.jpg",       # ms epoch with sub-index
        "1568579433476_1.jpeg",      # ms epoch with underscore index
        "1234567890123.png",         # generic ms epoch png
    ])
    def test_numeric_messaging_names_flagged(self, name):
        """Pure-numeric 10–13 digit stems (WhatsApp/Messenger Unix-epoch
        exports) must be filtered as reshared content."""
        assert is_reshared_filename(name) is True

    def test_numeric_filter_runs_with_custom_prefixes(self):
        """Custom prefix list narrows the prefix check, but the numeric
        pattern is universal and still applies."""
        assert is_reshared_filename(
            "1568579433476.jpg", prefixes=["custom_"]
        ) is True

    @pytest.mark.parametrize("name", [
        "IMG_20190303_194319.jpg",   # legit Android camera
        "IMG_1234.HEIC",             # legit iOS camera
        "DSC00123.JPG",              # legit DSLR
        "PXL_20240101_103000.jpg",   # legit Pixel camera
        "vacation.jpg",
        "20190518152454.jpg",        # 14-digit YYYYMMDDHHMMSS — real timestamp
        "12345.jpg",                 # 5-digit frame counter
        "IMG_1568579433476.jpg",     # numeric body but has IMG_ prefix
    ])
    def test_regular_names_not_flagged(self, name):
        assert is_reshared_filename(name) is False

    def test_full_path_uses_leaf_only(self):
        assert is_reshared_filename("/foo/bar/FB_IMG_42.jpg") is True
        assert is_reshared_filename("/foo/FB_IMG_/actual_photo.jpg") is False

    def test_custom_prefix_list(self):
        assert is_reshared_filename("CUSTOM_123.jpg", prefixes=["custom_"]) is True
        # Defaults no longer apply when a custom list is provided
        assert is_reshared_filename("FB_IMG_1.jpg", prefixes=["custom_"]) is False


class TestAssessReshared:
    """Reshared-filename gate inside assess()."""

    def _img(self) -> Image.Image:
        return Image.new("RGB", (1024, 768), color=(100, 100, 100))

    def test_reshared_filename_excluded(self):
        """A path matching a reshare prefix is rejected before CLIP runs."""
        with patch("src.privacy.is_document_clip", return_value=False) as doc_check, \
             patch("src.privacy.is_document_from_embedding", return_value=False) as emb_check:
            result = assess(
                img=self._img(),
                camera_model="iPhone", has_gps=True,
                lat=37.0, lon=-122.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
                filter_reshared=True,
                path="/photos/FB_IMG_12345.jpg",
            )
        assert result is True
        # CLIP checks must be short-circuited since filename filter hit first.
        doc_check.assert_not_called()
        emb_check.assert_not_called()

    def test_reshared_disabled_allows_through(self):
        """filter_reshared=False skips the filename filter entirely."""
        with patch("src.privacy.is_document_clip", return_value=False):
            result = assess(
                img=self._img(),
                camera_model="iPhone", has_gps=True,
                lat=37.0, lon=-122.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
                filter_reshared=False,
                path="/photos/FB_IMG_12345.jpg",
            )
        assert result is False

    def test_missing_path_skips_filename_check(self):
        """Back-compat: old callers that don't pass path must still work."""
        with patch("src.privacy.is_document_clip", return_value=False):
            result = assess(
                img=self._img(),
                camera_model="iPhone", has_gps=True,
                lat=37.0, lon=-122.0, face_count=1,
                home=None, home_radius_km=0.5,
                filter_screenshots=False,
                filter_documents=True,
                filter_home_private=False,
            )
        assert result is False


class TestReassessFromCache:
    """Cache reassessment must (a) revoke stale CLIP false-positives on photos
    that look like real photos, and (b) preserve flags it cannot recompute
    from cache (notably is_screenshot, which needs original image dims)."""

    def _make_db(self, tmp_path):
        from src import database

        db_path = str(tmp_path / "test.sqlite")
        database.init_db(db_path)
        return db_path

    def _insert(
        self, conn, path, *, is_private, clip_emb=None,
        camera_model="", has_gps=0, face_count=0,
        blur_score=0.0, exposure_score=0.5,
        detail_stddev=-1.0, flesh_fraction=-1.0,
    ):
        from src import database

        emb_blob = database.emb_to_blob(
            clip_emb if clip_emb is not None else np.ones(512, dtype=np.float32)
        )
        conn.execute(
            "INSERT INTO photos (path, file_hash, is_private, clip_emb, "
            "camera_model, has_gps, face_count, blur_score, exposure_score, "
            "detail_stddev, flesh_fraction) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (path, "h_" + path, is_private, emb_blob,
             camera_model, has_gps, face_count, blur_score, exposure_score,
             detail_stddev, flesh_fraction),
        )

    def _reassess(self, conn):
        from src.privacy import reassess_is_private_from_cache

        # Patch every CLIP/text gate to "negative" so the cache-recomputable
        # signal is unambiguously False; tests then exercise the prev/preserve
        # branch without flakiness from real CLIP scoring.
        with patch("src.privacy.is_document_from_embedding", return_value=False), \
             patch("src.privacy.is_text_heavy_from_embedding", return_value=False), \
             patch("src.privacy.is_boring_from_embedding", return_value=False), \
             patch("src.privacy.is_mundane_object_from_embedding", return_value=False), \
             patch("src.privacy.is_intimate_from_embedding", return_value=False):
            return reassess_is_private_from_cache(conn)

    def test_revokes_stale_flag_on_photo_with_camera_model(self, tmp_path):
        """A previously-flagged photo with a real camera model must be cleared."""
        from src import database

        db_path = self._make_db(tmp_path)
        with database.connect(db_path) as conn:
            self._insert(conn, "/p/iphone.jpg", is_private=1,
                         camera_model="iPhone 15", has_gps=0, face_count=0)
            conn.commit()
            self._reassess(conn)
            row = conn.execute(
                "SELECT is_private FROM photos WHERE path=?",
                ("/p/iphone.jpg",),
            ).fetchone()
        assert row["is_private"] == 0

    def test_revokes_stale_flag_on_photo_with_gps(self, tmp_path):
        from src import database

        db_path = self._make_db(tmp_path)
        with database.connect(db_path) as conn:
            self._insert(conn, "/p/gps.jpg", is_private=1,
                         camera_model="", has_gps=1, face_count=0)
            conn.commit()
            self._reassess(conn)
            row = conn.execute(
                "SELECT is_private FROM photos WHERE path=?",
                ("/p/gps.jpg",),
            ).fetchone()
        assert row["is_private"] == 0

    def test_revokes_stale_flag_on_photo_with_faces(self, tmp_path):
        from src import database

        db_path = self._make_db(tmp_path)
        with database.connect(db_path) as conn:
            self._insert(conn, "/p/portrait.jpg", is_private=1,
                         camera_model="", has_gps=0, face_count=2)
            conn.commit()
            self._reassess(conn)
            row = conn.execute(
                "SELECT is_private FROM photos WHERE path=?",
                ("/p/portrait.jpg",),
            ).fetchone()
        assert row["is_private"] == 0

    def test_preserves_flag_when_no_real_photo_signal(self, tmp_path):
        """Likely-screenshot rows (no camera, no GPS, no faces) keep their
        prior flag. Original image dims aren't in cache, so we can't
        re-run is_screenshot — preservation is the safe fallback."""
        from src import database

        db_path = self._make_db(tmp_path)
        with database.connect(db_path) as conn:
            self._insert(conn, "/p/screenshot.png", is_private=1,
                         camera_model="", has_gps=0, face_count=0)
            conn.commit()
            self._reassess(conn)
            row = conn.execute(
                "SELECT is_private FROM photos WHERE path=?",
                ("/p/screenshot.png",),
            ).fetchone()
        assert row["is_private"] == 1

    def test_clean_photos_remain_clean(self, tmp_path):
        """Photos that were never flagged stay unflagged regardless of signals."""
        from src import database

        db_path = self._make_db(tmp_path)
        with database.connect(db_path) as conn:
            self._insert(conn, "/p/clean.jpg", is_private=0,
                         camera_model="", has_gps=0, face_count=0)
            conn.commit()
            self._reassess(conn)
            row = conn.execute(
                "SELECT is_private FROM photos WHERE path=?",
                ("/p/clean.jpg",),
            ).fetchone()
        assert row["is_private"] == 0

    def test_preserves_accidental_closeup_flag(self, tmp_path):
        """An accidental close-up shot HAS a camera model, but the gate must
        refire from cached flesh/blur metrics instead of being revoked."""
        from src import database

        db_path = self._make_db(tmp_path)
        with database.connect(db_path) as conn:
            self._insert(conn, "/p/closeup.jpg", is_private=1,
                         camera_model="iPhone 15", has_gps=1, face_count=0,
                         blur_score=30.0, flesh_fraction=0.80,
                         detail_stddev=20.0)
            conn.commit()
            self._reassess(conn)
            row = conn.execute(
                "SELECT is_private FROM photos WHERE path=?",
                ("/p/closeup.jpg",),
            ).fetchone()
        assert row["is_private"] == 1

    def test_preserves_pitch_black_flag(self, tmp_path):
        """A lens-cap shot HAS a camera model; the pitch-black gate must
        refire from cached exposure/detail metrics."""
        from src import database

        db_path = self._make_db(tmp_path)
        with database.connect(db_path) as conn:
            self._insert(conn, "/p/lenscap.jpg", is_private=1,
                         camera_model="Pixel 7", has_gps=1, face_count=0,
                         exposure_score=0.01, detail_stddev=0.4,
                         flesh_fraction=0.0, blur_score=5.0)
            conn.commit()
            self._reassess(conn)
            row = conn.execute(
                "SELECT is_private FROM photos WHERE path=?",
                ("/p/lenscap.jpg",),
            ).fetchone()
        assert row["is_private"] == 1

    def test_sentinel_metrics_skip_replay_and_revoke(self, tmp_path):
        """A pre-migration row (-1 sentinels) with real-photo signals follows
        the existing revocation path — the metric gates must not fire on
        sentinel values."""
        from src import database

        db_path = self._make_db(tmp_path)
        with database.connect(db_path) as conn:
            self._insert(conn, "/p/legacy.jpg", is_private=1,
                         camera_model="iPhone 15", has_gps=1, face_count=1,
                         blur_score=30.0, exposure_score=0.01,
                         detail_stddev=-1.0, flesh_fraction=-1.0)
            conn.commit()
            self._reassess(conn)
            row = conn.execute(
                "SELECT is_private FROM photos WHERE path=?",
                ("/p/legacy.jpg",),
            ).fetchone()
        assert row["is_private"] == 0

    def test_clip_hit_overrides_real_photo_signals(self, tmp_path):
        """A photo with strong real-photo signals still gets flagged when a
        current CLIP rule fires (e.g. text-heavy)."""
        from src import database
        from src.privacy import reassess_is_private_from_cache

        db_path = self._make_db(tmp_path)
        with database.connect(db_path) as conn:
            self._insert(conn, "/p/quote_meme.jpg", is_private=0,
                         camera_model="iPhone", has_gps=1, face_count=0)
            conn.commit()
            with patch("src.privacy.is_document_from_embedding", return_value=False), \
                 patch("src.privacy.is_text_heavy_from_embedding", return_value=True), \
                 patch("src.privacy.is_boring_from_embedding", return_value=False), \
                 patch("src.privacy.is_intimate_from_embedding", return_value=False):
                reassess_is_private_from_cache(conn)
            row = conn.execute(
                "SELECT is_private FROM photos WHERE path=?",
                ("/p/quote_meme.jpg",),
            ).fetchone()
        assert row["is_private"] == 1


class TestTextHeavyFromEmbedding:
    """Per-variant binary softmax must catch any text-heavy variant a real
    photo doesn't, while staying conservative on real photos."""

    def _patched_features(self, scores_by_variant: list[float], baseline: float):
        """Build a torch-mock that returns softmax probabilities for each
        binary (variant_i, baseline) pair derived from the supplied scores.

        Each per-variant binary softmax is computed in the function under
        test from ``image @ pair.T * 100``. We mock the whole CLIP pipeline
        instead — return canned text features and pretend ``image_tensor @
        pair.T`` yields raw logits we control via the inputs.
        """
        import torch

        n_variants = len(_TEXT_HEAVY_PROMPTS)
        assert len(scores_by_variant) == n_variants

        # Build (n_variants + 1) text features. Each row is unit-norm.
        # We construct features so that <image, variant_i> == score_i and
        # <image, baseline> == baseline_score.
        # Trick: pick image = e0 (first basis vector), variant_i has
        # value `score_i` in dim 0 (and tiny noise elsewhere to be unit-norm).
        dim = 8
        image = np.zeros(dim, dtype=np.float32)
        image[0] = 1.0

        feats = np.zeros((n_variants + 1, dim), dtype=np.float32)
        for i, s in enumerate(scores_by_variant):
            feats[i, 0] = s
            # Pad with epsilon to avoid divide-by-zero on norm; normalise.
            feats[i, 1] = max(0.0, 1.0 - abs(s)) ** 0.5
        feats[-1, 0] = baseline
        feats[-1, 1] = max(0.0, 1.0 - abs(baseline)) ** 0.5
        # Normalise each row.
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

        text_feats = torch.tensor(feats)

        return image, text_feats

    def test_real_photo_does_not_trigger(self):
        """All variants score below baseline → max binary prob < 0.55."""
        image, feats = self._patched_features(
            scores_by_variant=[0.20, 0.18, 0.15, 0.22, 0.10, 0.05],
            baseline=0.40,
        )
        with patch(
            "src.privacy._get_text_heavy_features",
            return_value=(feats, None, None, "cpu"),
        ):
            assert is_text_heavy_from_embedding(image) is False

    def test_printed_page_variant_triggers(self):
        """Camera shot of a printed page: dominant variant is index 1
        (``a photo of a printed page, document, notebook, or letter``).
        Old single-prompt design only checked variant 0 (chat) and missed
        this; per-variant softmax catches it."""
        scores = [0.10, 0.95, 0.10, 0.10, 0.10, 0.10]  # variant 1 dominates
        image, feats = self._patched_features(
            scores_by_variant=scores, baseline=0.05
        )
        with patch(
            "src.privacy._get_text_heavy_features",
            return_value=(feats, None, None, "cpu"),
        ):
            assert is_text_heavy_from_embedding(image) is True

    def test_calendar_page_variant_triggers(self):
        """Camera shot of a calendar/schedule (variant 2)."""
        scores = [0.10, 0.10, 0.95, 0.10, 0.10, 0.10]
        image, feats = self._patched_features(
            scores_by_variant=scores, baseline=0.05
        )
        with patch(
            "src.privacy._get_text_heavy_features",
            return_value=(feats, None, None, "cpu"),
        ):
            assert is_text_heavy_from_embedding(image) is True

    def test_chat_screenshot_still_triggers(self):
        """Backwards-compat: chat screenshot (variant 0) still works."""
        scores = [0.95, 0.10, 0.10, 0.10, 0.10, 0.10]
        image, feats = self._patched_features(
            scores_by_variant=scores, baseline=0.05
        )
        with patch(
            "src.privacy._get_text_heavy_features",
            return_value=(feats, None, None, "cpu"),
        ):
            assert is_text_heavy_from_embedding(image) is True

    def test_threshold_respected(self):
        """A variant scoring just below the threshold must NOT trigger."""
        # Pair (0.50, 0.50) → binary softmax ≈ 0.5; below 0.55 default.
        scores = [0.50, 0.10, 0.10, 0.10, 0.10, 0.10]
        image, feats = self._patched_features(
            scores_by_variant=scores, baseline=0.50
        )
        with patch(
            "src.privacy._get_text_heavy_features",
            return_value=(feats, None, None, "cpu"),
        ):
            assert is_text_heavy_from_embedding(image) is False

    def test_none_embedding_returns_false(self):
        assert is_text_heavy_from_embedding(None) is False


class TestAssessStructuralBackstop:
    """assess() must use the structural document-page check to catch
    camera photos of printed pages that CLIP misses (the user-reported
    IMG_20190125_110258.jpg case — Android camera filename, real EXIF,
    text-heavy content)."""

    def test_structural_backstop_flags_text_document(self):
        img = _img((1920, 1080))
        with patch("src.privacy.is_document_from_embedding", return_value=False), \
             patch("src.privacy.is_text_heavy_from_embedding", return_value=False), \
             patch("src.privacy.is_boring_from_embedding", return_value=False), \
             patch("src.privacy.is_intimate_from_embedding", return_value=False), \
             patch("src.privacy.looks_like_text_document_page", return_value=True):
            result = assess(
                img=img,
                camera_model="Xiaomi Redmi",
                has_gps=False,
                lat=0.0, lon=0.0, face_count=0,
                home=None, home_radius_km=0.5,
                filter_screenshots=True,
                filter_documents=True,
                filter_text_heavy=True,
                filter_home_private=False,
                path="IMG_20190125_110258.jpg",
                clip_emb=np.zeros(512, dtype=np.float32),
            )
        assert result is True

    def test_structural_backstop_does_not_flag_normal_photo(self):
        """When the structural check returns False, a normal photo with
        no other CLIP/filter hits stays unflagged."""
        img = _img((1920, 1080))
        with patch("src.privacy.is_document_from_embedding", return_value=False), \
             patch("src.privacy.is_text_heavy_from_embedding", return_value=False), \
             patch("src.privacy.is_boring_from_embedding", return_value=False), \
             patch("src.privacy.is_intimate_from_embedding", return_value=False), \
             patch("src.privacy.looks_like_text_document_page", return_value=False):
            result = assess(
                img=img,
                camera_model="Xiaomi Redmi",
                has_gps=False,
                lat=0.0, lon=0.0, face_count=0,
                home=None, home_radius_km=0.5,
                filter_screenshots=True,
                filter_documents=True,
                filter_text_heavy=True,
                filter_home_private=False,
                path="IMG_20190125_110258.jpg",
                clip_emb=np.zeros(512, dtype=np.float32),
            )
        assert result is False

    def test_structural_backstop_disabled_when_text_heavy_off(self):
        """Both the CLIP text-heavy gate and the structural backstop are
        controlled by ``filter_text_heavy``. Off → neither runs."""
        img = _img((1920, 1080))
        with patch("src.privacy.is_document_from_embedding", return_value=False), \
             patch("src.privacy.is_boring_from_embedding", return_value=False), \
             patch("src.privacy.is_intimate_from_embedding", return_value=False), \
             patch("src.privacy.looks_like_text_document_page", return_value=True) as struct:
            result = assess(
                img=img,
                camera_model="Xiaomi Redmi",
                has_gps=False,
                lat=0.0, lon=0.0, face_count=0,
                home=None, home_radius_km=0.5,
                filter_screenshots=True,
                filter_documents=True,
                filter_text_heavy=False,
                filter_home_private=False,
                path="IMG_20190125_110258.jpg",
                clip_emb=np.zeros(512, dtype=np.float32),
            )
        assert result is False
        struct.assert_not_called()


# ---------------------------------------------------------------------------
# is_accidental_closeup tests
# ---------------------------------------------------------------------------

from src.privacy import is_accidental_closeup


class TestIsAccidentalCloseup:
    def test_flesh_heavy_and_blurry_is_closeup(self):
        assert is_accidental_closeup(flesh_fraction=0.80, blur_score=40.0) is True

    def test_flesh_heavy_but_sharp_is_not_closeup(self):
        # Sharp portrait close-up: high flesh but well-focused → keep.
        assert is_accidental_closeup(flesh_fraction=0.80, blur_score=500.0) is False

    def test_low_flesh_is_not_closeup(self):
        assert is_accidental_closeup(flesh_fraction=0.20, blur_score=10.0) is False

    def test_threshold_boundaries_inclusive(self):
        assert is_accidental_closeup(0.65, 120.0, 0.65, 120.0) is True
        assert is_accidental_closeup(0.6499, 120.0, 0.65, 120.0) is False
        assert is_accidental_closeup(0.65, 120.01, 0.65, 120.0) is False


class TestAssessAccidentalCloseup:
    def test_assess_flags_accidental_closeup(self):
        img = _img((1000, 1000))  # dims not a screen res
        flagged = assess(
            img=img, camera_model="Pixel 7", has_gps=True,
            lat=12.0, lon=77.0, face_count=0, home=None, home_radius_km=0.5,
            filter_screenshots=False, filter_documents=False,
            filter_text_heavy=False, filter_boring_objects=False,
            filter_intimate_content=False,
            filter_accidental_closeup=True,
            flesh_fraction=0.80, blur_score=30.0,
        )
        assert flagged is True

    def test_assess_keeps_sharp_portrait(self):
        img = _img((1000, 1000))
        flagged = assess(
            img=img, camera_model="Pixel 7", has_gps=True,
            lat=12.0, lon=77.0, face_count=1, home=None, home_radius_km=0.5,
            filter_screenshots=False, filter_documents=False,
            filter_text_heavy=False, filter_boring_objects=False,
            filter_intimate_content=False,
            filter_accidental_closeup=True,
            flesh_fraction=0.80, blur_score=600.0,
        )
        assert flagged is False

    def test_assess_gate_disabled(self):
        img = _img((1000, 1000))
        flagged = assess(
            img=img, camera_model="Pixel 7", has_gps=True,
            lat=12.0, lon=77.0, face_count=0, home=None, home_radius_km=0.5,
            filter_screenshots=False, filter_documents=False,
            filter_text_heavy=False, filter_boring_objects=False,
            filter_intimate_content=False,
            filter_accidental_closeup=False,
            flesh_fraction=0.99, blur_score=1.0,
        )
        assert flagged is False

    def test_assess_default_blur_inf_never_trips(self):
        # Caller omits blur_score → default +inf → gate can't fire even on flesh.
        img = _img((1000, 1000))
        flagged = assess(
            img=img, camera_model="Pixel 7", has_gps=True,
            lat=12.0, lon=77.0, face_count=0, home=None, home_radius_km=0.5,
            filter_screenshots=False, filter_documents=False,
            filter_text_heavy=False, filter_boring_objects=False,
            filter_intimate_content=False,
            filter_accidental_closeup=True,
            flesh_fraction=0.99,
        )
        assert flagged is False


# ---------------------------------------------------------------------------
# is_pitch_black_or_pure_white tests
# ---------------------------------------------------------------------------

from src.privacy import is_pitch_black_or_pure_white


class TestIsPitchBlackOrPureWhite:
    def test_pitch_black_frame_flagged(self):
        assert is_pitch_black_or_pure_white(exposure_score=0.01, detail_stddev=0.5) is True

    def test_pure_white_frame_flagged(self):
        assert is_pitch_black_or_pure_white(exposure_score=0.99, detail_stddev=0.5) is True

    def test_dark_but_textured_night_scene_kept(self):
        # City lights at night: dark mean but real tonal detail.
        assert is_pitch_black_or_pure_white(exposure_score=0.04, detail_stddev=25.0) is False

    def test_bright_snow_field_kept(self):
        assert is_pitch_black_or_pure_white(exposure_score=0.96, detail_stddev=18.0) is False

    def test_normal_exposure_flat_frame_kept(self):
        # Flat but mid-gray (e.g. fog) — not pitch black / pure white.
        assert is_pitch_black_or_pure_white(exposure_score=0.50, detail_stddev=1.0) is False

    def test_threshold_boundaries(self):
        # detail must be strictly < 2.0; exposure strictly outside [0.05, 0.95].
        assert is_pitch_black_or_pure_white(0.05, 1.0) is False
        assert is_pitch_black_or_pure_white(0.95, 1.0) is False
        assert is_pitch_black_or_pure_white(0.04, 2.0) is False


class TestAssessPitchBlack:
    def _assess(self, **kw):
        img = _img((1000, 1000))
        defaults = dict(
            img=img, camera_model="Pixel 7", has_gps=True,
            lat=12.0, lon=77.0, face_count=0, home=None, home_radius_km=0.5,
            filter_screenshots=False, filter_documents=False,
            filter_text_heavy=False, filter_boring_objects=False,
            filter_intimate_content=False, filter_accidental_closeup=False,
        )
        defaults.update(kw)
        return assess(**defaults)

    def test_assess_flags_pitch_black(self):
        assert self._assess(
            filter_pitch_black=True, exposure_score=0.01, detail_stddev=0.3,
        ) is True

    def test_assess_flags_pure_white(self):
        assert self._assess(
            filter_pitch_black=True, exposure_score=0.99, detail_stddev=0.3,
        ) is True

    def test_assess_gate_disabled(self):
        assert self._assess(
            filter_pitch_black=False, exposure_score=0.01, detail_stddev=0.3,
        ) is False

    def test_assess_default_exposure_never_trips(self):
        # Caller omits exposure_score → neutral 0.5 default → gate can't fire
        # even when detail_stddev is low.
        assert self._assess(
            filter_pitch_black=True, detail_stddev=0.0,
        ) is False
