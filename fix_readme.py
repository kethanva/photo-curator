import re

with open("README.md", "r") as f:
    text = f.read()

# Remove the thick ======== borders around ## headers
text = re.sub(r'={50,}\n(##.*?)\n={50,}', r'\1', text)

# Remove the thick ======== borders around empty or other lines
text = re.sub(r'={50,}\n', '\n', text)

# Replace the ------- borders around subsection headers with ###
# Format:
# -----------------------------...
# 2.1  SYSTEM CONTEXT
# -----------------------------...
text = re.sub(r'-{50,}\n(\d+\.\d+\s+.*?)\n-{50,}', r'### \1', text)

# Also remove standalone '-----' lines that might cause issues, unless in a table
text = re.sub(r'\n-{50,}\n', '\n\n', text)

with open("README.md", "w") as f:
    f.write(text)
