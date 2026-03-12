import re

with open('modules/julia/Project.toml', 'r') as f:
    content = f.read()

# Remove Test from deps if present
content = re.sub(r'Test = ".*"\n', '', content)

# Add Test to deps with the standard UUID
if '[deps]' in content:
    content = content.replace('[deps]\n', '[deps]\nTest = "8dfed614-e22c-11e8-a48c-94047424e788"\n')

# Ensure Test is NOT in extras
content = re.sub(r'Test = ".*"\n', '', content, flags=re.MULTILINE) # This might be too aggressive

with open('modules/julia/Project.toml', 'w') as f:
    f.write(content)
