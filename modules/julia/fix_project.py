import re

with open('modules/julia/Project.toml', 'r') as f:
    content = f.read()

# 1. Standardize UUIDs and sections
# Correct Test UUID to standard Julia v1.x (8dfed614-e22c-11e8-a48c-94047424e788)
# MerkabahCY Project.toml already has this in my recent write, but let's be sure.

# 2. In Julia, standard libraries like Test and LinearAlgebra
# SHOULD be in [deps] if the package uses them.
# The error "expected package Test to be registered" often means
# it's in [deps] but not in the registry, but for standard libs
# they ARE in the default registry.

# Wait, if Test is a standard library, it should definitely work if it's in [deps].
# Let's try to remove it from [extras] and put it ONLY in [deps].

# Remove from extras if it exists
content = re.sub(r'\[extras\]\nTest = ".*"\n', '[extras]\n', content)

# Add to deps if not already there
if 'Test = "8dfed614-e22c-11e8-a48c-94047424e788"' not in content:
    content = content.replace('[deps]\n', '[deps]\nTest = "8dfed614-e22c-11e8-a48c-94047424e788"\n')

with open('modules/julia/Project.toml', 'w') as f:
    f.write(content)
