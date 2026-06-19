import os
import re

# Paths to the three requirements.txt files
root_path = r"c:\Users\saib\Documents\testing ai\colab_ragflow\RAG\combine\requirements.txt"
rag_system_path = r"c:\Users\saib\Documents\testing ai\colab_ragflow\RAG\combine\RAG_system\requirements.txt"
final_rag_path = r"c:\Users\saib\Documents\testing ai\colab_ragflow\RAG\combine\final_rag\requirements.txt"

paths = [root_path, rag_system_path, final_rag_path]

# Dictionary to hold the most specific requirement for each package name
merged_requirements = {}

def parse_line(line):
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None
    # Match package name (word characters, dash, underscore, dot) optionally followed by version specifiers
    match = re.match(r'^([a-zA-Z0-9_\-\.]+)(.*)$', line)
    if match:
        package_name = match.group(1).lower().replace('_', '-')
        specifier = match.group(2).strip()
        return package_name, specifier
    return None, None

for path in paths:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                pkg_name, spec = parse_line(line)
                if pkg_name:
                    # If we don't have it yet, or the new one has a more specific specifier
                    if pkg_name not in merged_requirements:
                        merged_requirements[pkg_name] = (pkg_name, spec)
                    else:
                        existing_name, existing_spec = merged_requirements[pkg_name]
                        # If existing has no specifier but new one does, upgrade
                        if not existing_spec and spec:
                            merged_requirements[pkg_name] = (pkg_name, spec)
                        # Keep version specification with exact matching == or inequality >= if preferred
                        elif '==' in spec and '==' not in existing_spec:
                            merged_requirements[pkg_name] = (pkg_name, spec)
                        elif '>=' in spec and not existing_spec:
                            merged_requirements[pkg_name] = (pkg_name, spec)

# Format the merged requirements sorted alphabetically
final_lines = []
for pkg_name in sorted(merged_requirements.keys()):
    pkg, spec = merged_requirements[pkg_name]
    # Keep the original casing or capitalizations if nice, or standard lower
    # Let's map back to standard capitalization for well-known packages
    display_name = pkg
    if pkg == 'speechrecognition':
        display_name = 'SpeechRecognition'
    elif pkg == 'fastapi':
        display_name = 'fastapi'
    elif pkg == 'uvicorn':
        display_name = 'uvicorn'
    elif pkg == 'pydantic':
        display_name = 'pydantic'
    elif pkg == 'requests':
        display_name = 'requests'
    elif pkg == 'certifi':
        display_name = 'certifi'
    elif pkg == 'pymongo':
        display_name = 'pymongo'
    elif pkg == 'bcrypt':
        display_name = 'bcrypt'
    elif pkg == 'beautifulsoup4':
        display_name = 'beautifulsoup4'
    elif pkg == 'huggingface-hub':
        display_name = 'huggingface-hub'
    elif pkg == 'numpy':
        display_name = 'numpy'
    elif pkg == 'pandas':
        display_name = 'pandas'
    elif pkg == 'tiktoken':
        display_name = 'tiktoken'
    elif pkg == 'transformers':
        display_name = 'transformers'
    
    final_lines.append(f"{display_name}{spec}")

# Print the merged requirements
print("Merged Requirements count:", len(final_lines))
for line in final_lines:
    print(line)

# Write to the root requirements.txt
with open(root_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(final_lines) + '\n')

print("Wrote merged requirements to root.")

# Remove RAG_system/requirements.txt and final_rag/requirements.txt
for sub_path in [rag_system_path, final_rag_path]:
    if os.path.exists(sub_path):
        os.remove(sub_path)
        print(f"Deleted sub-requirements file: {sub_path}")
