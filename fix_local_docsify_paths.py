#!/usr/bin/env python3
"""
Fix image paths in simplified HTML <img> tags for local Docsify development.
Ensures <img src="path" alt="alt text"> uses relative paths like '../assets/image.png'.
"""

import os
import re
import glob

def fix_html_paths_for_local(file_path):
    """Fix image paths in simplified HTML <img> tags for local Docsify compatibility."""
    print(f"Processing for Local Docsify (HTML <img>): {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match simplified HTML img tags: <img src="path" alt="text">
    img_pattern = r'(<img\s+src=")([^"]+)("\s+alt="[^"]*">)'
    
    def replace_path_for_local(match):
        tag_start = match.group(1)  # <img src="
        src_path = match.group(2)   # path
        tag_end = match.group(3)    # " alt="...">
        new_src_path = src_path     # Default to no change
        
        if file_path.startswith('docs/'):
            file_dir_depth = len(file_path.split('/')) - 1 # docs/a.md depth 1, docs/sub/a.md depth 2

            if src_path.startswith('docs/assets/'):
                if file_dir_depth == 1: # File is in docs/ (e.g., docs/README.md)
                    new_src_path = src_path.replace('docs/assets/', 'assets/')
                else: # File is in docs/sub/
                    new_src_path = src_path.replace('docs/assets/', '../assets/')
            elif src_path.startswith('assets/'): # Path is like assets/img.png
                # If file is in docs/sub/ and path is assets/, it means it was probably meant to be ../assets/
                if file_dir_depth > 1 and not src_path.startswith('../'): 
                    new_src_path = f"../{src_path}" 
            elif src_path.startswith('results/'): # Path is like results/img.png
                if file_dir_depth == 1: # docs/file.md linking to results/
                    new_src_path = f"../{src_path}"
                elif file_dir_depth > 1: # docs/sub/file.md linking to results/
                    new_src_path = f"../../{src_path}"
        
        else: # File is in project root (e.g., experiments.md)
            # For root files, paths like 'docs/assets/...' or 'results/...' are expected for GitHub pages.
            # For local, root files usually don't need Docsify path transformation if served from root.
            pass
            
        if new_src_path != src_path:
            return f'{tag_start}{new_src_path}{tag_end}'
        return match.group(0)

    content = re.sub(img_pattern, replace_path_for_local, content)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Local HTML <img> paths updated: {file_path}")
        return True
    else:
        print(f"  - No Local HTML <img> path changes needed: {file_path}")
        return False

def main():
    """Main function to fix HTML <img> paths for local Docsify."""
    print("Fixing simplified HTML <img> image paths for Local Docsify...")
    print("=" * 60)
    
    md_files = glob.glob("docs/**/*.md", recursive=True)
    root_md_files = glob.glob("*.md")
    all_files = sorted([f for f in list(set(md_files + root_md_files)) if not f.endswith('DEPLOYMENT_GUIDE.md')])
    
    print(f"Found {len(all_files)} Markdown files to process:")
    for f in all_files:
        print(f"  - {f}")
    print()
    
    updated_count = 0
    for md_file in all_files:
        if fix_html_paths_for_local(md_file):
            updated_count += 1
            
    print("=" * 60)
    print(f"Local Docsify HTML <img> path fixing complete!")
    print(f"Updated {updated_count} out of {len(all_files)} files.")

if __name__ == "__main__":
    main() 