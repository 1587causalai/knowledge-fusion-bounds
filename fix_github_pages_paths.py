#!/usr/bin/env python3
"""
Fix image paths in simplified HTML <img> tags for GitHub Pages deployment.
Ensures <img src="path" alt="alt text"> uses paths like 'docs/assets/image.png'.
"""

import os
import re
import glob

def fix_html_paths_for_github(file_path):
    """Fix image paths in simplified HTML <img> tags for GitHub Pages compatibility."""
    print(f"Processing for GitHub Pages (HTML <img>): {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match simplified HTML img tags: <img src="path" alt="text">
    img_pattern = r'(<img\s+src=")([^"]+)("\s+alt="[^"]*">)'
    
    def replace_path_for_github(match):
        tag_start = match.group(1)  # <img src="
        src_path = match.group(2)   # path
        tag_end = match.group(3)    # " alt="...">
        new_src_path = src_path     # Default to no change

        if file_path.startswith('docs/'):
            if src_path.startswith('../assets/'):
                new_src_path = src_path.replace('../assets/', 'docs/assets/')
            elif src_path.startswith('../../assets/'): # Should not happen with correct local paths
                 new_src_path = src_path.replace('../../assets/', 'docs/assets/')
            elif src_path.startswith('assets/'): # For files directly in docs/ (e.g. docs/README.md)
                new_src_path = f"docs/{src_path}"
            elif src_path.startswith('../results/'):
                new_src_path = src_path.replace('../results/', 'results/')
            elif src_path.startswith('../../results/'):
                 new_src_path = src_path.replace('../../results/', 'results/')
            # results/ path from docs/ is already correct for GitHub if it means root /results
        
        else: # File is in project root
            # Paths like 'docs/assets/...' or 'results/...' are already correct
            pass

        if new_src_path != src_path:
            return f'{tag_start}{new_src_path}{tag_end}'
        return match.group(0) # Return original full match if no change

    content = re.sub(img_pattern, replace_path_for_github, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ GitHub HTML <img> paths updated: {file_path}")
        return True
    else:
        print(f"  - No GitHub HTML <img> path changes needed: {file_path}")
        return False

def main():
    """Main function to fix HTML <img> paths for GitHub Pages."""
    print("Fixing simplified HTML <img> image paths for GitHub Pages...")
    print("=" * 60)
    
    md_files = glob.glob("docs/**/*.md", recursive=True)
    root_md_files = glob.glob("*.md")
    # Exclude DEPLOYMENT_GUIDE.md to prevent its example paths from being changed
    all_files = sorted([f for f in list(set(md_files + root_md_files)) if not f.endswith('DEPLOYMENT_GUIDE.md')])
    
    print(f"Found {len(all_files)} Markdown files to process:")
    for f in all_files:
        print(f"  - {f}")
    print()
    
    updated_count = 0
    for md_file in all_files:
        if fix_html_paths_for_github(md_file):
            updated_count += 1
            
    print("=" * 60)
    print(f"GitHub Pages HTML <img> path fixing complete!")
    print(f"Updated {updated_count} out of {len(all_files)} files.")

if __name__ == "__main__":
    main() 