#!/usr/bin/env python3
"""
Fix image paths for local Docsify development.

This script converts GitHub Pages paths back to relative paths for local Docsify.
"""

import os
import re
import glob

def fix_local_paths_in_file(file_path):
    """Fix image paths in HTML img tags for local Docsify."""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match HTML img tags with src attribute
    img_pattern = r'<img\s+src="([^"]+)"([^>]*)>'
    
    def replace_local_path(match):
        src_path = match.group(1)
        other_attrs = match.group(2)
        
        # Determine the correct path based on file location
        if file_path.startswith('docs/'):
            # For files inside docs/ directory
            file_dir = os.path.dirname(file_path)
            
            if 'docs/assets/' in src_path:
                # Convert docs/assets/ to ../assets/ for local Docsify
                new_path = src_path.replace('docs/assets/', '../assets/')
                return f'<img src="{new_path}"{other_attrs}>'
            elif src_path.startswith('results/'):
                # Convert results/ to ../results/ for docs/ subdirectories
                if file_dir != 'docs':  # Not in docs root
                    new_path = f'../{src_path}'
                    return f'<img src="{new_path}"{other_attrs}>'
                else:  # In docs root
                    new_path = f'../{src_path}'
                    return f'<img src="{new_path}"{other_attrs}>'
        
        else:
            # For root level files, convert to docs/assets/ format
            if src_path.startswith('../assets/'):
                new_path = src_path.replace('../assets/', 'docs/assets/')
                return f'<img src="{new_path}"{other_attrs}>'
            elif src_path.startswith('../results/'):
                new_path = src_path.replace('../results/', 'results/')
                return f'<img src="{new_path}"{other_attrs}>'
        
        return match.group(0)
    
    # Apply the replacements
    content = re.sub(img_pattern, replace_local_path, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Updated {file_path}")
        return True
    else:
        print(f"  - No changes needed for {file_path}")
        return False

def main():
    """Main function to fix local Docsify paths."""
    print("Fixing image paths for local Docsify development...")
    print("=" * 60)
    
    # Find all Markdown files
    md_files = glob.glob("docs/**/*.md", recursive=True)
    root_md_files = glob.glob("*.md")
    all_files = sorted(list(set(md_files + root_md_files)))
    
    print(f"Found {len(all_files)} Markdown files:")
    for f in all_files:
        print(f"  - {f}")
    print()
    
    # Process each file
    updated_count = 0
    for md_file in all_files:
        if fix_local_paths_in_file(md_file):
            updated_count += 1
    
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Updated {updated_count} out of {len(all_files)} files.")
    
    print("\nLocal Docsify path format:")
    print("  - docs/subdirectory files: ../assets/image.png")
    print("  - docs/subdirectory files (results): ../results/image.png")
    print("  - docs/ root files: assets/image.png")
    print("  - root project files: docs/assets/image.png, results/image.png")

if __name__ == "__main__":
    main() 