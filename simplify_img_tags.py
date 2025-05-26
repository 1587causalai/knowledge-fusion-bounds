#!/usr/bin/env python3
"""
Simplify HTML <img> tags by removing complex style attributes.
Convert from: <img src="path" alt="text" style="...complex styles...">
Convert to: <img src="path" alt="text">
"""

import os
import re
import glob

def simplify_img_tags(file_path):
    """Simplify HTML <img> tags by removing style attributes."""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match HTML img tags with style attributes
    # Captures: <img src="..." alt="..." style="...">
    img_pattern = r'<img\s+src="([^"]+)"\s+alt="([^"]*)"\s+style="[^"]*">'
    
    def replace_img_tag(match):
        src_path = match.group(1)
        alt_text = match.group(2)
        # Return simplified img tag
        return f'<img src="{src_path}" alt="{alt_text}">'
    
    content = re.sub(img_pattern, replace_img_tag, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Simplified img tags in: {file_path}")
        return True
    else:
        print(f"  - No changes needed: {file_path}")
        return False

def main():
    """Main function to simplify HTML <img> tags."""
    print("Simplifying HTML <img> tags...")
    print("=" * 60)
    
    # Find all markdown files
    md_files = glob.glob("docs/**/*.md", recursive=True)
    root_md_files = glob.glob("*.md")
    all_files = sorted(list(set(md_files + root_md_files)))
    
    print(f"Found {len(all_files)} Markdown files to process:")
    for f in all_files:
        print(f"  - {f}")
    print()
    
    updated_count = 0
    for md_file in all_files:
        if simplify_img_tags(md_file):
            updated_count += 1
            
    print("=" * 60)
    print(f"HTML <img> tag simplification complete!")
    print(f"Updated {updated_count} out of {len(all_files)} files.")

if __name__ == "__main__":
    main() 