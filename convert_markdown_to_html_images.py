#!/usr/bin/env python3
"""
Convert Markdown image syntax to simplified HTML <img> tags.
Convert from: ![alt text](path/to/image.png)
Convert to: <img src="path/to/image.png" alt="alt text">
"""

import os
import re
import glob

def convert_markdown_images(file_path):
    """Convert Markdown image syntax to HTML <img> tags."""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match Markdown image syntax: ![alt text](path)
    markdown_img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_markdown_img(match):
        alt_text = match.group(1)
        src_path = match.group(2)
        # Return HTML img tag
        return f'<img src="{src_path}" alt="{alt_text}">'
    
    content = re.sub(markdown_img_pattern, replace_markdown_img, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Converted Markdown images to HTML: {file_path}")
        return True
    else:
        print(f"  - No Markdown images found: {file_path}")
        return False

def main():
    """Main function to convert Markdown images to HTML."""
    print("Converting Markdown image syntax to HTML <img> tags...")
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
        if convert_markdown_images(md_file):
            updated_count += 1
            
    print("=" * 60)
    print(f"Markdown to HTML image conversion complete!")
    print(f"Updated {updated_count} out of {len(all_files)} files.")

if __name__ == "__main__":
    main() 