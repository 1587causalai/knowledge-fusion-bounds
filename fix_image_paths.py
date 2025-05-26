#!/usr/bin/env python3
"""
Fix image paths in Markdown files for GitHub Pages compatibility.

This script converts relative image paths to absolute paths from project root,
ensuring images display correctly both locally and on GitHub Pages.
"""

import os
import re
import glob
from pathlib import Path

def fix_image_paths_in_file(file_path):
    """Fix image paths in a single Markdown file."""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Get the directory of the current file relative to project root
    file_dir = os.path.dirname(file_path)
    
    # Pattern to match image references: ![alt text](path)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif|svg))\)'
    
    def replace_path(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        
        # Skip if already absolute path from root (starts with docs/ or results/)
        if image_path.startswith(('docs/', 'results/', '/docs/', '/results/')):
            # Ensure no leading slash for GitHub Pages
            clean_path = image_path.lstrip('/')
            return f'![{alt_text}]({clean_path})'
        
        # Handle relative paths
        if image_path.startswith('../'):
            # Calculate the absolute path from project root
            # For files in docs/visualizations/, ../assets/ becomes docs/assets/
            if 'docs/visualizations' in file_dir or 'docs/experiments' in file_dir:
                if image_path.startswith('../assets/'):
                    new_path = image_path.replace('../assets/', 'docs/assets/')
                    return f'![{alt_text}]({new_path})'
            
            # For other relative paths, try to resolve them
            try:
                # Resolve relative path to absolute path from project root
                current_dir = Path(file_dir)
                image_full_path = current_dir / image_path
                # Get path relative to project root
                relative_to_root = os.path.relpath(image_full_path, '.')
                return f'![{alt_text}]({relative_to_root})'
            except:
                print(f"  Warning: Could not resolve path {image_path}")
                return match.group(0)
        
        # For paths that don't start with ../ but are not absolute
        # Check if they need docs/ or results/ prefix
        if not image_path.startswith(('docs/', 'results/')):
            # If file is in docs/ directory and path doesn't start with docs/
            if file_dir.startswith('docs/') and not image_path.startswith('docs/'):
                # Try to find the image in docs/assets/
                if os.path.exists(f'docs/assets/{os.path.basename(image_path)}'):
                    new_path = f'docs/assets/{os.path.basename(image_path)}'
                    return f'![{alt_text}]({new_path})'
            
            # If image exists in results/
            if os.path.exists(f'results/{os.path.basename(image_path)}'):
                new_path = f'results/{os.path.basename(image_path)}'
                return f'![{alt_text}]({new_path})'
        
        return match.group(0)
    
    # Apply the replacements
    content = re.sub(image_pattern, replace_path, content)
    
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
    """Main function to fix all Markdown files."""
    print("Fixing image paths in Markdown files...")
    print("=" * 50)
    
    # Find all Markdown files
    md_files = []
    
    # Root level MD files
    md_files.extend(glob.glob("*.md"))
    
    # MD files in docs/ and subdirectories
    md_files.extend(glob.glob("docs/**/*.md", recursive=True))
    
    # MD files in other directories
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in root:
            continue
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                if file_path not in md_files:
                    md_files.append(file_path)
    
    # Remove duplicates and sort
    md_files = sorted(list(set(md_files)))
    
    print(f"Found {len(md_files)} Markdown files:")
    for f in md_files:
        print(f"  - {f}")
    print()
    
    # Process each file
    updated_count = 0
    for md_file in md_files:
        if fix_image_paths_in_file(md_file):
            updated_count += 1
    
    print("=" * 50)
    print(f"Processing complete!")
    print(f"Updated {updated_count} out of {len(md_files)} files.")
    
    # Provide summary of expected paths
    print("\nExpected image path format after fix:")
    print("  - For images in docs/assets/: docs/assets/image.png")
    print("  - For images in results/: results/image.png")
    print("  - No leading slashes (for GitHub Pages compatibility)")

if __name__ == "__main__":
    main() 