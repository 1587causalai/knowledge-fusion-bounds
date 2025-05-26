#!/usr/bin/env python3
"""
Verify that all images referenced in Markdown files exist.
"""

import os
import re
import glob

def verify_images():
    """Verify all image references in Markdown files."""
    print("Verifying image paths in Markdown files...")
    print("=" * 50)
    
    # Find all Markdown files
    md_files = []
    md_files.extend(glob.glob("*.md"))
    md_files.extend(glob.glob("docs/**/*.md", recursive=True))
    
    # Remove duplicates and sort
    md_files = sorted(list(set(md_files)))
    
    missing_images = []
    total_images = 0
    
    # Pattern to match image references
    image_pattern = r'!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif|svg))\)'
    
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all image references
            matches = re.findall(image_pattern, content)
            
            if matches:
                print(f"\n📄 {md_file}:")
                
                for alt_text, image_path in matches:
                    total_images += 1
                    
                    # Check if image exists
                    if os.path.exists(image_path):
                        print(f"  ✅ {image_path}")
                    else:
                        print(f"  ❌ {image_path} (MISSING)")
                        missing_images.append((md_file, image_path))
        
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Verification complete!")
    print(f"Total images found: {total_images}")
    print(f"Missing images: {len(missing_images)}")
    
    if missing_images:
        print("\n❌ Missing images:")
        for md_file, image_path in missing_images:
            print(f"  {md_file} → {image_path}")
        return False
    else:
        print("\n✅ All images exist!")
        return True

if __name__ == "__main__":
    success = verify_images()
    exit(0 if success else 1) 