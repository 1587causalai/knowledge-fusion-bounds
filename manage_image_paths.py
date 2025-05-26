#!/usr/bin/env python3
"""
Unified script to manage image paths for both local Docsify development and GitHub Pages deployment.
Usage:
    python manage_image_paths.py local    # Fix paths for local Docsify development
    python manage_image_paths.py github   # Fix paths for GitHub Pages deployment
"""

import sys
import subprocess
import os

def run_script(script_name):
    """Run a Python script and return the result."""
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python manage_image_paths.py [local|github]")
        print()
        print("Commands:")
        print("  local   - Fix image paths for local Docsify development")
        print("  github  - Fix image paths for GitHub Pages deployment")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "local":
        print("🔧 Switching to LOCAL DOCSIFY mode...")
        print("This will update image paths for local development with 'docsify serve docs'")
        print()
        if run_script("fix_local_docsify_paths.py"):
            print()
            print("✅ Successfully configured for local Docsify development!")
            print("💡 You can now run: docsify serve docs")
        else:
            print("❌ Failed to configure for local development")
            sys.exit(1)
            
    elif mode == "github":
        print("🚀 Switching to GITHUB PAGES mode...")
        print("This will update image paths for GitHub Pages deployment")
        print()
        if run_script("fix_github_pages_paths.py"):
            print()
            print("✅ Successfully configured for GitHub Pages!")
            print("💡 You can now commit and push to deploy to GitHub Pages")
        else:
            print("❌ Failed to configure for GitHub Pages")
            sys.exit(1)
            
    else:
        print(f"❌ Unknown mode: {mode}")
        print("Use 'local' or 'github'")
        sys.exit(1)

if __name__ == "__main__":
    main() 