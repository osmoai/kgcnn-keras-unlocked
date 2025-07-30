#!/usr/bin/env python3
"""
Script to fix duplicate compile sections and add balanced accuracy metrics
"""

import re

def fix_compile_sections():
    # Read the file
    with open('keras-gcn-descs.py', 'r') as f:
        content = f.read()
    
    # First, remove any duplicate compile sections
    # Pattern to match consecutive compile sections
    pattern = r'("compile":\s*{[^}]+})\s*,\s*("compile":\s*{[^}]+})'
    
    def remove_duplicate_compile(match):
        # Keep the second one (more complete) and remove the first
        return match.group(2)
    
    content = re.sub(pattern, remove_duplicate_compile, content)
    
    # Now add metrics to compile sections that don't have them
    # Pattern for compile sections without metrics
    pattern_no_metrics = r'("compile":\s*{\s*"optimizer":\s*{[^}]+}(?:,\s*"loss":\s*"[^"]+")?\s*})'
    
    def add_metrics(match):
        compile_section = match.group(1)
        
        # Check if metrics already exist
        if '"metrics":' in compile_section:
            return compile_section
        
        # Add metrics before the closing brace
        if '"loss":' in compile_section:
            # Has loss, add metrics after loss
            return compile_section.replace('}', ', "metrics": ["accuracy", "balanced_accuracy", "precision", "recall"]}')
        else:
            # Only has optimizer, add metrics after optimizer
            return compile_section.replace('}', ', "metrics": ["accuracy", "balanced_accuracy", "precision", "recall"]}')
    
    content = re.sub(pattern_no_metrics, add_metrics, content)
    
    # Write back to file
    with open('keras-gcn-descs.py', 'w') as f:
        f.write(content)
    
    print("Fixed compile sections and added balanced accuracy metrics!")

if __name__ == "__main__":
    fix_compile_sections() 