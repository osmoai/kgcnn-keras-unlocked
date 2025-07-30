#!/usr/bin/env python3
"""
Script to add balanced accuracy metrics to all compile sections in keras-gcn-descs.py
"""

import re

def update_compile_sections():
    # Read the file
    with open('keras-gcn-descs.py', 'r') as f:
        content = f.read()
    
    # Pattern to match compile sections that don't already have metrics
    # This matches compile sections that end with just the optimizer
    pattern1 = r'("compile":\s*{\s*"optimizer":\s*{[^}]+}\s*})'
    
    # Pattern to match compile sections that have loss but no metrics
    pattern2 = r'("compile":\s*{\s*"optimizer":\s*{[^}]+},\s*"loss":\s*"[^"]+"\s*})'
    
    # Pattern to match compile sections that have optimizer and loss but no metrics
    pattern3 = r'("compile":\s*{\s*"optimizer":\s*{[^}]+},\s*"loss":\s*"[^"]+",\s*"metrics":\s*\[[^\]]+\]\s*})'
    
    # Function to add metrics to a compile section
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
    
    # Apply the replacement
    updated_content = re.sub(pattern1, add_metrics, content)
    updated_content = re.sub(pattern2, add_metrics, updated_content)
    
    # Write back to file
    with open('keras-gcn-descs.py', 'w') as f:
        f.write(updated_content)
    
    print("Updated compile sections with balanced accuracy metrics!")

if __name__ == "__main__":
    update_compile_sections() 