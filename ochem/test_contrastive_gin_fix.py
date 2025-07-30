#!/usr/bin/env python3
"""
Test script to verify that ContrastiveGIN model works with the fix.
"""

import os
import sys
import configparser
import subprocess

def test_contrastive_gin():
    """Test ContrastiveGIN model with the fix."""
    
    # Create a minimal config for testing
    config_content = """[Task]
train_mode = False
model_file = model.tar
train_data_file = ../../data/odorless_train_desc.csv
apply_data_file = ../../data/odorless_test_desc.csv
result_file = ../../data/test_contrastive_gin_results.csv

[Details]
batch = 32
architecture_name = ContrastiveGIN
nbepochs = 1
lr = 0.001
gpu = -1
seed = 10666
output_dim = 1
activation = sigmoid
lossdef = BCEmask
classification = True
overwrite = True

# Descriptor parameters
use_descriptors = True
descriptor_columns = desc0, desc1

[Models]
architectures = ContrastiveGIN
"""
    
    # Write test config
    with open("test_config.cfg", "w") as f:
        f.write(config_content)
    
    print("🧪 Testing ContrastiveGIN model...")
    
    # Run the keras script with test config
    try:
        cmd = [sys.executable, "keras-gcn-descs.py", "test_config.cfg"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ ContrastiveGIN test PASSED!")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print("❌ ContrastiveGIN test FAILED!")
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False
    finally:
        # Clean up test config
        if os.path.exists("test_config.cfg"):
            os.remove("test_config.cfg")

if __name__ == "__main__":
    test_contrastive_gin() 