#!/usr/bin/env python3
"""
Script to run training and inference for all architectures one by one.
This script modifies the config file for each architecture and saves results with architecture-specific names.

For each architecture, the script:
1. Trains the model using the train data
2. Runs inference on test data (odorless_test_desc.csv)
3. Runs inference on train data (odorless_train_desc.csv)

Results are saved as:
- odorless_results_train_desc_[architecture].csv (inference on train data)
- odorless_results_test_desc_[architecture].csv (inference on test data)
"""

import os
import sys
import subprocess
import configparser
import time
from pathlib import Path

def modify_config_file(config_path, architecture_name, train_mode, result_file, apply_data_file=None):
    """
    Modify the config file with new architecture name, train mode, result file, and apply data file.
    
    Args:
        config_path (str): Path to the config file
        architecture_name (str): Name of the architecture to use
        train_mode (str): "True" for training, "False" for inference
        result_file (str): Path to the result file
        apply_data_file (str): Path to the apply data file (for inference)
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Modify the configuration
    config['Task']['train_mode'] = train_mode
    config['Task']['result_file'] = result_file
    config['Details']['architecture_name'] = architecture_name
    
    # Modify apply_data_file if provided (for inference)
    if apply_data_file is not None:
        config['Task']['apply_data_file'] = apply_data_file
    
    # Write the modified config back to file
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    if apply_data_file:
        print(f"✅ Modified config: architecture={architecture_name}, train_mode={train_mode}, result_file={result_file}, apply_data_file={apply_data_file}")
    else:
        print(f"✅ Modified config: architecture={architecture_name}, train_mode={train_mode}, result_file={result_file}")

def run_keras_script(config_path):
    """
    Run the keras-gcn-descs.py script with the given config file.
    
    Args:
        config_path (str): Path to the config file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the directory where the config file is located
        script_dir = os.path.dirname(os.path.abspath(config_path))
        script_name = "keras-gcn-descs.py"
        config_name = os.path.basename(config_path)
        
        # Use the current Python interpreter (should be conda if activated)
        python_cmd = sys.executable
        
        cmd = [python_cmd, script_name, config_name]
        print(f"🚀 Running: {' '.join(cmd)}")
        print(f"📁 Working directory: {script_dir}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode == 0:
            print("✅ Script completed successfully")
            return True
        else:
            print(f"❌ Script failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

def main():
    # Configuration
    config_path = "config-desc-odorless.cfg"
    data_folder = "../../data"
    
    # Data file paths
    train_data_file = f"{data_folder}/odorless_train_desc.csv"
    test_data_file = f"{data_folder}/odorless_test_desc.csv"
    
    # Ensure data folder exists
    os.makedirs(data_folder, exist_ok=True)
    
    # List of all architectures to test
    architectures = [
        "CoAttentiveFP", "AttFP", "MoE", "ContrastiveGIN", "ContrastiveGAT", 
        "ContrastiveGATv2", "ContrastiveDMPNN", "ContrastiveAttFP", 
        "ContrastiveAddGNN", "ContrastivePNA", "DMPNNAttention", "GAT", 
        "GIN", "GINE", "rGIN", "rGINE", "RGCN", "NMPN", "CMPNN", "DGIN", 
        "DMPNN", "AddGNN", "PNA", "GRPE"
    ]
    
    print(f"🎯 Starting batch processing for {len(architectures)} architectures")
    print(f"📁 Config file: {config_path}")
    print(f"📁 Data folder: {data_folder}")
    print("=" * 80)
    
    successful_architectures = []
    failed_architectures = []
    
    for i, architecture in enumerate(architectures, 1):
        print(f"\n{'='*60}")
        print(f"🏗️  Processing architecture {i}/{len(architectures)}: {architecture}")
        print(f"{'='*60}")
        
        # Create architecture-specific result file names
        train_result_file = f"{data_folder}/odorless_results_train_desc_{architecture}.csv"
        test_result_file = f"{data_folder}/odorless_results_test_desc_{architecture}.csv"
        
        try:
            # Step 1: Training
            print(f"\n📚 Step 1: Training {architecture}")
            modify_config_file(config_path, architecture, "True", train_result_file)
            
            if run_keras_script(config_path):
                print(f"✅ Training completed for {architecture}")
                
                # Step 2: Inference on test data
                print(f"\n🔮 Step 2: Inference on test data for {architecture}")
                modify_config_file(config_path, architecture, "False", test_result_file, test_data_file)
                
                if run_keras_script(config_path):
                    print(f"✅ Test inference completed for {architecture}")
                    
                    # Step 3: Inference on train data
                    print(f"\n🔮 Step 3: Inference on train data for {architecture}")
                    modify_config_file(config_path, architecture, "False", train_result_file, train_data_file)
                    
                    if run_keras_script(config_path):
                        print(f"✅ Train inference completed for {architecture}")
                        successful_architectures.append(architecture)
                    else:
                        print(f"❌ Train inference failed for {architecture}")
                        failed_architectures.append(f"{architecture} (train inference)")
                else:
                    print(f"❌ Test inference failed for {architecture}")
                    failed_architectures.append(f"{architecture} (test inference)")
            else:
                print(f"❌ Training failed for {architecture}")
                failed_architectures.append(f"{architecture} (training)")
                
        except Exception as e:
            print(f"❌ Error processing {architecture}: {e}")
            failed_architectures.append(f"{architecture} (error)")
        
        # Small delay between architectures
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful architectures ({len(successful_architectures)}):")
    for arch in successful_architectures:
        print(f"   - {arch}")
    
    if failed_architectures:
        print(f"\n❌ Failed architectures ({len(failed_architectures)}):")
        for arch in failed_architectures:
            print(f"   - {arch}")
    else:
        print("\n🎉 All architectures completed successfully!")
    
    print(f"\n📁 Results saved in: {data_folder}")
    print(f"📁 File naming pattern: odorless_results_[train|test]_desc_[architecture].csv")
    print(f"📁 Config file used: {config_path}")
    print("=" * 80)

if __name__ == "__main__":
    main() 