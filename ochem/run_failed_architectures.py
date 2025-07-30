#!/usr/bin/env python3
"""
Script to rerun only the architectures that previously failed during inference.
This tests if the shape mismatch fixes work.
"""

import os
import sys
import configparser
import subprocess
from datetime import datetime

# List of architectures that previously failed
FAILED_ARCHITECTURES = [
    "MoE",                    # test inference
    "ContrastiveGIN",         # test inference  
    "ContrastiveGAT",         # test inference
    "ContrastiveGATv2",       # test inference
    "ContrastiveDMPNN",       # training
    "ContrastiveAttFP",       # training
    "ContrastiveAddGNN",      # training
    "ContrastivePNA",         # training
    "DMPNNAttention",         # test inference
    "rGINE",                  # test inference
    "NMPN",                   # training
    "GRPE"                    # test inference
]

def modify_config_file(config_path, architecture, train_mode, result_file, apply_data_file=None):
    """
    Modify the config file for a specific architecture.
    
    Args:
        config_path (str): Path to the config file
        architecture (str): Architecture name
        train_mode (str): "True" for training, "False" for inference
        result_file (str): Path to the result file
        apply_data_file (str): Path to the apply data file (for inference)
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Update architecture and training mode
    config['Details']['architecture_name'] = architecture
    config['Details']['train_mode'] = train_mode
    config['Task']['result_file'] = result_file
    
    # Update apply_data_file for inference
    if apply_data_file:
        config['Task']['apply_data_file'] = apply_data_file
    
    # Write back to config file
    with open(config_path, 'w') as f:
        config.write(f)
    
    print(f"✅ Modified config: architecture={architecture}, train_mode={train_mode}, result_file={result_file}" + 
          (f", apply_data_file={apply_data_file}" if apply_data_file else ""))

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
    """Main function to run failed architectures."""
    
    # Configuration
    config_path = "config-desc-odorless.cfg"
    data_folder = "../../data"
    train_data_file = "../../data/odorless_train_desc.csv"
    test_data_file = "../../data/odorless_test_desc.csv"
    
    # Ensure data folder exists
    os.makedirs(data_folder, exist_ok=True)
    
    print("=" * 80)
    print("🔄 RERUNNING PREVIOUSLY FAILED ARCHITECTURES")
    print("=" * 80)
    print(f"📁 Config file: {config_path}")
    print(f"📁 Data folder: {data_folder}")
    print(f"🏗️  Testing {len(FAILED_ARCHITECTURES)} architectures")
    print("=" * 80)
    
    successful_architectures = []
    failed_architectures = []
    
    for i, architecture in enumerate(FAILED_ARCHITECTURES, 1):
        print(f"\n{'='*60}")
        print(f"🏗️  Testing architecture {i}/{len(FAILED_ARCHITECTURES)}: {architecture}")
        print(f"{'='*60}")
        
        # Step 1: Training
        print(f"📚 Step 1: Training {architecture}")
        train_result_file = f"{data_folder}/odorless_results_train_desc_{architecture}.csv"
        modify_config_file(config_path, architecture, "True", train_result_file)
        
        if run_keras_script(config_path):
            print(f"✅ Training completed for {architecture}")
            
            # Step 2: Test Inference
            print(f"🔮 Step 2: Inference on test data for {architecture}")
            test_result_file = f"{data_folder}/odorless_results_test_desc_{architecture}.csv"
            modify_config_file(config_path, architecture, "False", test_result_file, test_data_file)
            
            if run_keras_script(config_path):
                print(f"✅ Test inference completed for {architecture}")
                
                # Step 3: Train Inference
                print(f"🔮 Step 3: Inference on train data for {architecture}")
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
    
    # Final summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY - FIXED ARCHITECTURES")
    print(f"{'='*80}")
    
    if successful_architectures:
        print(f"✅ Successful architectures ({len(successful_architectures)}):")
        for arch in successful_architectures:
            print(f"   - {arch}")
    else:
        print("❌ No architectures succeeded")
    
    if failed_architectures:
        print(f"\n❌ Still failed architectures ({len(failed_architectures)}):")
        for arch in failed_architectures:
            print(f"   - {arch}")
    else:
        print("\n🎉 All previously failed architectures are now working!")
    
    print(f"\n📁 Results saved in: {data_folder}")
    print(f"📁 File naming pattern: odorless_results_[train|test]_desc_[architecture].csv")
    print(f"📁 Config file used: {config_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 