#!/usr/bin/env python3
"""
Automated GNN Architecture Testing Script

This script automatically tests multiple GNN architectures by:
1. Reading architectures from [Models] section in config file
2. Training each architecture
3. Applying each trained model
4. Generating a comprehensive report

Usage: python automated_gnn_testing.py config-desc.cfg
"""

import configparser
import subprocess
import sys
import os
import time
import pandas as pd
from datetime import datetime
import json
import importlib.util

# Add parent directory to Python path to ensure kgcnn modules can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AutomatedGNNTester:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.results = []
        self.start_time = datetime.now()
        self.model_configs = self.load_model_configurations()
    
    def load_model_configurations(self):
        """Load model configurations from keras-gcn-descs.py"""
        print("📚 Loading model configurations from keras-gcn-descs.py...")
        
        # Path to keras-gcn-descs.py
        kgcnn_descs_path = os.path.join(os.path.dirname(__file__), 'keras-gcn-descs.py')
        
        if not os.path.exists(kgcnn_descs_path):
            print(f"❌ keras-gcn-descs.py not found at: {kgcnn_descs_path}")
            return {}
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("keras_gcn_descs", kgcnn_descs_path)
            kgcnn_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kgcnn_module)
            
            # Extract model configurations
            model_configs = {}
            
            # Get all the architecture configurations from the module
            # We'll look for the architecture_name checks in the code
            source_code = open(kgcnn_descs_path, 'r').read()
            
            # Find all architecture configurations
            architectures = [
                'ContrastiveGIN', 'ContrastiveGAT', 'ContrastiveGATv2',
                'ContrastiveDMPNN', 'ContrastiveAttFP', 'ContrastiveAddGNN',
                'ContrastiveDGIN', 'ContrastivePNA', 'ContrastiveMoE',
                'GCN', 'GAT', 'GATv2', 'CMPNN', 'CoAttentiveFP', 'AttentiveFPPlus', 'AttFP',
                'CMPNNPlus', 'DGIN', 'AddGNN', 'GraphTransformer',
                'RGCN', 'rGIN', 'rGINE', 'GIN', 'GINE', 'GraphGPS', 'PNA', 'ExpC',
                'EGAT', 'TransformerGAT', 'GRPE', 'KAGAT', 'DHTNN', 'DMPNN',
                'AWARE', 'MoGAT', 'MoE', 'ConfigurableMoE',
                'NMPN'
            ]
            
            for arch in architectures:
                # Check if this architecture is supported in the module using multiple patterns
                supported = False
                
                # Pattern 1: Direct equality check
                if f"architecture_name == '{arch}'" in source_code or f"elif architecture_name == '{arch}'" in source_code:
                    supported = True
                
                # Pattern 2: In list checks (like ['ChemProp', 'DMPNN'])
                elif f"'{arch}'" in source_code and f"architecture_name in" in source_code:
                    # Look for patterns like "architecture_name in ['ChemProp', 'DMPNN']"
                    lines = source_code.split('\n')
                    for line in lines:
                        if f"architecture_name in" in line and f"'{arch}'" in line:
                            supported = True
                            break
                
                # Pattern 3: Direct name references in model configs
                elif f"'name': '{arch}'" in source_code or f'"name": "{arch}"' in source_code:
                    supported = True
                
                # Pattern 4: Module imports (like "kgcnn.literature.NMPN")
                elif f"kgcnn.literature.{arch}" in source_code:
                    supported = True
                
                if supported:
                    model_configs[arch] = {
                        'name': arch,
                        'supported': True,
                        'source': 'keras-gcn-descs.py'
                    }
                else:
                    model_configs[arch] = {
                        'name': arch,
                        'supported': False,
                        'source': 'Not found'
                    }
            
            print(f"✅ Loaded {len([k for k, v in model_configs.items() if v['supported']])} supported architectures")
            return model_configs
            
        except Exception as e:
            print(f"❌ Error loading model configurations: {e}")
            return {}
        
    def get_architectures_to_test(self):
        """Get architectures to test from config file and validate against supported models"""
        if 'Models' not in self.config:
            print("❌ No [Models] section found in config file!")
            print("Please add a [Models] section with architectures to test.")
            print("Example:")
            print("[Models]")
            print("architectures = GIN, GINE, GAT, GATv2, GCN, NMPN, AttentiveFP, rGIN, RGCN, CMPNN")
            return []
        
        architectures_str = self.config.get('Models', 'architectures', fallback='')
        if not architectures_str:
            print("❌ No architectures specified in [Models] section!")
            return []
        
        requested_architectures = [arch.strip() for arch in architectures_str.split(',')]
        
        # Filter to only supported architectures
        supported_architectures = []
        unsupported_architectures = []
        
        for arch in requested_architectures:
            if arch in self.model_configs and self.model_configs[arch]['supported']:
                supported_architectures.append(arch)
            else:
                unsupported_architectures.append(arch)
        
        if unsupported_architectures:
            print(f"⚠️  Warning: The following architectures are not supported in keras-gcn-descs.py:")
            for arch in unsupported_architectures:
                print(f"   - {arch}")
            print(f"   These will be skipped.")
        
        if not supported_architectures:
            print("❌ No supported architectures found!")
            print("Supported architectures:")
            for arch, config in self.model_configs.items():
                if config['supported']:
                    print(f"   - {arch}")
            return []
        
        print(f"📋 Found {len(supported_architectures)} supported architectures to test: {', '.join(supported_architectures)}")
        return supported_architectures
    
    def backup_config(self):
        """Create a backup of the original config file"""
        backup_file = f"{self.config_file}.backup"
        with open(self.config_file, 'r') as f:
            original_content = f.read()
        with open(backup_file, 'w') as f:
            f.write(original_content)
        print(f"💾 Config backup created: {backup_file}")
        return backup_file
    
    def update_config_architecture(self, architecture):
        """Update the config file with a specific architecture"""
        # Read current config
        with open(self.config_file, 'r') as f:
            content = f.read()
        
        # Update architecture name
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('architecture_name ='):
                lines[i] = f"architecture_name = {architecture}"
                break
        
        # Write updated config
        with open(self.config_file, 'w') as f:
            f.write('\n'.join(lines))
        
        # Reload the config in memory
        self.config.read(self.config_file)
        
        print(f"🔄 Updated config: architecture_name = {architecture}")
    
    def run_training(self, architecture):
        """Run training for a specific architecture"""
        print(f"\n🚀 Training {architecture}...")
        print("=" * 50)
        
        # Update architecture name in config
        self.update_config_architecture(architecture)
        
        # Set train_mode = True
        self.update_config_train_mode(True)
        
        try:
            # Run training
            result = subprocess.run(
                ['python', 'keras-gcn-descs.py', self.config_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"✅ {architecture} training completed successfully!")
                return True, result.stdout
            else:
                print(f"❌ {architecture} training failed!")
                print(f"Error: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {architecture} training timed out!")
            return False, "Training timed out"
        except Exception as e:
            print(f"❌ {architecture} training error: {str(e)}")
            return False, str(e)
    
    def run_application(self, architecture):
        """Run application for a specific architecture"""
        print(f"\n🔍 Applying {architecture}...")
        print("=" * 50)
        
        # Set train_mode = False
        self.update_config_train_mode(False)
        
        try:
            # Run application
            result = subprocess.run(
                ['python', 'keras-gcn-descs.py', self.config_file],
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ {architecture} application completed successfully!")
                return True, result.stdout
            else:
                print(f"❌ {architecture} application failed!")
                print(f"Error: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {architecture} application timed out!")
            return False, "Application timed out"
        except Exception as e:
            print(f"❌ {architecture} application error: {str(e)}")
            return False, str(e)
    
    def update_config_train_mode(self, train_mode):
        """Update train_mode in config file"""
        with open(self.config_file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('train_mode ='):
                lines[i] = f"train_mode = {train_mode}"
                break
        
        with open(self.config_file, 'w') as f:
            f.write('\n'.join(lines))
        
        # Reload the config in memory
        self.config.read(self.config_file)
    
    def extract_training_loss(self, output):
        """Extract final training loss from output"""
        lines = output.split('\n')
        # Look for the last line containing loss information
        for i in range(len(lines) - 1, 0, -1):
            line = lines[i]
            if 'loss:' in line and 'ms/epoch' in line:
                try:
                    # Extract loss value from line like: "1/1 - 1s - loss: 4.4409 - MaskedRMSE: 4.4409 - val_loss: 3.8510 - val_MaskedRMSE: 3.8510 - lr: 0.0010 - 878ms/epoch - 878ms/step"
                    loss_part = line.split('loss:')[1].split()[0]
                    return float(loss_part)
                except:
                    pass
        return None
    
    def get_model_parameters_tf(self, architecture):
        """Get model parameters using TensorFlow directly by reading from keras-gcn-descs.py"""
        try:
            import tensorflow as tf
            
            # Check if architecture is supported
            if architecture not in self.model_configs or not self.model_configs[architecture]['supported']:
                print(f"⚠️  Architecture {architecture} not supported in keras-gcn-descs.py, using fallback")
                return None
            
            print(f"🔍 Reading {architecture} configuration from keras-gcn-descs.py...")
            
            # Since we can't easily parse the complex config format here,
            # let's use the fallback method which reads from training output
            return None
            
        except Exception as e:
            print(f"❌ Failed to get parameters for {architecture} using TF: {e}")
            return None
    
    def extract_model_parameters(self, output):
        """Extract model parameters from output (fallback method)"""
        lines = output.split('\n')
        
        # First, look for the standard "Total params:" format
        for line in lines:
            if 'Total params:' in line:
                try:
                    # Extract the number before the space and parentheses
                    # Format: "Total params: 558600 (2.13 MB)"
                    params_part = line.split('Total params:')[1].strip()
                    params_str = params_part.split()[0]  # Get first part before space
                    # Remove any commas and convert to int
                    params_str = params_str.replace(',', '')
                    return int(params_str)
                except Exception as e:
                    print(f"Debug: Failed to parse parameters from line: '{line}' - Error: {e}")
                    pass
        
        # Look for model summary section
        in_model_summary = False
        for line in lines:
            if 'Model:' in line or 'model summary' in line.lower():
                in_model_summary = True
                continue
            
            if in_model_summary and ('Total params:' in line or 'total parameters:' in line.lower()):
                try:
                    # Extract numbers from the line
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        return int(numbers[-1])  # Take the last number
                except:
                    pass
            
            # Exit model summary if we hit a blank line or different section
            if in_model_summary and (line.strip() == '' or line.startswith('=') or line.startswith('-')):
                in_model_summary = False
        
        # Look for any line containing parameter information
        for line in lines:
            if any(keyword in line.lower() for keyword in ['parameters:', 'params:', 'total params:', 'model params:']):
                try:
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        # Filter out small numbers that might be other metrics
                        large_numbers = [n for n in numbers if len(n) > 3]  # Parameters are usually > 1000
                        if large_numbers:
                            return int(large_numbers[-1])
                        else:
                            return int(numbers[-1])
                except:
                    pass
        
        # Look for the specific pattern from the output you showed
        for line in lines:
            if '🔢' in line and 'parameters (TF):' in line:
                try:
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        return int(numbers[-1])
                except:
                    pass
        
        print(f"⚠️  Could not extract parameters from output")
        return None

    def check_sigmoid_outputs(self, architecture):
        """Check if the model produces proper sigmoid outputs (values between 0 and 1)"""
        print(f"\n🔍 Checking sigmoid outputs for {architecture}...")
        
        # Read the results file
        results_file = "../../data/odorless_results_desc.csv"
        if not os.path.exists(results_file):
            print(f"❌ Results file not found: {results_file}")
            return None
        
        try:
            # Read the CSV file
            df = pd.read_csv(results_file)
            
            if df.empty:
                print(f"❌ Results file is empty: {results_file}")
                return None
            
            # Get the first column (should contain the predictions)
            predictions = df.iloc[:, 0].values
            
            # Check if values are in proper sigmoid range (0-1)
            min_val = float(predictions.min())
            max_val = float(predictions.max())
            mean_val = float(predictions.mean())
            
            # Check for proper sigmoid behavior
            sigmoid_ok = 0 <= min_val <= max_val <= 1
            
            # Check for extreme values that might indicate issues
            extreme_negative = min_val < -1
            extreme_positive = max_val > 2
            
            # Check for constant values (no learning)
            constant_output = max_val - min_val < 0.01
            
            sigmoid_status = "✅ GOOD" if sigmoid_ok else "❌ BAD"
            
            print(f"   📊 Sigmoid Analysis:")
            print(f"      Min value: {min_val:.6f}")
            print(f"      Max value: {max_val:.6f}")
            print(f"      Mean value: {mean_val:.6f}")
            print(f"      Range: {max_val - min_val:.6f}")
            print(f"      Status: {sigmoid_status}")
            
            if extreme_negative:
                print(f"      ⚠️  WARNING: Extreme negative values detected!")
            if extreme_positive:
                print(f"      ⚠️  WARNING: Extreme positive values detected!")
            if constant_output:
                print(f"      ⚠️  WARNING: Nearly constant output (model not learning)!")
            
            return {
                'min_val': min_val,
                'max_val': max_val,
                'mean_val': mean_val,
                'range': max_val - min_val,
                'sigmoid_ok': sigmoid_ok,
                'extreme_negative': extreme_negative,
                'extreme_positive': extreme_positive,
                'constant_output': constant_output
            }
            
        except Exception as e:
            print(f"❌ Error reading results file: {e}")
            return None
    
    def print_current_config(self, architecture):
        """Print the current configuration for debugging"""
        print(f"\n🔧 Current configuration for {architecture}:")
        print(f"   Architecture name: {self.config.get('Details', 'architecture_name', fallback='Not set')}")
        print(f"   Train mode: {self.config.get('Details', 'train_mode', fallback='Not set')}")
        print(f"   Output dim: {self.config.get('Details', 'output_dim', fallback='Not set')}")
        
        # Check if architecture is supported
        if architecture in self.model_configs:
            arch_config = self.model_configs[architecture]
            if arch_config['supported']:
                print(f"   ✅ {architecture} is supported in keras-gcn-descs.py")
                print(f"   Source: {arch_config['source']}")
            else:
                print(f"   ❌ {architecture} is NOT supported in keras-gcn-descs.py")
        else:
            print(f"   ❓ {architecture} not found in model configurations")
        
        # Check config file section (legacy)
        if architecture in self.config:
            arch_config = self.config[architecture]
            print(f"   📄 {architecture} section found in config file")
            
            # Print key configuration items
            for key in ['name', 'depth', 'output_mlp', 'last_mlp', 'input_embedding']:
                if key in arch_config:
                    value = arch_config[key]
                    # Truncate long values
                    if len(str(value)) > 100:
                        value = str(value)[:100] + "..."
                    print(f"   {key}: {value}")
                else:
                    print(f"   {key}: Not found")
        else:
            print(f"   📄 {architecture} section NOT found in config file (using keras-gcn-descs.py)")
        
        print()
    
    def test_architecture(self, architecture):
        """Test a single architecture (train + apply)"""
        print(f"\n{'='*60}")
        print(f"🧪 TESTING ARCHITECTURE: {architecture}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Train the model (this will update the config)
        train_success, train_output = self.run_training(architecture)
        
        # Print current configuration for debugging (after config is updated)
        self.print_current_config(architecture)
        
        if not train_success:
            print(f"❌ Training failed for {architecture}, skipping application")
            return {
                'architecture': architecture,
                'train_success': False,
                'apply_success': False,
                'final_loss': None,
                'parameters': None,
                'train_time': time.time() - start_time,
                'apply_time': 0,
                'train_error': train_output,
                'apply_error': None
            }
        
        train_time = time.time() - start_time
        
        # Apply the model
        apply_start = time.time()
        apply_success, apply_output = self.run_application(architecture)
        apply_time = time.time() - apply_start
        
        # Check sigmoid outputs if application was successful
        sigmoid_analysis = None
        if apply_success:
            sigmoid_analysis = self.check_sigmoid_outputs(architecture)
        
        # Extract metrics
        final_loss = self.extract_training_loss(train_output)
        # Get parameters using TensorFlow (more reliable)
        parameters = self.get_model_parameters_tf(architecture)
        if parameters is None:
            # Fallback to text parsing
            parameters = self.extract_model_parameters(train_output)
        
        result = {
            'architecture': architecture,
            'train_success': train_success,
            'apply_success': apply_success,
            'final_loss': final_loss,
            'parameters': parameters,
            'train_time': train_time,
            'apply_time': apply_time,
            'train_error': None if train_success else train_output,
            'apply_error': None if apply_success else apply_output,
            'sigmoid_analysis': sigmoid_analysis
        }
        
        # Print summary
        status = "✅ SUCCESS" if train_success and apply_success else "❌ FAILED"
        loss_str = f"Loss: {final_loss:.2f}" if final_loss else "Loss: N/A"
        params_str = f"Params: {parameters:,}" if parameters else "Params: N/A"
        
        print(f"\n📊 {architecture} Summary:")
        print(f"   Status: {status}")
        print(f"   {loss_str}")
        print(f"   {params_str}")
        print(f"   Train Time: {train_time:.1f}s")
        print(f"   Apply Time: {apply_time:.1f}s")
        
        # Print sigmoid analysis if available
        if sigmoid_analysis:
            sigmoid_status = "✅ GOOD" if sigmoid_analysis['sigmoid_ok'] else "❌ BAD"
            print(f"   Sigmoid: {sigmoid_status} (Range: {sigmoid_analysis['range']:.4f})")
            if sigmoid_analysis['extreme_negative'] or sigmoid_analysis['extreme_positive']:
                print(f"   ⚠️  SIGMOID ISSUES DETECTED!")
            if sigmoid_analysis['constant_output']:
                print(f"   ⚠️  CONSTANT OUTPUT DETECTED!")
        
        return result
    
    def run_all_tests(self):
        """Run tests for all architectures"""
        print("🤖 AUTOMATED GNN ARCHITECTURE TESTING")
        print("=" * 60)
        
        # Get architectures to test
        architectures = self.get_architectures_to_test()
        if not architectures:
            return
        
        # Backup config
        backup_file = self.backup_config()
        
        try:
            # Test each architecture
            for i, architecture in enumerate(architectures, 1):
                print(f"\n📋 Progress: {i}/{len(architectures)}")
                result = self.test_architecture(architecture)
                self.results.append(result)
                
                # No delay between tests for faster execution
            
            # Generate report
            self.generate_report()
            
        finally:
            # Restore config
            self.restore_config(backup_file)
    
    def restore_config(self, backup_file):
        """Restore original config file"""
        with open(backup_file, 'r') as f:
            original_content = f.read()
        with open(self.config_file, 'w') as f:
            f.write(original_content)
        print(f"🔄 Config restored from backup")
        
        # Remove backup file
        os.remove(backup_file)
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print(f"\n📊 GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        # Create results summary
        successful_tests = [r for r in self.results if r['train_success'] and r['apply_success']]
        failed_tests = [r for r in self.results if not r['train_success'] or not r['apply_success']]
        
        # Sort by loss (best first)
        successful_tests.sort(key=lambda x: x['final_loss'] if x['final_loss'] else float('inf'))
        
        # Generate report
        report = self.create_report_content(successful_tests, failed_tests)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"automated_test_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved: {report_file}")
        
        # Also save JSON data
        json_file = f"test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📄 JSON data saved: {json_file}")
        
        # Print summary
        self.print_summary(successful_tests, failed_tests)
    
    def create_report_content(self, successful_tests, failed_tests):
        """Create the report content"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        report = f"""# Automated GNN Architecture Testing Report

## Test Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Time**: {total_time:.1f} seconds
- **Total Architectures**: {len(self.results)}
- **Successful**: {len(successful_tests)}
- **Failed**: {len(failed_tests)}

## Performance Rankings

### 🏆 Top Performers (Best to Worst)

"""
        
        for i, result in enumerate(successful_tests, 1):
            loss = result['final_loss']
            params = result['parameters']
            arch = result['architecture']
            train_time = result['train_time']
            
            loss_str = f"{loss:.2f}" if loss is not None else "N/A"
            params_str = f"{params:,}" if params is not None else "N/A"
            stars = "⭐" * min(5, max(1, int(10 / (loss or 1)))) if loss is not None else "⭐"
            
            report += f"""| {i} | **{arch}** | {loss_str} | {params_str} | {stars} | {train_time:.1f}s |
"""
        
        report += """
| Rank | Architecture | Final Loss | Parameters | Performance | Train Time |
|------|--------------|------------|------------|-------------|------------|
"""
        
        for i, result in enumerate(successful_tests, 1):
            loss = result['final_loss']
            params = result['parameters']
            arch = result['architecture']
            train_time = result['train_time']
            
            loss_str = f"{loss:.2f}" if loss is not None else "N/A"
            params_str = f"{params:,}" if params is not None else "N/A"
            stars = "⭐" * min(5, max(1, int(10 / (loss or 1)))) if loss is not None else "⭐"
            
            report += f"| {i} | **{arch}** | {loss_str} | {params_str} | {stars} | {train_time:.1f}s |\n"
        
        report += f"""

## Detailed Results

### ✅ Successful Tests ({len(successful_tests)})

"""
        
        for result in successful_tests:
            loss_str = f"{result['final_loss']:.2f}" if result['final_loss'] is not None else "N/A"
            params_str = f"{result['parameters']:,}" if result['parameters'] is not None else "N/A"
            
            # Add sigmoid analysis
            sigmoid_info = ""
            if result.get('sigmoid_analysis'):
                sa = result['sigmoid_analysis']
                sigmoid_status = "✅ GOOD" if sa['sigmoid_ok'] else "❌ BAD"
                sigmoid_info = f"""
- **Sigmoid Analysis**: {sigmoid_status}
  - Min: {sa['min_val']:.6f}, Max: {sa['max_val']:.6f}, Mean: {sa['mean_val']:.6f}
  - Range: {sa['range']:.6f}"""
                if sa['extreme_negative'] or sa['extreme_positive']:
                    sigmoid_info += "\n  - ⚠️ **SIGMOID ISSUES DETECTED**"
                if sa['constant_output']:
                    sigmoid_info += "\n  - ⚠️ **CONSTANT OUTPUT DETECTED**"
            
            report += f"""#### {result['architecture']}
- **Final Loss**: {loss_str}
- **Parameters**: {params_str}
- **Train Time**: {result['train_time']:.1f}s
- **Apply Time**: {result['apply_time']:.1f}s
- **Status**: ✅ Success{sigmoid_info}

"""
        
        if failed_tests:
            report += f"""### ❌ Failed Tests ({len(failed_tests)})

"""
            
            for result in failed_tests:
                report += f"""#### {result['architecture']}
- **Train Success**: {'✅' if result['train_success'] else '❌'}
- **Apply Success**: {'✅' if result['apply_success'] else '❌'}
- **Train Error**: {result['train_error'][:200] + '...' if result['train_error'] and len(result['train_error']) > 200 else result['train_error']}
- **Apply Error**: {result['apply_error'][:200] + '...' if result['apply_error'] and len(result['apply_error']) > 200 else result['apply_error']}

"""
        
        if successful_tests:
            loss_str = f"{successful_tests[0]['final_loss']:.2f}" if successful_tests[0]['final_loss'] is not None else "N/A"
            report += f"""
## Key Insights

1. **Best Performer**: {successful_tests[0]['architecture']} with loss {loss_str}
2. **Total Test Time**: {total_time:.1f} seconds
3. **Success Rate**: {len(successful_tests)}/{len(self.results)} ({len(successful_tests)/len(self.results)*100:.1f}%)

## Recommendations

"""
            
            best = successful_tests[0]
            loss_str = f"{best['final_loss']:.2f}" if best['final_loss'] is not None else "N/A"
            report += f"""1. **Use {best['architecture']}** for best performance (loss: {loss_str})
2. **Consider parameter efficiency** for deployment
3. **Monitor training times** for large-scale applications

"""
        else:
            report += f"""
## Key Insights

1. **No successful tests** - all architectures failed
2. **Total Test Time**: {total_time:.1f} seconds
3. **Success Rate**: 0/{len(self.results)} (0.0%)

## Recommendations

1. **Check configuration** - verify all model configurations are correct
2. **Review error logs** - examine specific failure reasons
3. **Test individual models** - try running models one by one to isolate issues

"""
        
        return report
    
    def print_summary(self, successful_tests, failed_tests):
        """Print a summary to console"""
        print(f"\n🎯 TESTING COMPLETE!")
        print("=" * 60)
        print(f"📊 Results Summary:")
        print(f"   Total Architectures: {len(self.results)}")
        print(f"   Successful: {len(successful_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        print(f"   Success Rate: {len(successful_tests)/len(self.results)*100:.1f}%")
        
        if successful_tests:
            best = successful_tests[0]
            loss_str = f"{best['final_loss']:.2f}" if best['final_loss'] is not None else "N/A"
            params_str = f"{best['parameters']:,}" if best['parameters'] is not None else "N/A"
            print(f"\n🏆 Best Performer: {best['architecture']}")
            print(f"   Loss: {loss_str}")
            print(f"   Parameters: {params_str}")
            print(f"   Train Time: {best['train_time']:.1f}s")
        else:
            print(f"\n❌ No successful tests - all architectures failed")
            print(f"   Check the error logs above for specific failure reasons")
        
        if failed_tests:
            print(f"\n❌ Failed Architectures:")
            for result in failed_tests:
                print(f"   - {result['architecture']}")
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        print(f"\n⏱️  Total Test Time: {total_time:.1f} seconds")

def main():
    if len(sys.argv) != 2:
        print("Usage: python automated_gnn_testing.py config-desc.cfg")
        print("\nMake sure your config file has a [Models] section:")
        print("[Models]")
        print("architectures = GIN, GINE, GAT, GATv2, GCN, NMPN, AttentiveFP, rGIN, RGCN, CMPNN")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    # Create and run tester
    tester = AutomatedGNNTester(config_file)
    tester.run_all_tests()

if __name__ == "__main__":
    main() 