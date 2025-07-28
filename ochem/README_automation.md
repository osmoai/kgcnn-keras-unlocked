# 🤖 Automated GNN Architecture Testing

This automation system allows you to test multiple GNN architectures automatically by simply listing them in your config file.

## 🚀 Quick Start

### 1. Add [Models] Section to Config

Add this section to your `config-desc.cfg` file:

```ini
[Models]
# List of architectures to test automatically
# Separate multiple architectures with commas
architectures = GIN, GINE, GAT, GATv2, GCN, NMPN, AttentiveFP, rGIN, RGCN, CMPNN
```

### 2. Run Automated Testing

```bash
python automated_gnn_testing.py config-desc.cfg
```

### 3. Get Results

The script will automatically:
- ✅ Train each architecture
- ✅ Apply each trained model
- ✅ Generate comprehensive reports
- ✅ Restore your original config

## 📋 Available Architectures

| Architecture | Description | Status |
|--------------|-------------|---------|
| **GIN** | Graph Isomorphism Network | ✅ Supported |
| **GINE** | GIN with Edge features | ✅ Supported |
| **GAT** | Graph Attention Network | ✅ Supported |
| **GATv2** | Improved Graph Attention Network | ✅ Supported |
| **GCN** | Graph Convolutional Network | ✅ Supported |
| **NMPN** | Neural Message Passing Network | ✅ Supported |
| **AttentiveFP** | Attentive Fingerprints | ✅ Supported |
| **rGIN** | Randomized Graph Isomorphism Network | ✅ Supported |
| **RGCN** | Relational Graph Convolutional Network | ✅ Supported |
| **CMPNN** | Chemical Message Passing Neural Network | ✅ Supported |

## 📊 Generated Reports

The automation generates several output files:

### 1. Markdown Report
- **File**: `automated_test_report_YYYYMMDD_HHMMSS.md`
- **Content**: Comprehensive performance analysis with rankings

### 2. JSON Data
- **File**: `test_results_YYYYMMDD_HHMMSS.json`
- **Content**: Raw test data for further analysis

### 3. Console Output
- **Content**: Real-time progress and summary

## 🎯 Example Usage

### Test All Architectures
```bash
# Edit config-desc.cfg to include:
[Models]
architectures = GIN, GINE, GAT, GATv2, GCN, NMPN, AttentiveFP, rGIN, RGCN, CMPNN

# Run automation
python automated_gnn_testing.py config-desc.cfg
```

### Test Specific Architectures
```bash
# Edit config-desc.cfg to include:
[Models]
architectures = GAT, GATv2, GCN

# Run automation
python automated_gnn_testing.py config-desc.cfg
```

### Quick Test (2 architectures)
```bash
python test_automation.py
```

## 🔧 Configuration Options

### Timeouts
The script includes built-in timeouts:
- **Training**: 5 minutes per architecture
- **Application**: 1 minute per architecture

### Delays
- **Between tests**: 2 seconds (configurable in code)

### Backup
- **Config backup**: Automatically created and restored
- **Backup file**: `config-desc.cfg.backup` (auto-deleted after completion)

## 📈 Report Features

### Performance Rankings
- Sorted by final loss (best first)
- Parameter counts
- Training times
- Performance stars (⭐)

### Detailed Results
- Individual architecture analysis
- Success/failure status
- Error messages for failed tests
- Timing information

### Key Insights
- Best performer identification
- Success rate calculation
- Total testing time
- Recommendations

## 🛠️ Customization

### Add New Architectures
1. Ensure the architecture is supported in `keras-gcn-descs.py`
2. Add it to the `[Models]` section in your config
3. Run the automation

### Modify Timeouts
Edit the timeout values in `automated_gnn_testing.py`:
```python
timeout=300  # Training timeout (seconds)
timeout=60   # Application timeout (seconds)
```

### Change Delays
Edit the delay in `run_all_tests()`:
```python
time.sleep(2)  # Delay between tests (seconds)
```

## 🚨 Troubleshooting

### Common Issues

1. **"No [Models] section found"**
   - Add the `[Models]` section to your config file

2. **"No architectures specified"**
   - Add architectures to the `architectures =` line

3. **Training timeout**
   - Increase timeout in the script
   - Check if architecture is supported

4. **Config file not found**
   - Ensure you're in the correct directory
   - Check the config file path

### Error Recovery
- The script automatically restores your original config
- Failed tests are logged with error messages
- Partial results are saved even if some tests fail

## 📝 Example Output

```
🤖 AUTOMATED GNN ARCHITECTURE TESTING
============================================================
📋 Found 3 architectures to test: GCN, GAT, GATv2
💾 Config backup created: config-desc.cfg.backup

📋 Progress: 1/3
============================================================
🧪 TESTING ARCHITECTURE: GCN
============================================================

🚀 Training GCN...
==================================================
✅ GCN training completed successfully!

🔍 Applying GCN...
==================================================
✅ GCN application completed successfully!

📊 GCN Summary:
   Status: ✅ SUCCESS
   Loss: 4.33
   Params: 270,300
   Train Time: 12.3s
   Apply Time: 2.1s

🎯 TESTING COMPLETE!
============================================================
📊 Results Summary:
   Total Architectures: 3
   Successful: 3
   Failed: 0
   Success Rate: 100.0%

🏆 Best Performer: GATv2
   Loss: 2.70
   Parameters: 1,301,100
   Train Time: 45.2s

⏱️  Total Test Time: 89.7 seconds
```

## 🎉 Benefits

1. **Automation**: No manual config changes needed
2. **Comprehensive**: Tests both training and application
3. **Safe**: Automatic config backup and restoration
4. **Detailed**: Generates comprehensive reports
5. **Flexible**: Easy to test any combination of architectures
6. **Reliable**: Handles errors gracefully

## 🔄 Integration

This automation system integrates seamlessly with your existing workflow:
- Uses your existing `keras-gcn-descs.py` script
- Works with your current config file
- Maintains all your existing settings
- Adds only the automation layer

Perfect for systematic architecture comparison and benchmarking! 🚀 