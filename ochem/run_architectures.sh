#!/bin/bash

# Script to run all architectures training and inference
# This script navigates to the ochem directory and runs the Python script

echo "🚀 Starting architecture batch processing..."
echo "📁 Current directory: $(pwd)"

# Navigate to ochem directory
cd "$(dirname "$0")"

echo "📁 Changed to ochem directory: $(pwd)"

# Check if config file exists
if [ ! -f "config-desc-odorless.cfg" ]; then
    echo "❌ Error: config-desc-odorless.cfg not found in $(pwd)"
    exit 1
fi

# Check if keras-gcn-descs.py exists
if [ ! -f "keras-gcn-descs.py" ]; then
    echo "❌ Error: keras-gcn-descs.py not found in $(pwd)"
    exit 1
fi

echo "✅ All required files found"

# Check if we're in a conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "🐍 Using conda environment: $CONDA_DEFAULT_ENV"
    python_cmd="python"
else
    echo "🐍 No conda environment detected"
    echo "💡 Please activate your conda environment first:"
    echo "   conda activate osmo"
    echo "   Then run this script again"
    exit 1
fi

echo "🎯 Starting Python script with: $python_cmd"

# Run the Python script
$python_cmd run_all_architectures.py

echo "🏁 Script execution completed!" 