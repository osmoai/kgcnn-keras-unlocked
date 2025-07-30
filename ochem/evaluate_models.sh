#!/bin/bash

# Script to evaluate all trained models
# This script navigates to the ochem directory and runs the evaluation script

echo "📊 Starting model evaluation..."
echo "📁 Current directory: $(pwd)"

# Navigate to ochem directory
cd "$(dirname "$0")"

echo "📁 Changed to ochem directory: $(pwd)"

# Check if evaluation script exists
if [ ! -f "evaluate_all_models.py" ]; then
    echo "❌ Error: evaluate_all_models.py not found in $(pwd)"
    exit 1
fi

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

echo "🎯 Starting evaluation script with: $python_cmd"

# Run the evaluation script
$python_cmd evaluate_all_models.py

echo "🏁 Evaluation completed!" 