#!/bin/bash

# Build script for SAT Max Solver C++ extension

echo "Building SAT Max Solver C++ extension..."

# Activate virtual environment
source venv/bin/activate
cd custom/data_preprocessing/sat_max

# Check if pybind11 is available
python3 -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "pybind11 is not installed."
    read -p "Do you want to install pybind11 now? I will install it with (python3 -m pip install pybind11) (y/n): " yn
    case $yn in
        [Yy]* ) 
            echo "Installing pybind11..."
            python3 -m pip install pybind11
            ;;
        * )
            echo "pybind11 is required. Exiting."
            exit 1
            ;;
    esac
fi

# Build the extension
python3 setup.py build_ext --inplace

echo "Build complete!"
echo "You can now import sat_max_solver in Python"
