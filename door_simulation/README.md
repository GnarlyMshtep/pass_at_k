# Door Learning Simulation - C++ Version

This directory contains a C++ implementation of the door learning simulation algorithm, along with a Python plotting script for visualization.

## Files

- `door_learning_simulation.cpp` - Main C++ implementation
- `plot_learning_history.py` - Python script for plotting results
- `Makefile` - Build configuration
- `README.md` - This file

## Building and Running

### Prerequisites

- C++ compiler with C++11 support (g++, clang++, etc.)
- Python 3 with numpy and matplotlib (for plotting)

### Building the C++ Program

```bash
# Build the executable
make

# Or manually:
g++ -std=c++11 -Wall -Wextra -O2 -o door_learning_simulation door_learning_simulation.cpp
```

### Running the Simulation

```bash
# Run with default parameters
./door_learning_simulation

# Run with custom parameters
./door_learning_simulation --k 5 --a 0.5 0.4 0.05 0.02 0.03 --lr 0.02 --iterations 100 --simulations 10000

# Using make targets
make run
make run-example
```

### Command Line Arguments

- `--k <value>` - Number of chances/attempts (default: 5)
- `--a <a1> <a2> <a3> <a4> <a5>` - Initial strategy probabilities (default: 0.5 0.4 0.05 0.02 0.03)
- `--lr <value>` - Learning rate (default: 0.02)
- `--iterations <value>` - Number of learning iterations (default: 100)
- `--simulations <value>` - Number of simulations per iteration (default: 10000)
- `--help` - Show help message

### Example Usage

```bash
# Run a quick test with fewer iterations
./door_learning_simulation --k 3 --iterations 20 --simulations 1000

# Run with different initial probabilities
./door_learning_simulation --a 0.8 0.1 0.05 0.03 0.02 --iterations 50

# Run with higher learning rate
./door_learning_simulation --lr 0.05 --iterations 200
```

## Output

The C++ program generates:
1. Console output showing the learning progress
2. A history file named `learning_history_k<k>_lr<lr>_a1<a1>_a2<a2>_a3<a3>_a4<a4>_a5<a5>_<timestamp>.txt`

The history file contains:
- Header information with simulation parameters
- Data lines with format: `iteration a1 a2 a3 a4 a5`

## Plotting Results

### Install Python Dependencies

```bash
# Install required packages
pip install numpy matplotlib

# Or using make
make install-deps
```

### Generate Plots

```bash
# Plot individual history files
python3 plot_learning_history.py

# Plot all histories in a combined figure
python3 plot_learning_history.py --combined

# Using make targets
make plot
make plot-combined
```

### Plotting Options

- `--pattern <pattern>` - Pattern to match history files (default: learning_history_*.txt)
- `--combined` - Create a combined plot with all histories in subplots
- `--no-save` - Do not save plots to files
- `--no-show` - Do not display plots (useful for batch processing)

## Algorithm Description

The simulation implements a learning algorithm for the "infinite doors" problem:

### Problem Setup
- Infinite doors with reward behind door i with probability 1/2^i
- k chances to choose doors
- Five strategies to analyze:
  1. Choose door i on attempt i
  2. Choose door k+1-i on attempt i
  3. Always choose door 1
  4. Random door between 1 and k
  5. Random door with probability proportional to 1/2^i

### Learning Process
1. For each iteration, simulate all strategies
2. Rank strategies by performance
3. Update strategy probabilities based on ranking:
   - 1st place: +2×learning_rate
   - 2nd place: +learning_rate
   - 3rd place: no change
   - 4th place: -learning_rate
   - 5th place: -2×learning_rate
4. Normalize probabilities to sum to 1
5. Repeat for specified number of iterations

## Performance Notes

- The C++ version is significantly faster than the Python version
- Use fewer simulations per iteration for faster testing
- Use more simulations for more accurate results
- The algorithm is stochastic, so results may vary between runs

## Example Workflow

```bash
# 1. Build the program
make

# 2. Run multiple simulations with different parameters
./door_learning_simulation --k 5 --a 0.5 0.4 0.05 0.02 0.03 --iterations 100
./door_learning_simulation --k 5 --a 0.2 0.2 0.2 0.2 0.2 --iterations 100
./door_learning_simulation --k 5 --a 0.8 0.1 0.05 0.03 0.02 --iterations 100

# 3. Plot all results
python3 plot_learning_history.py --combined

# 4. Clean up
make clean
```
