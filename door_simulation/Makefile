# Makefile for Door Learning Simulation

CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
TARGET = door_learning_simulation
SOURCE = door_learning_simulation.cpp

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Run with default parameters
run: $(TARGET)
	./$(TARGET)

# Run with custom parameters (example)
run-example: $(TARGET)
	./$(TARGET) --k 5 --a 0.5 0.4 0.05 0.02 0.03 --lr 0.02 --iterations 50 --simulations 5000

# Install Python dependencies for plotting
install-deps:
	pip install numpy matplotlib

# Run the plotting script
plot:
	python3 plot_learning_history.py

# Run combined plotting
plot-combined:
	python3 plot_learning_history.py --combined

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the C++ executable"
	@echo "  clean        - Remove build artifacts"
	@echo "  run          - Run with default parameters"
	@echo "  run-example  - Run with example parameters"
	@echo "  install-deps - Install Python dependencies"
	@echo "  plot         - Plot individual history files"
	@echo "  plot-combined- Plot all histories in one figure"
	@echo "  help         - Show this help message"

.PHONY: all clean run run-example install-deps plot plot-combined help
