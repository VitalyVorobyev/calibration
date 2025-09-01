#!/bin/bash
# Development setup script for calibration library

set -e

echo "Setting up development environment for calibration library..."

# Install system dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing Ubuntu dependencies..."
    sudo apt update
    sudo apt install -y cmake ninja-build libeigen3-dev libceres-dev \
        nlohmann-json3-dev libgtest-dev libgmock-dev libboost-dev \
        libcli11-dev clang-tidy cppcheck clang-format lcov
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing macOS dependencies..."
    brew update
    brew install cmake ninja eigen ceres-solver nlohmann-json \
        googletest boost cli11 llvm
else
    echo "Unsupported OS. Please install dependencies manually."
    exit 1
fi

# Install pre-commit hooks
if command -v pip3 &> /dev/null; then
    pip3 install pre-commit
    pre-commit install
    echo "Pre-commit hooks installed"
else
    echo "Warning: pip3 not found. Pre-commit hooks not installed."
fi

# Build the project
echo "Building project..."
make clean
make build

# Run tests
echo "Running tests..."
make test

echo "Development environment setup complete!"
echo ""
echo "Available make targets:"
make help
