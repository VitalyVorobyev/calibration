#!/bin/bash

# Documentation generation script for Calibration Library
# This script builds Doxygen documentation and optionally opens it

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "🔧 Generating Doxygen documentation for Calibration Library..."

# Check if Doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "❌ Error: Doxygen is not installed"
    echo "   Please install Doxygen:"
    echo "   - Ubuntu/Debian: sudo apt-get install doxygen graphviz"
    echo "   - macOS: brew install doxygen graphviz"
    echo "   - Windows: Download from https://www.doxygen.nl/download.html"
    exit 1
fi

# Check if Doxyfile exists
if [[ ! -f "$PROJECT_ROOT/Doxyfile" ]]; then
    echo "❌ Error: Doxyfile not found in project root"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/doc/doxygen"

# Generate documentation
echo "📚 Running Doxygen..."
cd "$PROJECT_ROOT"
if doxygen Doxyfile; then
    echo "✅ Documentation generated successfully!"
    echo "📁 Output directory: $PROJECT_ROOT/doc/doxygen/html"
    echo "🌐 Main page: $PROJECT_ROOT/doc/doxygen/html/index.html"
    
    # Option to open documentation
    if [[ "$1" == "--open" ]] || [[ "$1" == "-o" ]]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open "$PROJECT_ROOT/doc/doxygen/html/index.html"
        elif command -v open &> /dev/null; then
            open "$PROJECT_ROOT/doc/doxygen/html/index.html"
        else
            echo "💡 To view documentation, open: $PROJECT_ROOT/doc/doxygen/html/index.html"
        fi
    else
        echo "💡 To view documentation, run: ./generate_docs.sh --open"
        echo "   or manually open: $PROJECT_ROOT/doc/doxygen/html/index.html"
    fi
else
    echo "❌ Error: Documentation generation failed"
    exit 1
fi
