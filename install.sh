#!/bin/bash

# bgand256 Installation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "Installing bgand256..."
echo "======================================"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Change to project directory
cd "$PROJECT_DIR"

# Install dependencies
echo "Installing dependencies..."
poetry install

# Run tests to ensure everything works
echo "Running tests..."
poetry run pytest

# Create symbolic link for global access (optional)
echo "Creating symbolic link for global access..."
LINK_PATH="$HOME/.local/bin/bgand256"
if [ -L "$LINK_PATH" ]; then
    rm "$LINK_PATH"
fi

# Create wrapper script
cat > "$LINK_PATH" << 'EOF'
#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")/../../../Projects/github.com/behrangsa/contrastfind"
poetry run bgand256 "$@"
EOF

chmod +x "$LINK_PATH"

echo ""
echo "Installation complete!"
echo "====================="
echo ""
echo "Usage:"
echo "  bgand256 find                    # Find 256 foreground colors"
echo "  bgand256 find --min-contrast 7.0  # Find with WCAG AAA"
echo "  bgand256 analyze '#ffffff'       # Analyze specific background"
echo ""
echo "For more options:"
echo "  bgand256 --help"
