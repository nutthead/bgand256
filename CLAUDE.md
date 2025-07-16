# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bgand256 is a Python tool that finds 256 foreground colors with good contrast against a given background color, following WCAG accessibility guidelines with a minimum contrast ratio of 4.5:1.

**Author Information:**
- Author: Behrang Saeedzadeh
- Email: hello@behrang.org
- Use this author information for any new files or code documentation

## Commands

### Development Setup
```bash
# Install dependencies using Poetry
poetry install

# Run shellcheck on install.sh before modifying
shellcheck install.sh
```

### Code Quality
```bash
# Lint and auto-fix code issues
poetry run ruff check src/ tests/ --fix

# Format code (88 char line length, Black-compatible)
poetry run ruff format src/ tests/

# Type checking with mypy (strict mode enabled)
poetry run mypy src/

# Type checking with Pyright (comprehensive static analysis)
poetry run pyright src/ tests/
```

### Testing
```bash
# Run all tests with coverage
poetry run pytest --cov=bgand256 --cov-report=term-missing

# Run a specific test file
poetry run pytest tests/test_specific.py -v

# Run tests matching a pattern
poetry run pytest -k "test_contrast" -v
```

### Running the Application
```bash
# Currently no CLI exists - needs implementation at src/bgand256/cli.py
# The intended command would be: bgand256

# Run the basic test script
cd src/bgand256 && python main.py
```

## Architecture

### Core Components

1. **Color Generation Algorithm** (`src/bgand256/colors.py`):
   - `find_foreground_colors()`: Main function that generates 256 colors with sufficient contrast
   - Uses HSL color space for systematic color generation (hue steps of 15°, saturation levels of 20/40/60/80%)
   - Implements WCAG contrast ratio calculation via `colour-science` library
   - Falls back to random sampling if systematic generation doesn't yield enough colors

2. **Color Science Integration**:
   - Leverages `colour-science` for accurate color space conversions
   - Proper luminance calculations following WCAG 2.0 guidelines
   - RGB normalization to [0,1] range for calculations

### Key Implementation Details

- **Contrast Ratio**: Minimum 4.5:1 (WCAG AA standard for normal text)
- **Color Generation Strategy**: 
  1. Systematic HSL exploration
  2. Random sampling fallback
  3. Returns up to 256 valid colors
- **Dependencies**: Heavy reliance on numpy arrays and colour-science for performance

### Known Issues

1. **Missing CLI**: The `cli.py` file referenced in pyproject.toml doesn't exist
2. **Naming Inconsistency**: Previously used multiple names, now standardized as "bgand256"
3. **Demo Script**: No demo script currently exists
4. **No Tests**: Test directory exists but contains no tests

### Future Development Notes

When implementing the CLI (`src/bgand256/cli.py`), ensure it:
- Accepts background color in multiple formats (hex, RGB, named colors)
- Outputs colors in user-specified format
- Provides options for different contrast ratios (AA vs AAA standards)
- Supports batch processing for multiple background colors