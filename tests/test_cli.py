"""Tests for bgand256.cli module."""

import json
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from bgand256.cli import (
    parse_hex_color,
    parse_rgb_color,
    parse_hsl_color,
    parse_hsv_color,
    parse_color,
    format_color_output,
    main
)


class TestParseHexColor:
    """Test the parse_hex_color function."""
    
    def test_valid_hex_uppercase(self):
        """Test valid uppercase hex color."""
        result = parse_hex_color("#FF0000")
        expected = (1.0, 0.0, 0.0)
        assert result == expected
    
    def test_valid_hex_lowercase(self):
        """Test valid lowercase hex color."""
        result = parse_hex_color("#00ff00")
        expected = (0.0, 1.0, 0.0)
        assert result == expected
    
    def test_valid_hex_mixed_case(self):
        """Test valid mixed case hex color."""
        result = parse_hex_color("#0000Ff")
        expected = (0.0, 0.0, 1.0)
        assert result == expected
    
    def test_valid_hex_with_whitespace(self):
        """Test valid hex color with whitespace."""
        result = parse_hex_color("  #FFFFFF  ")
        expected = (1.0, 1.0, 1.0)
        assert result == expected
    
    def test_invalid_missing_hash(self):
        """Test invalid hex without # prefix."""
        result = parse_hex_color("FF0000")
        assert result is None
    
    def test_invalid_too_short(self):
        """Test invalid hex that's too short."""
        result = parse_hex_color("#FF00")
        assert result is None
    
    def test_invalid_too_long(self):
        """Test invalid hex that's too long."""
        result = parse_hex_color("#FF000000")
        assert result is None
    
    def test_invalid_non_hex_chars(self):
        """Test invalid hex with non-hexadecimal characters."""
        result = parse_hex_color("#GGGGGG")
        assert result is None
    
    def test_black_and_white(self):
        """Test black and white hex colors."""
        black = parse_hex_color("#000000")
        white = parse_hex_color("#FFFFFF")
        assert black == (0.0, 0.0, 0.0)
        assert white == (1.0, 1.0, 1.0)


class TestParseRgbColor:
    """Test the parse_rgb_color function."""
    
    def test_valid_rgb_basic(self):
        """Test valid basic RGB color."""
        result = parse_rgb_color("rgb(255, 0, 0)")
        expected = (1.0, 0.0, 0.0)
        assert result == expected
    
    def test_valid_rgb_with_extra_spaces(self):
        """Test valid RGB with extra spaces."""
        result = parse_rgb_color("rgb( 0 , 255 , 0 )")
        expected = (0.0, 1.0, 0.0)
        assert result == expected
    
    def test_valid_rgb_case_insensitive(self):
        """Test RGB parsing is case insensitive."""
        result = parse_rgb_color("RGB(0, 0, 255)")
        expected = (0.0, 0.0, 1.0)
        assert result == expected
    
    def test_valid_rgb_mixed_case(self):
        """Test RGB with mixed case."""
        result = parse_rgb_color("Rgb(128, 128, 128)")
        expected = (128/255, 128/255, 128/255)
        assert result == expected
    
    def test_valid_rgb_edge_values(self):
        """Test RGB with edge values (0 and 255)."""
        result = parse_rgb_color("rgb(0, 255, 0)")
        expected = (0.0, 1.0, 0.0)
        assert result == expected
    
    def test_invalid_rgb_out_of_range_high(self):
        """Test invalid RGB with values > 255."""
        result = parse_rgb_color("rgb(256, 0, 0)")
        assert result is None
    
    def test_invalid_rgb_out_of_range_negative(self):
        """Test invalid RGB with negative values."""
        result = parse_rgb_color("rgb(-1, 0, 0)")
        assert result is None
    
    def test_invalid_rgb_wrong_format(self):
        """Test invalid RGB format."""
        result = parse_rgb_color("rgb(255, 0)")
        assert result is None
    
    def test_invalid_rgb_non_numeric(self):
        """Test invalid RGB with non-numeric values."""
        result = parse_rgb_color("rgb(red, green, blue)")
        assert result is None
    
    @patch('builtins.int')
    def test_invalid_rgb_value_error(self, mock_int):
        """Test RGB parsing ValueError exception handling."""
        # Force a ValueError by providing a string that matches the regex but has invalid int conversion
        mock_int.side_effect = ValueError("Invalid int")
        result = parse_rgb_color("rgb(255, 0, 0)")
        assert result is None
    
    def test_invalid_not_rgb_format(self):
        """Test string that's not in RGB format."""
        result = parse_rgb_color("#FF0000")
        assert result is None


class TestParseHslColor:
    """Test the parse_hsl_color function."""
    
    def test_valid_hsl_basic(self):
        """Test valid basic HSL color."""
        result = parse_hsl_color("hsl(0, 100%, 50%)")
        assert result is not None
        assert len(result) == 3
        assert all(0.0 <= c <= 1.0 for c in result)
    
    def test_valid_hsl_case_insensitive(self):
        """Test HSL parsing is case insensitive."""
        result = parse_hsl_color("HSL(120, 50%, 50%)")
        assert result is not None
    
    def test_valid_hsl_with_decimals(self):
        """Test HSL with decimal values."""
        result = parse_hsl_color("hsl(180.5, 75.5%, 25.5%)")
        assert result is not None
    
    def test_valid_hsl_edge_values(self):
        """Test HSL with edge values."""
        result = parse_hsl_color("hsl(0, 0%, 0%)")
        assert result is not None
        result = parse_hsl_color("hsl(360, 100%, 100%)")
        assert result is not None
    
    def test_invalid_hsl_hue_out_of_range(self):
        """Test invalid HSL with hue > 360."""
        result = parse_hsl_color("hsl(361, 50%, 50%)")
        assert result is None
    
    def test_invalid_hsl_saturation_out_of_range(self):
        """Test invalid HSL with saturation > 100%."""
        result = parse_hsl_color("hsl(180, 101%, 50%)")
        assert result is None
    
    def test_invalid_hsl_lightness_out_of_range(self):
        """Test invalid HSL with lightness > 100%."""
        result = parse_hsl_color("hsl(180, 50%, 101%)")
        assert result is None
    
    def test_invalid_hsl_wrong_format(self):
        """Test invalid HSL format."""
        result = parse_hsl_color("hsl(180, 50)")
        assert result is None
    
    def test_invalid_hsl_missing_percent(self):
        """Test invalid HSL missing % signs."""
        result = parse_hsl_color("hsl(180, 50, 50)")
        assert result is None
    
    @patch('colour.models.rgb.cylindrical.HSL_to_RGB')
    def test_hsl_attribute_error_handling(self, mock_hsl_to_rgb):
        """Test HSL parsing AttributeError exception handling."""
        # Mock colour function to raise AttributeError
        mock_hsl_to_rgb.side_effect = AttributeError("Mock error")
        result = parse_hsl_color("hsl(180, 50%, 50%)")
        assert result is None


class TestParseHsvColor:
    """Test the parse_hsv_color function."""
    
    def test_valid_hsv_basic(self):
        """Test valid basic HSV color."""
        result = parse_hsv_color("hsv(0, 100%, 100%)")
        assert result is not None
        assert len(result) == 3
        assert all(0.0 <= c <= 1.0 for c in result)
    
    def test_valid_hsv_case_insensitive(self):
        """Test HSV parsing is case insensitive."""
        result = parse_hsv_color("HSV(240, 50%, 75%)")
        assert result is not None
    
    def test_valid_hsv_with_decimals(self):
        """Test HSV with decimal values."""
        result = parse_hsv_color("hsv(120.5, 80.5%, 60.5%)")
        assert result is not None
    
    def test_invalid_hsv_out_of_range(self):
        """Test invalid HSV with out of range values."""
        assert parse_hsv_color("hsv(361, 50%, 50%)") is None
        assert parse_hsv_color("hsv(180, 101%, 50%)") is None
        assert parse_hsv_color("hsv(180, 50%, 101%)") is None
    
    def test_invalid_hsv_wrong_format(self):
        """Test invalid HSV format."""
        result = parse_hsv_color("hsv(180, 50)")
        assert result is None
    
    @patch('colour.models.rgb.cylindrical.HSV_to_RGB')
    def test_hsv_attribute_error_handling(self, mock_hsv_to_rgb):
        """Test HSV parsing AttributeError exception handling."""
        # Mock colour function to raise AttributeError
        mock_hsv_to_rgb.side_effect = AttributeError("Mock error")
        result = parse_hsv_color("hsv(180, 50%, 50%)")
        assert result is None


class TestParseColor:
    """Test the parse_color function."""
    
    def test_parse_color_hex(self):
        """Test parse_color with hex input."""
        result = parse_color("#FF0000")
        expected = (1.0, 0.0, 0.0)
        assert result == expected
    
    def test_parse_color_rgb(self):
        """Test parse_color with RGB input."""
        result = parse_color("rgb(0, 255, 0)")
        expected = (0.0, 1.0, 0.0)
        assert result == expected
    
    def test_parse_color_hsl(self):
        """Test parse_color with HSL input."""
        result = parse_color("hsl(240, 100%, 50%)")
        assert result is not None
        assert len(result) == 3
    
    def test_parse_color_hsv(self):
        """Test parse_color with HSV input."""
        result = parse_color("hsv(120, 100%, 100%)")
        assert result is not None
        assert len(result) == 3
    
    def test_parse_color_invalid(self):
        """Test parse_color with invalid input."""
        with pytest.raises(ValueError) as exc_info:
            parse_color("invalid color")
        
        assert "Invalid color format" in str(exc_info.value)
        assert "Supported formats" in str(exc_info.value)
    
    def test_parse_color_whitespace(self):
        """Test parse_color handles whitespace."""
        result = parse_color("  #FF0000  ")
        expected = (1.0, 0.0, 0.0)
        assert result == expected


class TestFormatColorOutput:
    """Test the format_color_output function."""
    
    def test_format_hex(self):
        """Test formatting colors as hex."""
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        result = format_color_output(colors, "hex")
        expected = ["#FF0000", "#00FF00", "#0000FF"]
        assert result == expected
    
    def test_format_rgb(self):
        """Test formatting colors as RGB."""
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        result = format_color_output(colors, "rgb")
        expected = ["rgb(255, 0, 0)", "rgb(0, 255, 0)"]
        assert result == expected
    
    def test_format_raw(self):
        """Test formatting colors as raw values."""
        colors = [(1.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
        result = format_color_output(colors, "raw")
        expected = ["(1.0000, 0.0000, 0.0000)", "(0.5000, 0.5000, 0.5000)"]
        assert result == expected
    
    def test_format_default_hex(self):
        """Test default format is hex."""
        colors = [(1.0, 0.0, 0.0)]
        result = format_color_output(colors)
        expected = ["#FF0000"]
        assert result == expected
    
    def test_format_rounding(self):
        """Test color value rounding."""
        # Test values that need rounding
        colors = [(0.9999, 0.0001, 0.5019)]
        result = format_color_output(colors, "hex")
        # Should round 0.9999*255 = 254.9745 to 255
        # Should round 0.0001*255 = 0.0255 to 0
        # Should round 0.5019*255 = 127.9845 to 128
        assert result == ["#FF0080"]


class TestMainCLI:
    """Test the main CLI function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "Find foreground colors with good contrast" in result.output
        assert "--background-color" in result.output
    
    def test_cli_version(self):
        """Test CLI version output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert "bgand256, version" in result.output
    
    @patch('bgand256.cli.generate_readable_colors')
    def test_cli_basic_usage(self, mock_generate):
        """Test basic CLI usage."""
        mock_generate.return_value = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        
        runner = CliRunner()
        result = runner.invoke(main, ['-b', '#000000'])
        
        assert result.exit_code == 0
        assert "Found 2 colors" in result.output
        assert "#FF0000" in result.output
        assert "#00FF00" in result.output
        mock_generate.assert_called_once()
    
    @patch('bgand256.cli.generate_readable_colors')
    def test_cli_rgb_format(self, mock_generate):
        """Test CLI with RGB output format."""
        mock_generate.return_value = [(1.0, 0.0, 0.0)]
        
        runner = CliRunner()
        result = runner.invoke(main, ['-b', 'rgb(255, 255, 255)', '--format', 'rgb'])
        
        assert result.exit_code == 0
        assert "rgb(255, 0, 0)" in result.output
    
    @patch('bgand256.cli.generate_readable_colors')
    def test_cli_json_output(self, mock_generate):
        """Test CLI with JSON output."""
        mock_generate.return_value = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        
        runner = CliRunner()
        result = runner.invoke(main, ['-b', '#000000', '--json'])
        
        assert result.exit_code == 0
        # Parse the JSON output
        output_data = json.loads(result.output.strip())
        assert output_data == ["#FF0000", "#00FF00"]
    
    @patch('bgand256.cli.generate_readable_colors')
    def test_cli_number_limit(self, mock_generate):
        """Test CLI with number limit."""
        mock_generate.return_value = [(1.0, 0.0, 0.0)] * 10
        
        runner = CliRunner()
        result = runner.invoke(main, ['-b', '#000000', '-n', '3'])
        
        assert result.exit_code == 0
        assert "Found 3 colors" in result.output
    
    def test_cli_invalid_color(self):
        """Test CLI with invalid color input."""
        runner = CliRunner()
        result = runner.invoke(main, ['-b', 'invalid'])
        
        assert result.exit_code == 1
        assert "Invalid color format" in result.output
    
    def test_cli_missing_background(self):
        """Test CLI without required background color."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()
    
    @patch('bgand256.cli.generate_readable_colors')
    def test_cli_hsl_input(self, mock_generate):
        """Test CLI with HSL color input."""
        mock_generate.return_value = [(0.5, 0.5, 0.5)]
        
        runner = CliRunner()
        result = runner.invoke(main, ['-b', 'hsl(180, 50%, 50%)'])
        
        assert result.exit_code == 0
        mock_generate.assert_called_once()
    
    @patch('bgand256.cli.generate_readable_colors')
    def test_cli_hsv_input(self, mock_generate):
        """Test CLI with HSV color input."""
        mock_generate.return_value = [(0.5, 0.5, 0.5)]
        
        runner = CliRunner()
        result = runner.invoke(main, ['-b', 'hsv(240, 50%, 75%)'])
        
        assert result.exit_code == 0
        mock_generate.assert_called_once()
    
    @patch('bgand256.cli.generate_readable_colors')
    def test_cli_exception_handling(self, mock_generate):
        """Test CLI handles unexpected exceptions."""
        mock_generate.side_effect = Exception("Unexpected error")
        
        runner = CliRunner()
        result = runner.invoke(main, ['-b', '#000000'])
        
        assert result.exit_code == 1
        assert "Unexpected error" in result.output
    
    @patch('bgand256.cli.generate_readable_colors')
    def test_cli_raw_format(self, mock_generate):
        """Test CLI with raw output format."""
        mock_generate.return_value = [(0.5, 0.5, 0.5)]
        
        runner = CliRunner()
        result = runner.invoke(main, ['-b', '#000000', '--format', 'raw'])
        
        assert result.exit_code == 0
        assert "(0.5000, 0.5000, 0.5000)" in result.output


@pytest.mark.parametrize("color_input,expected_valid", [
    ("#FF0000", True),
    ("rgb(255, 0, 0)", True),
    ("hsl(0, 100%, 50%)", True),
    ("hsv(0, 100%, 100%)", True),
    ("invalid", False),
    ("#GG0000", False),
    ("rgb(256, 0, 0)", False),
    ("hsl(361, 50%, 50%)", False),
])
def test_color_parsing_parametrized(color_input, expected_valid):
    """Parametrized test for color parsing."""
    if expected_valid:
        result = parse_color(color_input)
        assert result is not None
        assert len(result) == 3
        assert all(isinstance(c, float) for c in result)
    else:
        with pytest.raises(ValueError):
            parse_color(color_input)


def test_main_execution():
    """Test the __main__ execution path."""
    # Test that the module can be executed as main
    import subprocess
    import sys
    
    # Run the module as main with --help to avoid hanging
    result = subprocess.run(
        [sys.executable, '-m', 'bgand256.cli', '--help'],
        capture_output=True,
        text=True,
        cwd='/home/amadeus/Projects/github.com/behrangsa/contrastfind'
    )
    
    # Should succeed and show help
    assert result.returncode == 0
    assert "Find foreground colors" in result.stdout