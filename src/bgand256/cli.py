"""Command-line interface for bgand256."""

import re
import sys
from typing import Tuple, Optional

import click
import colour
import numpy as np

from . import __version__
from .pymp import generate_readable_colors


def parse_hex_color(color_str: str) -> Optional[Tuple[float, float, float]]:
    """Parse hexadecimal color format #RRGGBB."""
    color_str = color_str.strip()
    if not color_str.startswith('#'):
        return None
    
    hex_str = color_str[1:]
    if len(hex_str) != 6:
        return None
    
    try:
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return (r, g, b)
    except ValueError:
        return None


def parse_rgb_color(color_str: str) -> Optional[Tuple[float, float, float]]:
    """Parse RGB color format rgb(R, G, B)."""
    pattern = r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
    match = re.match(pattern, color_str.strip(), re.IGNORECASE)
    
    if not match:
        return None
    
    try:
        r = int(match.group(1))
        g = int(match.group(2))
        b = int(match.group(3))
        
        if not all(0 <= val <= 255 for val in [r, g, b]):
            return None
        
        return (r / 255.0, g / 255.0, b / 255.0)
    except ValueError:
        return None


def parse_hsl_color(color_str: str) -> Optional[Tuple[float, float, float]]:
    """Parse HSL color format hsl(H, S%, L%)."""
    pattern = r'hsl\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*%\s*,\s*(\d+(?:\.\d+)?)\s*%\s*\)'
    match = re.match(pattern, color_str.strip(), re.IGNORECASE)
    
    if not match:
        return None
    
    try:
        h = float(match.group(1))
        s = float(match.group(2))
        l = float(match.group(3))
        
        if not (0 <= h <= 360 and 0 <= s <= 100 and 0 <= l <= 100):
            return None
        
        # Convert HSL to RGB using colour-science
        hsl = np.array([h/360, s/100, l/100])
        rgb = colour.models.rgb.cylindrical.HSL_to_RGB(hsl)
        return tuple(rgb)
    except (ValueError, AttributeError):
        return None


def parse_hsv_color(color_str: str) -> Optional[Tuple[float, float, float]]:
    """Parse HSV color format hsv(H, S%, V%)."""
    pattern = r'hsv\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*%\s*,\s*(\d+(?:\.\d+)?)\s*%\s*\)'
    match = re.match(pattern, color_str.strip(), re.IGNORECASE)
    
    if not match:
        return None
    
    try:
        h = float(match.group(1))
        s = float(match.group(2))
        v = float(match.group(3))
        
        if not (0 <= h <= 360 and 0 <= s <= 100 and 0 <= v <= 100):
            return None
        
        # Convert HSV to RGB using colour-science
        hsv = np.array([h/360, s/100, v/100])
        rgb = colour.models.rgb.cylindrical.HSV_to_RGB(hsv)
        return tuple(rgb)
    except (ValueError, AttributeError):
        return None


def parse_color(color_str: str) -> Tuple[float, float, float]:
    """Parse color string in various formats."""
    color_str = color_str.strip()
    
    # Try each format
    parsers = [
        parse_hex_color,
        parse_rgb_color,
        parse_hsl_color,
        parse_hsv_color
    ]
    
    for parser in parsers:
        result = parser(color_str)
        if result is not None:
            return result
    
    raise ValueError(
        f"Invalid color format: '{color_str}'. "
        "Supported formats: #RRGGBB, rgb(R,G,B), hsl(H,S%,L%), hsv(H,S%,V%)"
    )


def format_color_output(colors: list, format_type: str = "hex") -> list:
    """Format colors for output."""
    formatted = []
    
    for rgb in colors:
        r, g, b = rgb
        
        if format_type == "hex":
            r_int = int(round(r * 255))
            g_int = int(round(g * 255))
            b_int = int(round(b * 255))
            formatted.append(f"#{r_int:02X}{g_int:02X}{b_int:02X}")
        elif format_type == "rgb":
            r_int = int(round(r * 255))
            g_int = int(round(g * 255))
            b_int = int(round(b * 255))
            formatted.append(f"rgb({r_int}, {g_int}, {b_int})")
        else:  # raw
            formatted.append(f"({r:.4f}, {g:.4f}, {b:.4f})")
    
    return formatted


@click.command()
@click.version_option(version=__version__, prog_name="bgand256")
@click.option(
    '-b', '--background-color',
    required=True,
    help='Background color in format: #RRGGBB, rgb(R,G,B), hsl(H,S%,L%), or hsv(H,S%,V%)'
)
@click.option(
    '-f', '--format',
    type=click.Choice(['hex', 'rgb', 'raw'], case_sensitive=False),
    default='hex',
    help='Output format for colors (default: hex)'
)
@click.option(
    '-n', '--number',
    type=click.IntRange(1, 256),
    default=256,
    help='Number of colors to generate (default: 256)'
)
@click.option(
    '--json',
    is_flag=True,
    help='Output as JSON array'
)
def main(background_color: str, format: str, number: int, json: bool):
    """Find foreground colors with good contrast against a background color.
    
    bgand256 generates up to 256 colors that meet WCAG AA contrast ratio
    requirements (4.5:1) against the specified background color.
    
    Examples:
    
        bgand256 -b "#000000"
        
        bgand256 -b "rgb(255, 255, 255)" --format rgb
        
        bgand256 -b "hsl(180, 50%, 50%)" --number 100 --json
    """
    try:
        # Parse the background color
        bg_rgb = parse_color(background_color)
        
        # Generate colors
        colors = generate_readable_colors(bg_rgb)
        
        # Limit to requested number
        colors = colors[:number]
        
        # Format output
        formatted_colors = format_color_output(colors, format.lower())
        
        # Output results
        if json:
            import json as json_module
            click.echo(json_module.dumps(formatted_colors, indent=2))
        else:
            click.echo(f"Found {len(formatted_colors)} colors with good contrast against {background_color}:")
            click.echo()
            
            # Display in columns for better readability
            cols = 4
            for i in range(0, len(formatted_colors), cols):
                row = formatted_colors[i:i+cols]
                click.echo("  " + "  ".join(f"{color:16}" for color in row))
    
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()