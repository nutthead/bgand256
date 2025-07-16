"""Command-line interface for bgand256."""

import sys

import click

from . import __version__
from .color_utils import format_color_output, parse_color
from .colors import generate_readable_colors
from .image_generation import create_png_grid


@click.command()
@click.version_option(version=__version__, prog_name="bgand256")
@click.option(
    '-b', '--background-color',
    required=True,
    help=(
        'Background color in format: #RRGGBB, rgb(R,G,B), '
        'hsl(H,S%,L%), or hsv(H,S%,V%)'
    )
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
    '-F', '--output-format',
    type=click.Choice(['grid', 'json', 'png'], case_sensitive=False),
    default='grid',
    help='Output format (default: grid)'
)
@click.option(
    '-c', '--columns',
    type=click.IntRange(1, 32),
    default=4,
    help='Number of columns for grid/PNG layout (default: 4)'
)
@click.option(
    '-o', '--output',
    type=str,
    help='Output file path (required for PNG format)'
)
def main(
    background_color: str,
    format: str,
    number: int,
    output_format: str,
    columns: int,
    output: str
):
    """Find foreground colors with good contrast against a background color.

    bgand256 generates up to 256 colors that meet WCAG AA contrast ratio
    requirements (4.5:1) against the specified background color.

    Examples:

        bgand256 -b "#000000"

        bgand256 -b "rgb(255, 255, 255)" --format rgb

        bgand256 -b "hsl(180, 50%, 50%)" --number 100 -F json

        bgand256 -b "#ff0000" -F png -c 8 -o colors.png
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
        if output_format == 'json':
            import json as json_module
            click.echo(json_module.dumps(formatted_colors, indent=2))
        elif output_format == 'png':
            if not output:
                click.echo("Error: PNG output requires -o/--output filename", err=True)
                sys.exit(1)

            try:
                create_png_grid(bg_rgb, colors, columns, output)
            except Exception as e:
                click.echo(f"Error creating PNG: {e}", err=True)
                sys.exit(1)
        else:  # grid format
            click.echo(
                f"Found {len(formatted_colors)} colors with good contrast "
                f"against {background_color}:"
            )
            click.echo()

            # Display in columns for better readability
            for i in range(0, len(formatted_colors), columns):
                row = formatted_colors[i:i+columns]
                click.echo("  " + "  ".join(f"{color:16}" for color in row))

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
