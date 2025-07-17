"""Command-line interface for bgand256."""

import sys

import click

from . import __version__
from .color_utils import format_color_output, parse_color
from .colors import generate_readable_colors
from .contrast_algorithms import AlgorithmType, generate_contrasting_colors
from .image_generation import create_png_grid


@click.command()
@click.version_option(version=__version__, prog_name="bgand256")
@click.option(
    "-b",
    "--background-color",
    required=True,
    help=(
        "Background color in format: #RRGGBB, rgb(R,G,B), hsl(H,S%,L%), or hsv(H,S%,V%)"
    ),
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["hex", "rgb", "raw"], case_sensitive=False),
    default="hex",
    help="Output format for colors (default: hex)",
)
@click.option(
    "-n",
    "--number",
    type=click.IntRange(1, 256),
    default=256,
    help="Number of colors to generate (default: 256)",
)
@click.option(
    "-F",
    "--output-format",
    type=click.Choice(["grid", "json", "png"], case_sensitive=False),
    default="grid",
    help="Output format (default: grid)",
)
@click.option(
    "-c",
    "--columns",
    type=click.IntRange(1, 32),
    default=4,
    help="Number of columns for grid/PNG layout (default: 4)",
)
@click.option(
    "-o", "--output", type=str, help="Output file path (required for PNG format)"
)
@click.option(
    "--tile-size",
    type=click.IntRange(8, 64),
    default=16,
    help="Size of square tiles in pixels for PNG format (default: 16)",
)
@click.option(
    "--tile-margin",
    type=click.IntRange(0, 20),
    default=5,
    help="Margin between tiles in pixels for PNG format (default: 5)",
)
@click.option(
    "-a",
    "--algorithm",
    type=click.Choice(
        ["standard", "delta-e", "cam16ucs", "hsl-greedy"], case_sensitive=False
    ),
    default="standard",
    help=(
        "Color generation algorithm (default: standard). "
        "Options: "
        "standard (original algorithm), "
        "delta-e (highest quality, slowest), "
        "cam16ucs (balanced quality/speed), "
        "hsl-greedy (fastest)"
    ),
)
@click.option(
    "--contrast-ratio",
    type=click.FloatRange(1.0, 21.0),
    default=4.5,
    help=(
        "Minimum contrast ratio for WCAG compliance (default: 4.5). "
        "Common values: 3.0 (AA large text), 4.5 (AA normal text), "
        "7.0 (AAA normal text)"
    ),
)
def main(
    background_color: str,
    format: str,
    number: int,
    output_format: str,
    columns: int,
    output: str,
    tile_size: int,
    tile_margin: int,
    algorithm: str,
    contrast_ratio: float,
) -> None:
    """Find foreground colors with good contrast against a background color.

    bgand256 generates up to 256 colors that meet WCAG AA contrast ratio
    requirements (4.5:1) against the specified background color.

    The new algorithm options also ensure high contrast between the generated
    colors themselves, creating maximally distinct color palettes.

    Examples:

        bgand256 -b "#000000"

        bgand256 -b "rgb(255, 255, 255)" --format rgb

        bgand256 -b "hsl(180, 50%, 50%)" --number 100 -F json

        bgand256 -b "#ff0000" -F png -c 8 -o colors.png

        bgand256 -b "#000000" -F png --tile-size 32 --tile-margin 10 -o large_tiles.png

        bgand256 -b "#000000" -a delta-e -n 50

        bgand256 -b "#ffffff" -a cam16ucs --number 100 -F json

        bgand256 -b "#808080" -a hsl-greedy -F png -o fast_colors.png

        bgand256 -b "#faf4ed" -a delta-e --contrast-ratio 2.0 -n 50

        bgand256 -b "#000000" --contrast-ratio 7.0 -n 100
    """
    try:
        # Parse the background color
        bg_rgb = parse_color(background_color)

        # Generate colors using selected algorithm
        if algorithm == "standard":
            colors = generate_readable_colors(bg_rgb)
        else:
            # Use advanced contrasting algorithms
            algorithm_type: AlgorithmType = algorithm  # type: ignore[assignment]
            colors = generate_contrasting_colors(bg_rgb, number, algorithm_type, contrast_ratio)

        # Limit to requested number
        colors = colors[:number]

        # Format output
        formatted_colors = format_color_output(colors, format.lower())

        # Output results
        if output_format == "json":
            import json as json_module

            click.echo(json_module.dumps(formatted_colors, indent=2))
        elif output_format == "png":
            if not output:
                click.echo("Error: PNG output requires -o/--output filename", err=True)
                sys.exit(1)

            try:
                create_png_grid(bg_rgb, colors, columns, output, tile_size, tile_margin)
            except Exception as e:
                click.echo(f"Error creating PNG: {e}", err=True)
                sys.exit(1)
        else:  # grid format
            algorithm_desc = {
                "standard": "standard (background contrast only)",
                "delta-e": "Delta E 2000 (highest quality mutual contrast)",
                "cam16ucs": "CAM16UCS (balanced quality/speed)",
                "hsl-greedy": "HSL Greedy (fastest mutual contrast)",
            }
            algo_text = algorithm_desc.get(algorithm, algorithm)

            click.echo(
                f"Found {len(formatted_colors)} colors with good contrast "
                f"against {background_color} using {algo_text}:"
            )
            click.echo()

            # Display in columns for better readability
            for i in range(0, len(formatted_colors), columns):
                row = formatted_colors[i : i + columns]
                click.echo("  " + "  ".join(f"{color:16}" for color in row))

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
