"""Image generation utilities for bgand256."""

import math

import click
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def create_png_grid(
    background_color: tuple[float, float, float],
    colors: list[tuple[float, float, float]],
    columns: int,
    output_file: str,
    tile_size: int = 16,
    tile_margin: int = 5,
) -> None:
    """Create a PNG image with colors arranged in a grid."""
    n_colors = len(colors)
    if n_colors == 0:
        raise ValueError("No colors provided")

    # Calculate grid dimensions
    rows = math.ceil(n_colors / columns)

    # Image dimensions using the provided formula
    w = (columns * (tile_size + tile_margin)) + tile_margin
    # Extra bottom margin
    h = (rows * (tile_size + tile_margin)) + tile_margin + tile_margin

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)  # type: ignore[misc]

    # Set background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Remove axes
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    # Draw color tiles
    for i, color in enumerate(colors):
        row = i // columns
        col = i % columns

        # Calculate position (flipping y-axis for proper display)
        x = tile_margin + col * (tile_size + tile_margin)
        y = h - tile_margin - (row + 1) * (tile_size + tile_margin)

        # Create rectangle
        rect = patches.Rectangle(
            (x, y), tile_size, tile_size, linewidth=0, facecolor=color
        )
        ax.add_patch(rect)

    # Save the image
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0, dpi=100)  # type: ignore[misc]
    plt.close()

    click.echo(f"PNG grid saved to: {output_file}")
