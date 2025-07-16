"""Tests for image generation utilities."""

import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from bgand256.image_generation import create_png_grid


class TestCreatePngGrid:
    """Test the create_png_grid function."""

    def test_empty_colors_raises_error(self):
        """Test that empty colors list raises ValueError."""
        with pytest.raises(ValueError, match="No colors provided"):
            create_png_grid(
                background_color=(0.0, 0.0, 0.0),
                colors=[],
                columns=2,
                output_file="test.png"
            )

    def test_basic_grid_creation(self):
        """Test basic PNG grid creation with default parameters."""
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        background_color = (0.5, 0.5, 0.5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            try:
                with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                     patch('matplotlib.pyplot.tight_layout') as mock_tight_layout, \
                     patch('matplotlib.pyplot.savefig') as mock_savefig, \
                     patch('matplotlib.pyplot.close') as mock_close, \
                     patch('click.echo') as mock_echo:

                    # Mock figure and axis
                    mock_fig = MagicMock()
                    mock_ax = MagicMock()
                    mock_subplots.return_value = (mock_fig, mock_ax)

                    create_png_grid(
                        background_color=background_color,
                        colors=colors,
                        columns=2,
                        output_file=tmp.name
                    )

                    # Verify matplotlib calls
                    mock_subplots.assert_called_once_with(
                        figsize=(47/100, 52/100), dpi=100
                    )
                    mock_fig.patch.set_facecolor.assert_called_once_with(background_color)
                    mock_ax.set_facecolor.assert_called_once_with(background_color)
                    mock_ax.set_xlim.assert_called_once_with(0, 47)
                    mock_ax.set_ylim.assert_called_once_with(0, 52)
                    mock_ax.axis.assert_called_once_with('off')
                    mock_tight_layout.assert_called_once()
                    mock_savefig.assert_called_once_with(
                        tmp.name, bbox_inches='tight', pad_inches=0, dpi=100
                    )
                    mock_close.assert_called_once()
                    mock_echo.assert_called_once_with(f"PNG grid saved to: {tmp.name}")

                    # Verify rectangles were added (3 colors = 3 rectangles)
                    assert mock_ax.add_patch.call_count == 3

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_custom_tile_size_and_margin(self):
        """Test PNG grid creation with custom tile size and margin."""
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        background_color = (0.0, 0.0, 0.0)
        tile_size = 32
        tile_margin = 10

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            try:
                with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                     patch('matplotlib.pyplot.tight_layout'), \
                     patch('matplotlib.pyplot.savefig'), \
                     patch('matplotlib.pyplot.close'), \
                     patch('click.echo'):

                    mock_fig = MagicMock()
                    mock_ax = MagicMock()
                    mock_subplots.return_value = (mock_fig, mock_ax)

                    create_png_grid(
                        background_color=background_color,
                        colors=colors,
                        columns=2,
                        output_file=tmp.name,
                        tile_size=tile_size,
                        tile_margin=tile_margin
                    )

                    # Expected dimensions with custom settings
                    expected_w = (2 * (32 + 10)) + 10  # 94
                    expected_h = (1 * (32 + 10)) + 10 + 10  # 62

                    mock_subplots.assert_called_once_with(
                        figsize=(expected_w/100, expected_h/100), dpi=100
                    )
                    mock_ax.set_xlim.assert_called_once_with(0, expected_w)
                    mock_ax.set_ylim.assert_called_once_with(0, expected_h)

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_single_color_single_column(self):
        """Test PNG grid creation with single color and single column."""
        colors = [(1.0, 1.0, 1.0)]
        background_color = (0.0, 0.0, 0.0)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            try:
                with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                     patch('matplotlib.pyplot.tight_layout'), \
                     patch('matplotlib.pyplot.savefig'), \
                     patch('matplotlib.pyplot.close'), \
                     patch('click.echo'):

                    mock_fig = MagicMock()
                    mock_ax = MagicMock()
                    mock_subplots.return_value = (mock_fig, mock_ax)

                    create_png_grid(
                        background_color=background_color,
                        colors=colors,
                        columns=1,
                        output_file=tmp.name
                    )

                    # Expected dimensions for 1 color in 1 column
                    expected_w = (1 * (16 + 5)) + 5  # 26
                    expected_h = (1 * (16 + 5)) + 5 + 5  # 31

                    mock_subplots.assert_called_once_with(
                        figsize=(expected_w/100, expected_h/100), dpi=100
                    )
                    mock_ax.add_patch.assert_called_once()

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_multiple_rows_calculation(self):
        """Test grid calculation with multiple rows."""
        colors = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 0.0, 1.0)
        ]
        background_color = (0.2, 0.2, 0.2)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            try:
                with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                     patch('matplotlib.pyplot.tight_layout'), \
                     patch('matplotlib.pyplot.savefig'), \
                     patch('matplotlib.pyplot.close'), \
                     patch('click.echo'):

                    mock_fig = MagicMock()
                    mock_ax = MagicMock()
                    mock_subplots.return_value = (mock_fig, mock_ax)

                    create_png_grid(
                        background_color=background_color,
                        colors=colors,
                        columns=3,
                        output_file=tmp.name
                    )

                    # 5 colors in 3 columns = 2 rows (math.ceil(5/3) = 2)
                    rows = math.ceil(5 / 3)
                    expected_w = (3 * (16 + 5)) + 5  # 68
                    expected_h = (rows * (16 + 5)) + 5 + 5  # 52

                    mock_subplots.assert_called_once_with(
                        figsize=(expected_w/100, expected_h/100), dpi=100
                    )
                    mock_ax.set_xlim.assert_called_once_with(0, expected_w)
                    mock_ax.set_ylim.assert_called_once_with(0, expected_h)

                    # Verify all 5 rectangles were added
                    assert mock_ax.add_patch.call_count == 5

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_rectangle_positioning(self):
        """Test that rectangles are positioned correctly."""
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
        background_color = (0.0, 0.0, 0.0)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            try:
                with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                     patch('matplotlib.pyplot.tight_layout'), \
                     patch('matplotlib.pyplot.savefig'), \
                     patch('matplotlib.pyplot.close'), \
                     patch('click.echo'), \
                     patch('matplotlib.patches.Rectangle') as mock_rectangle:

                    mock_fig = MagicMock()
                    mock_ax = MagicMock()
                    mock_subplots.return_value = (mock_fig, mock_ax)

                    create_png_grid(
                        background_color=background_color,
                        colors=colors,
                        columns=2,
                        output_file=tmp.name
                    )

                    # Verify Rectangle calls with correct positions
                    assert mock_rectangle.call_count == 4

                    # Calculate expected positions for 4 colors in 2 columns
                    # Using tile_size=16, tile_margin=5, h=52

                    expected_calls = [
                        # Color 0: row=0, col=0
                        ((5, 26), 16, 16),  # x=5, y=52-5-(0+1)*21=26
                        # Color 1: row=0, col=1
                        ((26, 26), 16, 16),  # x=5+1*21=26, y=26
                        # Color 2: row=1, col=0
                        ((5, 5), 16, 16),   # x=5, y=52-5-(1+1)*21=5
                        # Color 3: row=1, col=1
                        ((26, 5), 16, 16),  # x=26, y=5
                    ]

                    for i, (expected_pos, expected_w, expected_h) in enumerate(
                        expected_calls
                    ):
                        call_args = mock_rectangle.call_args_list[i]
                        assert call_args[0][0] == expected_pos  # Position
                        assert call_args[0][1] == expected_w    # Width
                        assert call_args[0][2] == expected_h    # Height
                        assert call_args[1]['linewidth'] == 0
                        assert call_args[1]['facecolor'] == colors[i]

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_zero_margin(self):
        """Test PNG grid creation with zero margin."""
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        background_color = (0.5, 0.5, 0.5)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            try:
                with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                     patch('matplotlib.pyplot.tight_layout'), \
                     patch('matplotlib.pyplot.savefig'), \
                     patch('matplotlib.pyplot.close'), \
                     patch('click.echo'):

                    mock_fig = MagicMock()
                    mock_ax = MagicMock()
                    mock_subplots.return_value = (mock_fig, mock_ax)

                    create_png_grid(
                        background_color=background_color,
                        colors=colors,
                        columns=2,
                        output_file=tmp.name,
                        tile_size=16,
                        tile_margin=0
                    )

                    # Expected dimensions with zero margin
                    expected_w = (2 * (16 + 0)) + 0  # 32
                    expected_h = (1 * (16 + 0)) + 0 + 0  # 16

                    mock_subplots.assert_called_once_with(
                        figsize=(expected_w/100, expected_h/100), dpi=100
                    )
                    mock_ax.set_xlim.assert_called_once_with(0, expected_w)
                    mock_ax.set_ylim.assert_called_once_with(0, expected_h)

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_large_grid(self):
        """Test PNG grid creation with many colors."""
        # Reduce to 6 colors to avoid >1.0 values
        colors = [(i/10, (i+1)/10, (i+2)/10) for i in range(6)]
        background_color = (0.1, 0.1, 0.1)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            try:
                with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                     patch('matplotlib.pyplot.tight_layout'), \
                     patch('matplotlib.pyplot.savefig'), \
                     patch('matplotlib.pyplot.close'), \
                     patch('click.echo'):

                    mock_fig = MagicMock()
                    mock_ax = MagicMock()
                    mock_subplots.return_value = (mock_fig, mock_ax)

                    create_png_grid(
                        background_color=background_color,
                        colors=colors,
                        columns=4,
                        output_file=tmp.name
                    )

                    # 6 colors in 4 columns = 2 rows (math.ceil(6/4) = 2)
                    rows = math.ceil(6 / 4)
                    expected_w = (4 * (16 + 5)) + 5  # 89
                    expected_h = (rows * (16 + 5)) + 5 + 5  # 52

                    mock_subplots.assert_called_once_with(
                        figsize=(expected_w/100, expected_h/100), dpi=100
                    )

                    # Verify all 6 rectangles were added
                    assert mock_ax.add_patch.call_count == 6

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

