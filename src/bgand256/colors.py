import colour
import numpy as np

def compute_luminance(rgb):
    """Compute relative luminance of an sRGB color."""
    def linearize(c):
        if c <= 0.03928:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    r_lin = linearize(r)
    g_lin = linearize(g)
    b_lin = linearize(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

def contrast_ratio(l1, l2):
    """Calculate WCAG contrast ratio between two luminance values."""
    light = max(l1, l2)
    dark = min(l1, l2)
    return (light + 0.05) / (dark + 0.05)

def generate_readable_colors(background_rgb):
    # Normalize background to [0,1] if needed
    background_rgb = np.array(background_rgb, dtype=float)
    if np.max(background_rgb) > 1.0:
        background_rgb /= 255.0

    L_bg = compute_luminance(background_rgb)
    valid_colors = []

    # Initial HSL grid
    for H in range(0, 360, 15):
        for S in [0.2, 0.4, 0.6, 0.8]:
            for L in [0.2, 0.4, 0.6, 0.8]:
                hsl = np.array([H/360, S, L])
                rgb = colour.models.rgb.cylindrical.HSL_to_RGB(hsl)
                L_c = compute_luminance(rgb)
                if contrast_ratio(L_bg, L_c) >= 4.5:
                    valid_colors.append(tuple(rgb))
                if len(valid_colors) >= 256:
                    return valid_colors[:256]

    # Expand grid if needed
    for L in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for H in range(0, 360, 15):
            for S in [0.2, 0.4, 0.6, 0.8]:
                hsl = np.array([H/360, S, L])
                rgb = colour.models.rgb.cylindrical.HSL_to_RGB(hsl)
                L_c = compute_luminance(rgb)
                if contrast_ratio(L_bg, L_c) >= 4.5:
                    valid_colors.append(tuple(rgb))
                if len(valid_colors) >= 256:
                    return valid_colors[:256]

    # Fallback to random sampling if insufficient colors
    while len(valid_colors) < 256:
        H = np.random.uniform(0, 1)
        S = np.random.uniform(0.2, 0.8)
        L = np.random.uniform(0, 1)
        hsl = np.array([H, S, L])
        rgb = colour.models.rgb.cylindrical.HSL_to_RGB(hsl)
        L_c = compute_luminance(rgb)
        if contrast_ratio(L_bg, L_c) >= 4.5:
            valid_colors.append(tuple(rgb))

    return valid_colors[:256]
