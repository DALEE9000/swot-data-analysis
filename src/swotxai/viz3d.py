"""Interactive 3D views (plotly).

The sea surface is genuinely three-dimensional data, so a rotatable surface is
the honest geometry here: SSH as elevation, with surface-velocity draped on as
color, shows geostrophic flow wrapping the highs and lows. Shares the plot
theme conventions of :mod:`swotxai.experiments` (defaults suit a white
background; the app passes its dark theme).
"""
from __future__ import annotations

import numpy as np

from swotxai.experiments import _apply_base_layout, _theme

# SWOT variables usable as the surface elevation, in preference order.
ELEVATION_VARS = ["ssha_filtered", "mdt"]


def surface_elevation_var(swot_ds) -> str | None:
    """The variable to use as surface elevation for this dataset, or None."""
    return next((v for v in ELEVATION_VARS if v in swot_ds), None)


def ssh_surface_figure(
    swot_ds,
    drape,
    drape_label: str,
    colorbar_label: str = "m/s",
    cmax: float | None = 0.3,
    theme: dict | None = None,
):
    """SSH as a 3D surface, colored by a draped field (velocity, SST, ...).

    ``swot_ds`` is one regridded SWOT dataset (2D ``lat``/``lon`` coords and an
    elevation variable from :data:`ELEVATION_VARS`). ``drape`` is a DataArray
    or 2D array on the same grid whose values color the surface. Returns a
    plotly Figure, or ``None`` if the dataset has no elevation variable.
    """
    import plotly.graph_objects as go

    t = _theme(theme)

    elev_var = surface_elevation_var(swot_ds)
    if elev_var is None:
        return None

    lon = np.asarray(swot_ds["lon"].values, dtype=float)
    lat = np.asarray(swot_ds["lat"].values, dtype=float)
    z = np.asarray(swot_ds[elev_var].values, dtype=float)
    c = np.asarray(drape.values if hasattr(drape, "values") else drape, dtype=float)

    # Color range: fixed for velocity-like fields, robust percentile otherwise.
    finite_c = c[np.isfinite(c)]
    if cmax is None:
        cmax = float(np.percentile(finite_c, 98)) if finite_c.size else 1.0
    cmin = 0.0

    hover = np.where(
        np.isfinite(c),
        np.char.mod("<b>%.3f", c) + f" {colorbar_label}</b>",
        "<b>no data</b>",
    )

    fig = go.Figure(go.Surface(
        x=lon, y=lat, z=z, surfacecolor=c,
        colorscale="Viridis", cmin=cmin, cmax=cmax,
        text=hover,
        hovertemplate=(
            "%{text}<br>"
            + elev_var + " %{z:.3f} m<br>"
            "%{x:.2f}°E · %{y:.2f}°N<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text=f"{drape_label} ({colorbar_label})",
                       font=dict(color=t["ink_muted"], size=11)),
            thickness=12, outlinewidth=0,
            tickfont=dict(color=t["ink_muted"], size=10),
        ),
        lighting=dict(ambient=0.75, diffuse=0.5, specular=0.1),
        connectgaps=False,
    ))

    # Lon/lat keep their true footprint ratio; SSH relief is exaggerated for legibility.
    lon_f, lat_f = lon[np.isfinite(lon)], lat[np.isfinite(lat)]
    lon_span = float(lon_f.max() - lon_f.min()) if lon_f.size else 1.0
    lat_span = float(lat_f.max() - lat_f.min()) if lat_f.size else 1.0
    xy_norm = max(lon_span, lat_span) or 1.0

    axis_common = dict(
        showbackground=False,
        gridcolor=t["grid"],
        zerolinecolor=t["grid"],
        tickfont=dict(color=t["ink_muted"], size=10),
    )
    fig.update_layout(scene=dict(
        xaxis=dict(title=dict(text="lon", font=dict(color=t["ink_muted"])), **axis_common),
        yaxis=dict(title=dict(text="lat", font=dict(color=t["ink_muted"])), **axis_common),
        zaxis=dict(title=dict(text=f"{elev_var} (m)", font=dict(color=t["ink_muted"])), **axis_common),
        aspectmode="manual",
        aspectratio=dict(x=lon_span / xy_norm, y=lat_span / xy_norm, z=0.35),
        camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)),
    ))
    _apply_base_layout(fig, t, 620)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig
