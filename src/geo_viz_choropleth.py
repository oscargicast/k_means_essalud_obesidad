import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from .geo_viz_base import (
    CLINICAL_PRESETS,
    aggregate_by_department,
    load_geojson,
    merge_with_geojson,
)


def plot_choropleth_map(
    ax,
    dept_stats_full,
    geojson_data,
    metric_col,
    title,
    cmap_name="RdYlGn_r",
    vmin=None,
    vmax=None,
    hide_ticks=True,
):
    value_map = dict(
        zip(dept_stats_full["departamento_cod"], dept_stats_full[metric_col])
    )

    base_col = metric_col.replace("_mean", "")

    if vmin is None and base_col in CLINICAL_PRESETS:
        vmin = CLINICAL_PRESETS[base_col]["vmin"]
    elif vmin is None:
        vmin = dept_stats_full[metric_col].min()

    if vmax is None and base_col in CLINICAL_PRESETS:
        vmax = CLINICAL_PRESETS[base_col]["vmax"]
    elif vmax is None:
        vmax = dept_stats_full[metric_col].max()

    if base_col in CLINICAL_PRESETS and "threshold" in CLINICAL_PRESETS[base_col]:
        threshold = CLINICAL_PRESETS[base_col]["threshold"]
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=threshold, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.get_cmap(cmap_name)

    for feature in geojson_data["features"]:
        dept_id = feature["properties"]["FIRST_IDDP"]

        if dept_id in value_map:
            value = value_map[dept_id]
            color = cmap(norm(value))
        else:
            color = "lightgray"

        if feature["geometry"]["type"] == "Polygon":
            coords_list = [feature["geometry"]["coordinates"]]
        elif feature["geometry"]["type"] == "MultiPolygon":
            coords_list = feature["geometry"]["coordinates"]

        for coords in coords_list:
            for ring in coords:
                polygon = MplPolygon(
                    ring, facecolor=color, edgecolor="black", linewidth=0.5, alpha=0.8
                )
                ax.add_patch(polygon)

    ax.set_xlim(-82, -68)
    ax.set_ylim(-19, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud", fontsize=12)
    ax.set_ylabel("Latitud", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{metric_col.replace('_', ' ').title()}", fontsize=11)

    return ax


def create_department_choropleth(
    df,
    geojson_path,
    value_cols=["tri", "glu"],
    titles=None,
    figsize=(20, 10),
    **kwargs,
):
    geojson_data = load_geojson(geojson_path)
    dept_stats = aggregate_by_department(df, value_cols)
    dept_stats_full = merge_with_geojson(dept_stats, geojson_data)

    fig, axes = plt.subplots(1, len(value_cols), figsize=figsize)
    if len(value_cols) == 1:
        axes = [axes]

    if titles is None:
        titles = [f"Mapa Coroplético - {col.upper()}" for col in value_cols]

    for i, (col, title) in enumerate(zip(value_cols, titles)):
        metric_col = f"{col}_mean"
        plot_choropleth_map(
            axes[i], dept_stats_full, geojson_data, metric_col, title, **kwargs
        )

    plt.tight_layout()
    return fig, axes, dept_stats_full
