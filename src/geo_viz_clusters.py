import seaborn as sns

from .geo_viz_base import add_department_boundaries


def plot_cluster_map(
    ax,
    df,
    cluster_col,
    geojson_data,
    lon_col="longitude",
    lat_col="latitude",
    palette="tab10",
    s=12,
    alpha=0.4,
    title=None,
    hide_ticks=True,
    add_boundaries=True,
):
    if add_boundaries:
        add_department_boundaries(ax, geojson_data)

    sns.scatterplot(
        data=df,
        x=lon_col,
        y=lat_col,
        hue=cluster_col,
        style=cluster_col,
        palette=palette,
        markers=["o", "s", "^", "D", "v"],
        s=s,
        alpha=alpha,
        ax=ax,
        legend="brief",
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_xlim(-82, -68)
    ax.set_ylim(-19, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud", fontsize=11)
    ax.set_ylabel("Latitud", fontsize=11)

    if title:
        ax.set_title(title, fontsize=13)

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if ax.get_legend():
        ax.legend(
            title="Cluster",
            loc="best",
            frameon=True,
            fancybox=True,
            fontsize=10,
            title_fontsize=11,
        )

    return ax
