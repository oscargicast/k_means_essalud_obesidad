import math

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


def train_kmeans_range(X, K_range, n_init="auto", random_state=0):
    fits = []

    for k in K_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        model.fit(X)
        fits.append(model)

    return fits


def calculate_cluster_metrics(X, fits, K_range):
    db_scores = []
    ch_scores = []
    wcss = []

    for model_fit in fits:
        db = davies_bouldin_score(X, model_fit.labels_)
        ch = calinski_harabasz_score(X, model_fit.labels_)
        db_scores.append(db)
        ch_scores.append(ch)
        wcss.append(model_fit.inertia_)

    best_k_db = list(K_range)[db_scores.index(min(db_scores))]
    best_k_ch = list(K_range)[ch_scores.index(max(ch_scores))]

    return db_scores, ch_scores, wcss, best_k_db, best_k_ch


def plot_elbow_curve(K_range, wcss, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    sns.lineplot(
        x=list(K_range),
        y=wcss,
        marker="o",
        markerfacecolor="red",
        markeredgecolor="red",
        color="blue",
        linewidth=2,
    )
    plt.grid(True, alpha=0.3)
    plt.xlabel("Número de Clusters (K)", fontsize=12)
    plt.ylabel("WCSS", fontsize=12)
    plt.title(
        "Within-Cluster Sum of Squares (Método del Codo)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(K_range)
    plt.tight_layout()
    return plt.gcf()


def plot_cluster_metrics(K_range, db_scores, ch_scores, wcss, figsize=(24, 5)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    sns.lineplot(
        x=list(K_range),
        y=db_scores,
        marker="o",
        markerfacecolor="red",
        markeredgecolor="red",
        color="darkred",
        linewidth=2,
        ax=axes[0],
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("Número de Clusters (K)", fontsize=12)
    axes[0].set_ylabel("Davies-Bouldin Index", fontsize=12)
    axes[0].set_title(
        "Davies-Bouldin Index por K",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].set_xticks(K_range)

    sns.lineplot(
        x=list(K_range),
        y=ch_scores,
        marker="o",
        markerfacecolor="green",
        markeredgecolor="green",
        color="darkgreen",
        linewidth=2,
        ax=axes[1],
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Número de Clusters (K)", fontsize=12)
    axes[1].set_ylabel("Calinski-Harabasz Score", fontsize=12)
    axes[1].set_title(
        "Calinski-Harabasz Score por K",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xticks(K_range)

    sns.lineplot(
        x=list(K_range),
        y=wcss,
        marker="o",
        markerfacecolor="blue",
        markeredgecolor="blue",
        color="darkblue",
        linewidth=2,
        ax=axes[2],
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Número de Clusters (K)", fontsize=12)
    axes[2].set_ylabel("WCSS", fontsize=12)
    axes[2].set_title(
        "Within-Cluster Sum of Squares",
        fontsize=13,
        fontweight="bold",
    )
    axes[2].set_xticks(K_range)

    plt.suptitle(
        "Métricas de Validación de Clusters", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    return fig, axes


def plot_cluster_scatterplots(
    df,
    fits,
    K_range,
    x_col,
    y_col,
    best_k=None,
    x_label=None,
    y_label=None,
    title=None,
    x_lim=None,
    y_lim=None,
    clinical_thresholds=None,
    vertical_thresholds=None,
    figsize_per_row=(20, 6),
    palette="tab10",
    s=20,
    alpha=0.5,
):
    n_plots = len(K_range)
    cols = min(3, n_plots)
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per_row[0], rows * figsize_per_row[1])
    )
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (k, model_fit) in enumerate(zip(K_range, fits)):
        ax = axes[idx]

        df_temp = df[[x_col, y_col]].copy()
        df_temp["cluster"] = model_fit.labels_

        sns.scatterplot(
            data=df_temp,
            x=x_col,
            y=y_col,
            hue="cluster",
            palette=palette,
            s=s,
            alpha=alpha,
            ax=ax,
            legend="brief",
            edgecolor="none",
        )

        if clinical_thresholds:
            for threshold in clinical_thresholds:
                ax.axhline(
                    threshold["value"],
                    color=threshold.get("color", "crimson"),
                    linestyle=threshold.get("linestyle", "--"),
                    linewidth=threshold.get("linewidth", 1.5),
                    alpha=threshold.get("alpha", 0.5),
                    label=threshold.get("label", ""),
                )

        if vertical_thresholds:
            for threshold in vertical_thresholds:
                ax.axvline(
                    threshold["value"],
                    color=threshold.get("color", "crimson"),
                    linestyle=threshold.get("linestyle", "--"),
                    linewidth=threshold.get("linewidth", 1.5),
                    alpha=threshold.get("alpha", 0.5),
                    label=threshold.get("label", ""),
                )

        ax.set_xlabel(x_label or x_col, fontsize=12)
        ax.set_ylabel(y_label or y_col, fontsize=12)

        title_k = f"K = {k}"
        if best_k and k == best_k:
            title_k += "  ★ ÓPTIMO"
        ax.set_title(title_k, fontsize=14, fontweight="bold")

        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)

        ax.grid(True, alpha=0.3, linestyle=":")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[:k], labels[:k], title="Cluster", loc="upper left", fontsize=9
        )

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    if title:
        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.00)

    plt.tight_layout()
    return fig, axes


def plot_3d_clusters(
    df,
    cluster_col,
    x_col,
    y_col,
    z_col,
    x_label=None,
    y_label=None,
    z_label=None,
    title=None,
    x_lim=None,
    y_lim=None,
    z_lim=None,
    figsize=(14, 10),
    s=5,
    alpha=0.3,
    palette="tab10",
    view=(20, 45),
    sample_n=None,
    random_state=42,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Sample data if requested
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state)

    unique_clusters = sorted(df[cluster_col].unique())
    n_clusters = len(unique_clusters)

    colors = sns.color_palette(palette, n_clusters)
    color_map = dict(zip(unique_clusters, colors))

    cluster_colors = df[cluster_col].map(color_map)

    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        df[z_col],
        c=list(cluster_colors),
        s=s,
        alpha=alpha,
        edgecolors="none",
    )

    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    ax.set_xlabel(x_label or x_col, fontsize=12, labelpad=10)
    ax.set_ylabel(y_label or y_col, fontsize=12, labelpad=10)
    ax.set_zlabel(z_label or z_col, fontsize=12, labelpad=10)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if z_lim:
        ax.set_zlim(z_lim)

    ax.view_init(elev=view[0], azim=view[1])

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map[cluster], label=f"Cluster {cluster}")
        for cluster in unique_clusters
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    return fig, ax


def plot_comparison_boxplots(
    dataframes,
    labels,
    value_cols,
    col_labels=None,
    figsize=None,
    palette="Set2",
    showfliers=True,
    show_stats=False,
):
    """
    Crea boxplots para comparar distribuciones entre múltiples DataFrames.

    Útil para visualizar el efecto de limpieza de outliers, transformaciones,
    o cualquier comparación entre diferentes versiones de un dataset.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        Lista de DataFrames a comparar (e.g., [df_original, df_clean])
    labels : list of str
        Etiquetas para cada DataFrame (e.g., ["Original", "Sin outliers"])
    value_cols : list of str
        Nombres de columnas a graficar (e.g., ["tri", "glu", "edad_paciente"])
    col_labels : list of str, optional
        Etiquetas legibles para cada columna. Si None, usa value_cols.
    figsize : tuple, optional
        Tamaño de figura (ancho, alto). Si None, calcula automáticamente.
    palette : str, default "Set2"
        Paleta de colores de seaborn
    showfliers : bool, default True
        Si True, muestra outliers como puntos individuales
    show_stats : bool, default False
        Si True, anota estadísticas (media, mediana, Q1, Q3) dentro de cada boxplot

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, np.ndarray
        Figura y array de ejes de matplotlib

    Example
    -------
    >>> fig, axes = plot_comparison_boxplots(
    ...     dataframes=[df, df_clean],
    ...     labels=["Original", "Sin outliers"],
    ...     value_cols=["tri", "glu"],
    ...     col_labels=["Triglicéridos (mg/dL)", "Glucosa (mg/dL)"]
    ... )
    >>> plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    """
    import pandas as pd

    if len(dataframes) != len(labels):
        raise ValueError("dataframes y labels deben tener la misma longitud")

    if col_labels is None:
        col_labels = value_cols

    if len(value_cols) != len(col_labels):
        raise ValueError("value_cols y col_labels deben tener la misma longitud")

    n_cols = len(value_cols)

    if figsize is None:
        figsize = (7 * n_cols, 6)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    if n_cols == 1:
        axes = [axes]

    combined_dfs = []
    for df, label in zip(dataframes, labels):
        df_temp = df[value_cols].copy()
        df_temp["Dataset"] = label
        combined_dfs.append(df_temp)

    df_combined = pd.concat(combined_dfs, ignore_index=True)

    for idx, (col, col_label) in enumerate(zip(value_cols, col_labels)):
        ax = axes[idx]

        sns.boxplot(
            data=df_combined,
            x="Dataset",
            y=col,
            palette=palette,
            ax=ax,
            showfliers=showfliers,
            linewidth=1.5,
        )

        ax.set_xlabel("", fontsize=12)
        ax.set_ylabel(col_label, fontsize=12)
        ax.set_title(col_label, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle=":")
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=10)

        if show_stats:
            for group_idx, (df_data, label) in enumerate(zip(dataframes, labels)):
                mean_val = df_data[col].mean()
                median_val = df_data[col].median()
                q1_val = df_data[col].quantile(0.25)
                q3_val = df_data[col].quantile(0.75)

                stats_text = (
                    f"Media: {mean_val:.1f}\n"
                    f"Mediana: {median_val:.1f}\n"
                    f"Q1: {q1_val:.1f}\n"
                    f"Q3: {q3_val:.1f}"
                )

                ax.text(
                    group_idx,
                    median_val,
                    stats_text,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="black",
                        linewidth=1.5,
                    ),
                    fontsize=9,
                    weight="bold",
                )

    plt.tight_layout()
    return fig, axes
