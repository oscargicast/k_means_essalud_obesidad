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
        markersize=5,
        markerfacecolor="#8B4513",
        markeredgecolor="#8B4513",
        color="#8B4513",
        linewidth=1.5,
        ax=axes[0],
    )
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlabel("Número de Clusters (K)", fontsize=10)
    axes[0].set_ylabel("Davies-Bouldin Index", fontsize=10)
    axes[0].set_title("Davies-Bouldin Index por K", fontsize=11)
    axes[0].set_xticks(K_range)

    sns.lineplot(
        x=list(K_range),
        y=ch_scores,
        marker="o",
        markersize=5,
        markerfacecolor="#556B2F",
        markeredgecolor="#556B2F",
        color="#556B2F",
        linewidth=1.5,
        ax=axes[1],
    )
    axes[1].grid(True, alpha=0.2)
    axes[1].set_xlabel("Número de Clusters (K)", fontsize=10)
    axes[1].set_ylabel("Calinski-Harabasz Score", fontsize=10)
    axes[1].set_title("Calinski-Harabasz Score por K", fontsize=11)
    axes[1].set_xticks(K_range)

    sns.lineplot(
        x=list(K_range),
        y=wcss,
        marker="o",
        markersize=5,
        markerfacecolor="#2E5266",
        markeredgecolor="#2E5266",
        color="#2E5266",
        linewidth=1.5,
        ax=axes[2],
    )
    axes[2].grid(True, alpha=0.2)
    axes[2].set_xlabel("Número de Clusters (K)", fontsize=10)
    axes[2].set_ylabel("WCSS", fontsize=10)
    axes[2].set_title("Within-Cluster Sum of Squares", fontsize=11)
    axes[2].set_xticks(K_range)

    plt.suptitle("Métricas de Validación de Clusters", fontsize=12, y=1.02)
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
    palette="Set2",
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

        ax.set_xlabel(x_label or x_col, fontsize=10)
        ax.set_ylabel(y_label or y_col, fontsize=10)

        title_k = f"K = {k}"
        if best_k and k == best_k:
            title_k += "  ★ ÓPTIMO"
        ax.set_title(title_k, fontsize=11)

        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)

        ax.grid(True, alpha=0.2, linestyle=":")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[:k], labels[:k], title="Cluster", loc="upper left", fontsize=9
        )

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    if title:
        plt.suptitle(title, fontsize=12, y=1.00)

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
    palette="Set2",
    view=(20, 45),
    sample_n=None,
    random_state=42,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state)

    unique_clusters = sorted(df[cluster_col].unique())
    n_clusters = len(unique_clusters)

    colors = sns.color_palette(palette, n_clusters)
    color_map = dict(zip(unique_clusters, colors))

    cluster_colors = df[cluster_col].map(color_map)

    ax.scatter(
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

    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    ax.set_xlabel(x_label or x_col, fontsize=10, labelpad=10)
    ax.set_ylabel(y_label or y_col, fontsize=10, labelpad=10)
    ax.set_zlabel(z_label or z_col, fontsize=10, labelpad=10)

    if title:
        ax.set_title(title, fontsize=11, pad=20)

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
    stats_y_offset=3.0,
    normal_ranges=None,
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
    stats_y_offset : float or list of float, default 3.0
        Multiplicador para calcular posición Y del cuadro de estadísticas.
        - Si float: mismo offset para todos los subplots
        - Si lista: un offset por subplot (debe tener len = len(value_cols))
        Fórmula: y_position = Q3 * stats_y_offset. Ajustar si el cuadro se superpone.
    normal_ranges : dict, optional
        Diccionario con rangos normales por columna. Ej: {"tri": (0, 150), "glu": (70, 99)}
        Dibuja franjas verdes horizontales indicando valores normales.

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

    if isinstance(stats_y_offset, (list, tuple)):
        if len(stats_y_offset) != n_cols:
            raise ValueError(
                f"stats_y_offset como lista debe tener longitud {n_cols}, "
                f"pero tiene {len(stats_y_offset)}"
            )
        stats_y_offsets = list(stats_y_offset)
    else:
        stats_y_offsets = [stats_y_offset] * n_cols

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
            hue="Dataset",
            palette=palette,
            legend=False,
            ax=ax,
            showfliers=showfliers,
            linewidth=1.5,
        )

        if normal_ranges and col in normal_ranges:
            ymin, ymax = normal_ranges[col]
            ax.axhspan(
                ymin,
                ymax,
                alpha=0.08,
                color="green",
                zorder=0,
                label=f"Rango normal ({ymin}-{ymax})",
            )

        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel(col_label, fontsize=10)
        ax.set_title(col_label, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y", linestyle=":")
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=9)

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

                y_position = q3_val * stats_y_offsets[idx]

                ax.text(
                    group_idx,
                    y_position,
                    stats_text,
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="#666666",
                        linewidth=1,
                    ),
                    fontsize=9,
                )

    plt.tight_layout()
    return fig, axes


def plot_variables_by_dataset(
    df_with_outliers,
    df_clean,
    value_cols=["tri", "glu"],
    col_labels=None,
    dataset_labels=None,
    figsize=(16, 8),
    palette="Set2",
    showfliers=True,
    show_stats=False,
    stats_y_offset=2.0,
    normal_ranges=None,
):
    """
    Crea boxplots comparando variables agrupadas por dataset.

    A diferencia de plot_comparison_boxplots (que agrupa por variable y muestra datasets),
    esta función agrupa por dataset y muestra variables.

    Útil para comparar múltiples métricas clínicas dentro de cada condición
    (con/sin outliers, antes/después tratamiento, etc.)

    Parameters
    ----------
    df_with_outliers : pd.DataFrame
        DataFrame con datos originales (con outliers)
    df_clean : pd.DataFrame
        DataFrame con datos limpios (sin outliers)
    value_cols : list of str
        Nombres de columnas a graficar (e.g., ["tri", "glu"])
    col_labels : list of str, optional
        Etiquetas legibles para cada columna. Si None, usa value_cols.
    dataset_labels : list of str, optional
        Etiquetas para los datasets. Default: ["Con outliers", "Sin outliers"]
    figsize : tuple, default (16, 8)
        Tamaño de figura (ancho, alto)
    palette : str, default "Set2"
        Paleta de colores de seaborn
    showfliers : bool, default True
        Si True, muestra outliers como puntos individuales
    show_stats : bool, default False
        Si True, anota estadísticas (media, mediana, Q1, Q3) dentro de cada boxplot
    stats_y_offset : float or list of float, default 2.0
        Multiplicador para calcular posición Y del cuadro de estadísticas.
        - Si float: mismo offset para todas las variables
        - Si lista: un offset por variable (debe tener len = len(value_cols))
        Fórmula: y_position = Q3 * stats_y_offset. Ajustar si el cuadro se superpone.
    normal_ranges : dict, optional
        Diccionario con rangos normales por columna. Ej: {"tri": (0, 150), "glu": (70, 99)}

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, np.ndarray
        Figura y array de ejes de matplotlib

    Example
    -------
    >>> fig, axes = plot_variables_by_dataset(
    ...     df_with_outliers=df,
    ...     df_clean=df_clean,
    ...     value_cols=["tri", "glu"],
    ...     col_labels=["Triglicéridos (mg/dL)", "Glucosa (mg/dL)"],
    ...     normal_ranges={"tri": (0, 150), "glu": (70, 99)},
    ...     show_stats=True
    ... )
    """
    import pandas as pd

    if col_labels is None:
        col_labels = value_cols

    if len(value_cols) != len(col_labels):
        raise ValueError("value_cols y col_labels deben tener la misma longitud")

    n_vars = len(value_cols)

    if isinstance(stats_y_offset, (list, tuple)):
        if len(stats_y_offset) != n_vars:
            raise ValueError(
                f"stats_y_offset como lista debe tener longitud {n_vars}, "
                f"pero tiene {len(stats_y_offset)}"
            )
        stats_y_offsets = list(stats_y_offset)
    else:
        stats_y_offsets = [stats_y_offset] * n_vars

    if dataset_labels is None:
        dataset_labels = ["Con outliers", "Sin outliers"]

    dataframes = [df_with_outliers, df_clean]
    n_datasets = len(dataframes)

    fig, axes = plt.subplots(1, n_datasets, figsize=figsize)

    if n_datasets == 1:
        axes = [axes]

    for ds_idx, (df_data, ds_label) in enumerate(zip(dataframes, dataset_labels)):
        ax = axes[ds_idx]

        combined_dfs = []
        for col, col_label in zip(value_cols, col_labels):
            df_temp = pd.DataFrame({col: df_data[col], "Variable": col_label})
            combined_dfs.append(df_temp)

        df_combined = pd.concat(combined_dfs, ignore_index=True)
        df_combined_melted = df_combined.melt(
            id_vars="Variable", var_name="Metric", value_name="Value"
        )

        sns.boxplot(
            data=df_combined_melted,
            x="Variable",
            y="Value",
            hue="Variable",
            palette=palette,
            legend=False,
            ax=ax,
            showfliers=showfliers,
            linewidth=1.5,
        )

        if normal_ranges:
            for col_idx, col in enumerate(value_cols):
                if col in normal_ranges:
                    ymin, ymax = normal_ranges[col]
                    ax.axhspan(
                        ymin,
                        ymax,
                        alpha=0.08,
                        color="green",
                        zorder=0,
                    )

        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("Valor", fontsize=10)
        ax.set_title(ds_label, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y", linestyle=":")
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=9)

        if show_stats:
            for col_idx, (col, col_label) in enumerate(zip(value_cols, col_labels)):
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

                y_position = q3_val * stats_y_offsets[col_idx]

                ax.text(
                    col_idx,
                    y_position,
                    stats_text,
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="#666666",
                        linewidth=1,
                    ),
                    fontsize=9,
                )

    plt.tight_layout()
    return fig, axes
