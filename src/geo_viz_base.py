import json

import pandas as pd
from matplotlib.patches import Polygon as MplPolygon

CLINICAL_PRESETS = {
    "tri": {
        "vmin": 75,
        "vmax": 225,
        "threshold": 150,
        "label": "Triglicéridos (mg/dL)",
    },
    "glu": {
        "vmin": 70,
        "vmax": 130,
        "threshold": 100,
        "label": "Glucosa (mg/dL)",
    },
}


def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_by_department(df, value_cols=["tri", "glu"]):
    df_agg = df.copy()
    df_agg["departamento_cod"] = (df_agg["ubigeo"] // 10000).astype(str).str.zfill(2)

    agg_dict = {}
    for col in value_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
    agg_dict["lat_mean"] = ("latitude", "mean")
    agg_dict["lon_mean"] = ("longitude", "mean")
    agg_dict["n_obs"] = (value_cols[0], "count")

    dept_stats = df_agg.groupby("departamento_cod").agg(**agg_dict).reset_index()
    return dept_stats


def merge_with_geojson(dept_stats, geojson_data):
    dept_names = {}
    for feature in geojson_data["features"]:
        dept_name = feature["properties"]["NOMBDEP"]
        dept_id = feature["properties"]["FIRST_IDDP"]
        dept_names[dept_id] = dept_name

    dept_names_df = pd.DataFrame(
        list(dept_names.items()), columns=["departamento_cod", "departamento_nombre"]
    )
    return dept_stats.merge(dept_names_df, on="departamento_cod", how="left")


def add_department_boundaries(
    ax, geojson_data, linewidth=0.3, edgecolor="black", alpha=0.5
):
    for feature in geojson_data["features"]:
        if feature["geometry"]["type"] == "Polygon":
            coords_list = [feature["geometry"]["coordinates"]]
        elif feature["geometry"]["type"] == "MultiPolygon":
            coords_list = feature["geometry"]["coordinates"]

        for coords in coords_list:
            for ring in coords:
                polygon = MplPolygon(
                    ring,
                    facecolor="none",
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                )
                ax.add_patch(polygon)
    return ax
