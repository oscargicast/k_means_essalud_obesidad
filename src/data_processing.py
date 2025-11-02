import numpy as np
from feature_engine.outliers import OutlierTrimmer
from sklearn.preprocessing import LabelEncoder, normalize


def extract_procedure_values(df, tri_name, glu_name):
    df = df.copy()

    df["tri"] = np.where(
        df["procedimiento_1"] == tri_name,
        df["resultado_1"],
        np.where(df["procedimiento_2"] == tri_name, df["resultado_2"], np.nan),
    )

    df["glu"] = np.where(
        df["procedimiento_1"] == glu_name,
        df["resultado_1"],
        np.where(df["procedimiento_2"] == glu_name, df["resultado_2"], np.nan),
    )

    df = df.drop(
        columns=["procedimiento_1", "resultado_1", "procedimiento_2", "resultado_2"]
    )

    return df


def apply_outlier_trimming(df, value_cols, folds):
    df_clean = df.copy()

    for col, fold in zip(value_cols, folds):
        trimmer = OutlierTrimmer(
            capping_method="iqr", tail="both", fold=fold, variables=[col]
        )
        df_clean = trimmer.fit_transform(df_clean)

    return df_clean


def create_ml_features(df, feature_cols, encode_categorical=True):
    df_ml = df[feature_cols].copy()

    df_ml = df_ml.dropna()

    if encode_categorical and "sexo_paciente" in feature_cols:
        le_sexo = LabelEncoder()
        df_ml["sexo_encoded"] = le_sexo.fit_transform(df_ml["sexo_paciente"])
        categorical_features = ["sexo_paciente"]
    else:
        categorical_features = []

    numeric_features = [col for col in feature_cols if col not in categorical_features]

    return df_ml, numeric_features


def normalize_features(X):
    return normalize(X)
