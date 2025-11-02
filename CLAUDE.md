# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

**Academic Paper**: Predicting Laboratory Result Delays in Outpatient Care of Patients with Obesity-Related Conditions in EsSalud (Peru, 2020–2024)

**Objective**: Model the variable `DIFERIMIENTO` (number of days between sample collection and result reading) to predict laboratory turnaround time and identify factors associated with delays in result processing.

**Role**: You are an expert data scientist helping implement a regression model for a small academic paper.

## Dataset

- **Source**: EsSalud Open Data, Peru
- **Size**: ~295,000 laboratory exams (training: 2020–2024), ~64K test records
- **Unit of observation**: One row = one laboratory exam (glucose or triglycerides)
- **Population**: Patients diagnosed with obesity-related conditions (ICD-10/CIE-10 codes)

### Data Files

- `data/dataset_202001_202404_training.csv` - Training set (~295K records, Jan 2020 - Apr 2024)
- `data/dataset_202405_202411_test.csv` - Test set (~64K records, May 2024 - Nov 2024)
- `data/sample.csv` - Quick testing sample (99 records)
- `data/ubigeo_distrito.csv` - UBIGEO geographic coordinates (latitude, longitude)
- `data/peru_departamental.geojson` - Peru department boundaries for choropleth maps
- `data/Diccionario_ExamenesLaboratorio_ConsultaExterna_PatologíasRelacionadas_Obesidad.xlsx` - Data dictionary
- `data/Metadatos_ExamenesLaboratorio_ConsultaExterna_PatologíasRelacionadas_Obesidad_0.docx` - Metadata documentation

**Important**: CSV files use semicolon (`;`) as delimiter, not comma.

### Key Columns

**Target Variable**:
- `DIFERIMIENTO_1`, `DIFERIMIENTO_2`: Integer, days between sample collection and result reading (already calculated in dataset)
- Note: The target variable is PRE-CALCULATED in the dataset, no need to derive from dates

**Features**:
- `EDAD_PACIENTE`: Numeric, patient age
- `SEXO_PACIENTE`: Categorical (MASCULINO/FEMENINO)
- `EDAD_MEDICO`: Numeric, physician age
- `DIAGNOSTICO` / `COD_DIAG`: Categorical, ICD-10 diagnosis codes and descriptions
- `RED`: Categorical, health network (RED_ASISTENCIAL)
- `IPRESS`: Categorical, health facility
- `DEPARTAMENTO`, `PROVINCIA`, `DISTRITO`, `UBIGEO`: Geographic hierarchy
- `SERVICIO_HOSPITALARIO`: Categorical, hospital service (e.g., ENDOCRINOLOGIA, PEDIATRIA)
- `AREA_HOSPITALARIA`: Categorical (e.g., CONSULTA EXTERNA)
- `FECHA_MUESTRA`: Date in YYYYMMDD format (sample collection date)
- `FEC_RESULTADO_1`, `FEC_RESULTADO_2`: Dates in YYYYMMDD format (result reading dates)
- `PROCEDIMIENTO_1`, `PROCEDIMIENTO_2`: String, exam type (e.g., "TRIGLICERIDOS", "DOSAJE DE GLUCOSA EN SANGRE")
- `RESULTADO_1`, `RESULTADO_2`: Numeric, lab result values
- `UNIDADES_1`, `UNIDADES_2`: String, measurement units (e.g., "mg/dL")

**Privacy Note**: `ID_PACIENTE` and `ID_MEDICO` are base64-encoded for anonymization.

### Derived Features to Create

- `MES`: Month extracted from `FECHA_MUESTRA`
- `DIA_SEMANA`: Day of week from `FECHA_MUESTRA`
- `EXAMEN`: Extracted from `PROCEDIMIENTO_1`/`PROCEDIMIENTO_2` (GLUCOSE vs TRIGLYCERIDES)
- Consider interaction terms between region and exam type

## Modeling Approach

### Models to Train

1. **Linear Regression** (baseline)
2. **RandomForestRegressor**
3. **XGBoostRegressor**
4. **LightGBMRegressor**

### Evaluation Metrics

- **Mean Absolute Error (MAE)** - primary metric
- **R² Score**
- **5-fold Cross-validation** (stratified by region or year)

### Feature Importance

- Generate **SHAP values** for interpretability
- Create SHAP summary plots and feature importance plots
- Identify main predictors (e.g., RED_ASISTENCIAL, EXAMEN, MES, DEPARTAMENTO)

## Data Preprocessing Requirements

### Cleaning Steps

1. **Remove invalid records**:
   - Negative `DIFERIMIENTO` values
   - Null `DIFERIMIENTO` values

2. **Handle outliers**:
   - Clip `DIFERIMIENTO > 30` days as anomalies OR cap to 30
   - Document and justify outlier handling approach

3. **Categorical encoding**:
   - Use target encoding or frequency encoding for high-cardinality variables (IPRESS, DIAGNOSTICO)
   - One-hot encoding for low-cardinality variables (SEXO, AREA_HOSPITALARIA)

4. **Temporal split** (IMPORTANT):
   - **Train**: 2020–2023
   - **Test**: 2024
   - Do NOT use random split - maintain temporal ordering

### Date Handling

- Parse dates in YYYYMMDD format: `pd.to_datetime(df['FECHA_MUESTRA'], format='%Y%m%d')`
- **IMPORTANT**: `DIFERIMIENTO_1` and `DIFERIMIENTO_2` are ALREADY calculated in the dataset
- Each record has 2 procedures (glucose and triglycerides) with potentially different delays
- Common approach: Use `DIFERIMIENTO_1` as primary target, or take minimum/maximum of both delays

## Project Deliverables

1. **EDA Notebook**:
   - Distribution of `DIFERIMIENTO` (histogram, summary stats)
   - Correlation heatmap
   - Boxplots of delays by region, exam type, service
   - Time series of average delays over months/years

2. **Model Training & Evaluation Notebook**:
   - Preprocessing pipeline
   - Model training for all 4 models
   - Cross-validation results
   - Final test set evaluation

3. **Visualizations for Paper**:
   - SHAP summary plot
   - Feature importance comparison across models
   - Model comparison table (MAE, R²)
   - Geographic visualization of delays (optional)

4. **Discussion Summary**:
   - Main predictors of delay
   - Model performance comparison
   - Interpretation of results for healthcare context

## Development Environment

- **Python Version**: 3.12 (see `.python-version`)
- **Package Manager**: `uv` (recommended) or pip
- **Dependencies**: Defined in `pyproject.toml`

### Common Commands

```bash
# Install dependencies
uv sync  # or: pip install -e .

# Start Jupyter
jupyter lab  # or: jupyter notebook

# Load data in Python
import pandas as pd
df = pd.read_csv('data/sample.csv', delimiter=';')
df = pd.read_csv('data/dataset_202001_202404_training.csv', delimiter=';')

# Load geographic data
ubigeo_df = pd.read_csv('data/ubigeo_distrito.csv')
df = df.merge(ubigeo_df, left_on='UBIGEO', right_on='inei', how='left')

# Quick data inspection
head -20 data/sample.csv
wc -l data/*.csv
```

### Installed Dependencies

Current project dependencies (see `pyproject.toml`):
- pandas, numpy (data manipulation)
- scikit-learn (modeling, preprocessing, metrics)
- matplotlib, seaborn (visualization)
- statsmodels (statistical analysis)
- jupyter (notebooks)

**Note**: xgboost, lightgbm, and shap are NOT yet installed. Add them when needed:
```bash
uv add xgboost lightgbm shap
```

## Project Structure

Current codebase organization:

```
lab_delay_prediction_essalud/
├── data/                       # Raw data (large CSVs not in git)
│   ├── dataset_202001_202404_training.csv  # 295K training records
│   ├── dataset_202405_202411_test.csv      # 64K test records
│   ├── sample.csv              # 99 sample records for testing
│   ├── ubigeo_distrito.csv     # Geographic coordinates
│   └── peru_departamental.geojson  # Department boundaries
├── nb/                         # Jupyter notebooks (main workspace)
│   ├── 01_data_loading.ipynb   # ✅ EDA and geographic visualization
│   ├── k_means.ipynb           # K-means clustering experiments
│   ├── modelo_no_supervisados_k_means.ipynb
│   ├── projecto_final.ipynb
│   └── S4_Algoritmo_Arbol_de_Clasificación.ipynb
├── src/                        # Reusable Python modules
│   ├── __init__.py
│   └── geo_viz.py              # ✅ Geographic visualization utilities
├── .gitignore
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Locked dependency versions
└── CLAUDE.md                   # This file
```

**Note**: The project currently focuses on EDA and unsupervised learning. The delay prediction modeling (supervised regression) is not yet implemented.

## Existing Code & Key Patterns

### Geographic Visualization Module (`src/geo_viz.py`)

Reusable functions for creating choropleth maps and geographic visualizations:

```python
from src.geo_viz import create_department_choropleth

# Example usage
fig, axes, dept_stats = create_department_choropleth(
    df=df,
    geojson_path='data/peru_departamental.geojson',
    value_cols=['tri', 'glu'],
    titles=['Triglicéridos', 'Glucosa']
)
```

Key functions:
- `load_geojson()` - Load Peru GeoJSON boundaries
- `aggregate_by_department()` - Aggregate lab data by department code
- `merge_with_geojson()` - Merge stats with department names
- `plot_choropleth_map()` - Create single choropleth visualization
- `create_department_choropleth()` - Complete pipeline for multiple metrics

### Data Loading Pattern (from `nb/01_data_loading.ipynb`)

**Standard workflow**:
1. Load raw CSV with `;` delimiter
2. Merge with UBIGEO coordinates for mapping
3. Extract `tri` and `glu` values from dual-procedure format
4. Create clinical status categories (normal/abnormal)

```python
import pandas as pd
import numpy as np

# Load data
df_raw = pd.read_csv('data/dataset_202001_202404_training.csv', delimiter=';')

# Add geographic coordinates
ubigeo_df = pd.read_csv('data/ubigeo_distrito.csv')
df_raw = df_raw.merge(ubigeo_df, left_on='UBIGEO', right_on='inei', how='left')

# Extract procedure values
TRI = "TRIGLICERIDOS"
GLU = "DOSAJE DE GLUCOSA EN SANGRE, CUANTITATIVO (EXCEPTO CINTA REACTIVA)"

df['tri'] = np.where(df['procedimiento_1'] == TRI, df['resultado_1'],
                     np.where(df['procedimiento_2'] == TRI, df['resultado_2'], np.nan))

df['glu'] = np.where(df['procedimiento_1'] == GLU, df['resultado_1'],
                     np.where(df['procedimiento_2'] == GLU, df['resultado_2'], np.nan))
```

**Clinical Reference Ranges** (from EDA):
- **Glucose normal**: 70-99 mg/dL
- **Triglycerides normal**: < 150 mg/dL
- Dataset statistics: ~47.7% have elevated triglycerides, ~41.0% have abnormal glucose

### UBIGEO Geographic Hierarchy

- **UBIGEO format**: 6-digit code (DDPPDD)
- **Department code**: First 2 digits (`ubigeo // 10000`)
- Merge pattern: `left_on='UBIGEO', right_on='inei'` from `ubigeo_distrito.csv`

## Implementation Tips for Delay Prediction

1. **Multiple procedures per record**: Each row has 2 procedures with potentially different delays
   - Current approach in dataset: Use `DIFERIMIENTO_1` and `DIFERIMIENTO_2` separately
   - Alternative: Take minimum or maximum delay as target

2. **Geographic features**: Use hierarchical encoding
   - Extract department code: `(df['ubigeo'] // 10000).astype(str).str.zfill(2)`
   - Health network (RED) and facility (IPRESS) are high-cardinality (30 and 262 unique values)
   - Consider target encoding with cross-validation

3. **Temporal features**: Dates are in YYYYMMDD integer format
   - Extract: `pd.to_datetime(df['FECHA_MUESTRA'], format='%Y%m%d')`
   - Derive: month, day of week, year, quarter

4. **Clinical interpretation**: For academic paper
   - Use clinical reference ranges in discussions
   - Geographic visualizations show regional patterns
   - Consider delays in context of lab value abnormality rates

5. **Cross-validation**: Use TimeSeriesSplit or stratify by region to respect data structure

## Academic Paper Focus

Remember this is for an academic publication, so:
- Document all preprocessing decisions clearly
- Report confidence intervals or standard errors for metrics
- Compare against baseline (simple mean predictor)
- Discuss practical implications for EsSalud healthcare operations
- Keep code reproducible and well-commented
