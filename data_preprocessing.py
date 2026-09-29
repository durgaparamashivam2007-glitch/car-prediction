import os
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

DATASET_PATH = 'Car details v3.csv'
PIPELINE_PATH = 'car_price_pipeline.joblib'
METADATA_PATH = 'model_metadata.joblib'

CATEGORICAL_COLS = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']
NUMERICAL_COLS = ['year', 'km_driven', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']
TARGET_COL = 'selling_price'

def extract_float(val):
    """Extract float number from text strings like '23.4 kmpl' or '1248 CC'."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.search(r'([0-9]+\.?[0-9]*)', val_str)
    if match:
        try:
            res = float(match.group(1))
            return res if res > 0 else np.nan
        except:
            return np.nan
    return np.nan

def load_and_preprocess_data(csv_path=DATASET_PATH):
    """Load Car details v3.csv, clean strings, drop duplicates, parse columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset '{csv_path}' not found!")
    
    df = pd.read_csv(csv_path)
    # Clean duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Feature extraction
    df['brand'] = df['name'].apply(lambda x: str(x).split()[0].strip())
    df['mileage_kmpl'] = df['mileage'].apply(extract_float)
    df['engine_cc'] = df['engine'].apply(extract_float)
    df['max_power_bhp'] = df['max_power'].apply(extract_float)
    
    return df

def build_preprocessor():
    """Build Scikit-learn ColumnTransformer for preprocessing."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, NUMERICAL_COLS),
        ('cat', cat_pipeline, CATEGORICAL_COLS)
    ])
    return preprocessor

def train_and_evaluate_all():
    """Train all regression models, evaluate, select best model, and save artifacts."""
    df = load_and_preprocess_data()
    
    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=15),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    feature_importances = {}

    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }

        # Save feature importances if tree model
        if hasattr(model, 'feature_importances_'):
            try:
                # get transformed column names
                ohe_cols = pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(CATEGORICAL_COLS)
                all_feature_names = list(NUMERICAL_COLS) + list(ohe_cols)
                importances = model.feature_importances_
                fi_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)
                feature_importances[name] = fi_series.to_dict()
            except Exception as e:
                pass

    best_name = max(results, key=lambda k: results[k]['R2'])
    
    # Re-train best model on full dataset or keep train pipeline
    best_model_obj = models[best_name]
    best_pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', best_model_obj)
    ])
    best_pipeline.fit(X_train, y_train)
    
    # Store predictions for actual vs predicted plot
    y_test_pred = best_pipeline.predict(X_test)
    sample_eval = pd.DataFrame({
        'Actual': y_test.values[:300],
        'Predicted': y_test_pred[:300]
    })

    metadata = {
        'best_model_name': best_name,
        'results': results,
        'feature_importances': feature_importances,
        'sample_eval': sample_eval,
        'categorical_cols': CATEGORICAL_COLS,
        'numerical_cols': NUMERICAL_COLS,
        'unique_brands': sorted(df['brand'].unique().tolist()),
        'unique_fuels': sorted(df['fuel'].unique().tolist()),
        'unique_seller_types': sorted(df['seller_type'].unique().tolist()),
        'unique_transmissions': sorted(df['transmission'].unique().tolist()),
        'unique_owners': ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        'numerical_defaults': {
            'year': int(df['year'].median()),
            'km_driven': int(df['km_driven'].median()),
            'mileage_kmpl': float(df['mileage_kmpl'].median()),
            'engine_cc': float(df['engine_cc'].median()),
            'max_power_bhp': float(df['max_power_bhp'].median()),
            'seats': float(df['seats'].median())
        },
        'numerical_ranges': {
            'year': (int(df['year'].min()), int(df['year'].max())),
            'km_driven': (int(df['km_driven'].min()), int(df['km_driven'].max())),
            'mileage_kmpl': (float(df['mileage_kmpl'].min()), float(df['mileage_kmpl'].max())),
            'engine_cc': (float(df['engine_cc'].min()), float(df['engine_cc'].max())),
            'max_power_bhp': (float(df['max_power_bhp'].min()), float(df['max_power_bhp'].max())),
            'seats': (int(df['seats'].min()), int(df['seats'].max()))
        }
    }

    joblib.dump(best_pipeline, PIPELINE_PATH)
    joblib.dump(metadata, METADATA_PATH)
    return best_pipeline, metadata

def load_or_train_pipeline():
    """Load existing pipeline or train a new one if missing."""
    if os.path.exists(PIPELINE_PATH) and os.path.exists(METADATA_PATH):
        pipeline = joblib.load(PIPELINE_PATH)
        metadata = joblib.load(METADATA_PATH)
        return pipeline, metadata
    else:
        return train_and_evaluate_all()

if __name__ == '__main__':
    pipe, meta = train_and_evaluate_all()
    print("Pipeline and metadata saved successfully!")
