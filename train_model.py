import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# STEP 1: Data Loading and Initial Inspection
# ==========================================

# Load the simulation dataset generated from ANSYS
file_path = 'pbga_simulation_data.csv' 
df = pd.read_csv(file_path)

# Display the first 5 rows to verify structure
print("--- First 5 Rows of the Dataset ---")
print(df.head())

# Check dataset information (data types, non-null counts)
print("\n--- Dataset Information ---")
print(df.info())

# Check for any missing or null values
print("\n--- Missing Values Count per Column ---")
print(df.isnull().sum())


# ==========================================
# STEP 2: Feature & Target Definition & Grouped Split
# ==========================================

# 2.1 Define Feature columns (Using exact column names from your dataset)
feature_columns = [
    'Die thickness (mm)', 
    'Substrate thickness (mm)', 
    'Mold compound thickness (mm)', 
    'EMC Young\'s modulus (GPa)', 
    'EMC CTE (ppm/°C)', 
    'Temperature (°C)'
]

# 2.2 Define Target columns (Using exact column names from your dataset)
target_columns = [
    'Warpage (µm)', 
    'Max Von Mises Stress (MPa)'
]

# 2.3 Separate features (X), targets (y), and the grouping identifier (Configuration ID)
X = df[feature_columns]
y = df[target_columns]
groups = df['Configuration ID']  # Prevents data leakage across thermal steps of the same config

# 2.4 Apply GroupShuffleSplit to divide into Training and Testing sets
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print("\n--- Dataset Splitting Summary ---")
print(f"Training set shape (X_train): {X_train.shape}")
print(f"Testing set shape (X_test): {X_test.shape}")
print(f"Unique configurations in training groups: {groups_train.nunique()}")
print(f"Unique configurations in testing groups: {groups_test.nunique()}")


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold

# ==========================================
# STEP 3: Hyperparameter Tuning (RF & XGBoost)
# ==========================================

# 3.1 Define GroupKFold for cross-validation inside GridSearchCV
gkf = GroupKFold(n_splits=5)

# ------------------------------------------
# 3.2 Random Forest Tuning (Using n_jobs=1 for stability)
# ------------------------------------------
print("\n--- Tuning Random Forest Regressor ---")
rf_model = RandomForestRegressor(random_state=42)

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf_grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=gkf,
    scoring='r2',
    n_jobs=1  # Changed to 1 to prevent multiprocessing termination errors on Windows
)

rf_grid_search.fit(X_train, y_train, groups=groups_train)

best_rf = rf_grid_search.best_estimator_
print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")


# ------------------------------------------
# 3.3 XGBoost Tuning (Using n_jobs=1 for stability)
# ------------------------------------------
print("\n--- Tuning XGBoost Regressor ---")
xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')

xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}

xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    cv=gkf,
    scoring='r2',
    n_jobs=1  # Changed to 1 to prevent multiprocessing termination errors on Windows
)

xgb_grid_search.fit(X_train, y_train, groups=groups_train)

best_xgb = xgb_grid_search.best_estimator_
print(f"Best XGBoost Parameters: {xgb_grid_search.best_params_}")


import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# STEP 4: Model Evaluation & Performance Metrics
# ==========================================

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    print(f"\n--- {model_name} Performance on Test Set ---")
    targets = ['Warpage (µm)', 'Max Von Mises Stress (MPa)']
    metrics_results = {}
    
    for i, target_name in enumerate(targets):
        y_true_t = y_test.iloc[:, i]
        y_pred_t = y_pred[:, i]
        
        r2 = r2_score(y_true_t, y_pred_t)
        mae = mean_absolute_error(y_true_t, y_pred_t)
        rmse = np.sqrt(mean_squared_error(y_true_t, y_pred_t))
        mape = np.mean(np.abs((y_true_t - y_pred_t) / (y_true_t + 1e-8))) * 100
        
        print(f"  Target: {target_name}")
        print(f"    R² Score : {r2:.4f}")
        print(f"    MAE      : {mae:.4f}")
        print(f"    RMSE     : {rmse:.4f}")
        print(f"    MAPE     : {mape:.2f}%")
        
        metrics_results[target_name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        
    return y_pred, metrics_results

# Evaluate tuned Random Forest and XGBoost
rf_pred, rf_metrics = evaluate_model(best_rf, X_test, y_test, "Random Forest Regressor")
xgb_pred, xgb_metrics = evaluate_model(best_xgb, X_test, y_test, "XGBoost Regressor")


# ==========================================
# STEP 5: Weighted Trade-Off Analysis
# ==========================================
print("\n--- STEP 5: Weighted Trade-Off Analysis ---")

# Computing average R² across both targets (Warpage and Von Mises Stress)
rf_avg_r2 = (rf_metrics['Warpage (µm)']['R2'] + rf_metrics['Max Von Mises Stress (MPa)']['R2']) / 2
xgb_avg_r2 = (xgb_metrics['Warpage (µm)']['R2'] + xgb_metrics['Max Von Mises Stress (MPa)']['R2']) / 2

print(f"Random Forest Overall Average R²: {rf_avg_r2:.4f}")
print(f"XGBoost Overall Average R²      : {xgb_avg_r2:.4f}")

# Selecting the superior model based on weighted multi-target performance
if xgb_avg_r2 >= rf_avg_r2:
    final_model = best_xgb
    final_model_name = "XGBoost Regressor"
else:
    final_model = best_rf
    final_model_name = "Random Forest Regressor"

print(f"\n>>> Selected Final Surrogate Model: {final_model_name} <<<")


# ==========================================
# STEP 6: Model Serialization (Saving for Streamlit)
# ==========================================
model_filename = 'pbga_surrogate_model.pkl'
joblib.dump(final_model, model_filename)
print(f"\nSuccessfully serialized and saved the final model as '{model_filename}'!")
