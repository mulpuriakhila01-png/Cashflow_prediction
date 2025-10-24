import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
st.set_page_config(page_title='Cashflow Prediction App', layout='wide')

st.title('💸 Cashflow Prediction Dashboard')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")


data = pd.read_csv('Cashflow_dataset.csv')
st.write("First 5 rows:")
st.write(data.head())

# Observation: Dataset preview shows numeric and categorical company data.

st.write("Dataset Shape:", data.shape)

# Observation: Confirms total rows and columns in the dataset.

st.write("\nDataset Info:")
data.info()

# Observation: Displays column datatypes and non-null counts.


st.write("\nStatistical Summary of Numeric Columns:")
st.write(data.describe().T)

# Observation: Provides descriptive statistics for numeric features.

# Missing Values Check
missing=data.isna().sum()
missing

# Observation: There are 4 columns with missing data

# Checking percentage  missing values
missing_percent = (missing / len(data)) * 100
missing_df = pd.concat([missing, missing_percent], axis=1)
missing_df.columns = ['Missing Values', 'Percent']
st.write("\nColumns with Missing Values:")
st.write(missing_df[missing_df['Missing Values'] > 0])

# FEATURE ENGINEERING
# Example features: Profit = Revenue - COGS - Operating_Expenses
data['Profits'] = data['Revenue'] - data['COGS'] - data['Operating_Expenses']

# Cashflow ratios
data['CapEx_ratio'] = data['CapEx'] / (data['Revenue'] + 1e-6)
data['Debt_ratio'] = data['Debt_Raised'] / (data['Revenue'] + 1e-6)
data['Investment_ratio'] = (data['Investments_Bought'] - data['Investments_Sold'])
data['Cash_Revenue_Ratio'] = data['Net_Cash_Flow'] / (data['Revenue'] + 1e-6)
data['Debt_to_Equity'] = data['Debt_Raised'] / (data['Equity_Issued'] + 1e-6)
data['Operating_Margin'] = data['Profits'] / (data['Revenue'] + 1e-6)

# Observation: New financial ratios and profit metrics derived successfully.


#Handle Missing Data
# Separate numeric and categorical columns
numeric_cols = ['Revenue', 'COGS', 'Operating_Expenses', 'Accounts_Receivable',
                'Accounts_Payable', 'Taxes_Paid', 'CapEx', 'Asset_Sale_Proceeds',
                 'Investments_Bought', 'Investments_Sold', 'Debt_Raised',
                'Debt_Repaid', 'Interest_Paid', 'Equity_Issued', 'Dividends_Paid','Profits',
                'CapEx_ratio','Debt_ratio','Investment_ratio','Cash_Revenue_Ratio','Debt_to_Equity',
                'Operating_Margin']
categorical_cols = ['Company_Name', 'Month']

# Check missing values
for col in categorical_cols:
    data[col].fillna('Unknown', inplace=True)
for col in numeric_cols:
    data[col].fillna(data[col].median(), inplace=True)
# Impute missing values in 'Debt_ratio' before scaling
#data['Debt_ratio'].fillna(data['Debt_ratio'].median(), inplace=True)

# Observation: Missing numeric values filled with median; categorical with 'Unknown'.


# Correlation heatmap
plt.figure(figsize=(15,8))
sns.heatmap(data[numeric_cols + ['Net_Cash_Flow']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
st.pyplot(plt)


# Observation: Reveals strong correlations between Revenue, Profits, and Net Cash Flow.


# Scatter Plot: Revenue vs Next Month Cash Flow
plt.figure(figsize=(6,4))
sns.scatterplot(x='Revenue', y='Net_Cash_Flow', data=data)
plt.title("Revenue vs Net Cash Flow")
st.pyplot(plt)

# Observation: Positive linear relation visible between revenue and cash flow.

# Boxplot before outlier treatment
plt.figure(figsize=(12,8))
sns.boxplot(data=data[numeric_cols], orient="h", color='lightblue')
plt.title("Boxplot of Numeric Features (Before Outlier Treatment)", fontsize=14)
plt.tight_layout()
st.pyplot(plt)

# Observation: Multiple features show extreme outliers.

#Outlier Treatment (IQR Capping)
def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower, lower,
                          np.where(df[column] > upper, upper, df[column]))
    return df

for col in numeric_cols:
    data = cap_outliers_iqr(data, col)
    st.write(f"{col} outliers capped.")

# Observation: Outliers reduced using IQR-based capping.


missing=data.isna().sum()
missing

#Outlier Visualization (After)
plt.figure(figsize=(12,8))
sns.boxplot(data=data[numeric_cols], orient='h')
plt.title("Boxplot of Numeric Features (After Outlier Treatment)")
st.pyplot(plt)

# One-Hot Encoding
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]),
                            columns=encoder.get_feature_names_out(categorical_cols))
data_encoded = pd.concat([data.drop(columns=categorical_cols).reset_index(drop=True),
                          encoded_cols.reset_index(drop=True)], axis=1)

# Observation: One-hot encoding successfully created binary categorical features.

#Feature Scaling
scaler = StandardScaler()
data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])

# Observation: Numeric features standardized for model stability.

#Define Target & Predictors
X = data_encoded.drop(columns=['Net_Cash_Flow'])
y = data_encoded['Net_Cash_Flow']


# OLS Regression for Feature Significance
# P-Value Feature Selection (OLS)
from statsmodels.api import OLS, add_constant

X_sm = add_constant(X)
ols_model = OLS(y, X_sm).fit()
st.write("\nOLS Summary for p-values:")
st.write(ols_model.summary())

# Select significant features (p < 0.05)
p_values = ols_model.pvalues
significant_features = p_values[p_values < 0.05].index.tolist()
if 'const' in significant_features:
    significant_features.remove('const')
st.write("\nSignificant Features (p < 0.05):", significant_features)
#Observation: Features with statistically significant impact identified.

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X[significant_features], y, test_size=0.2, random_state=42)

# Observation: Data split into 80% train and 20% test sets.

st.write("Train Shape:", X_train.shape, y_train.shape)
st.write("Test Shape:", X_test.shape, y_test.shape)

# Model Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n_samples = len(y_true)
    n_features = X_train.shape[1]
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    st.write(f"\n{model_name} Evaluation Metrics:")
    st.write(f"Mean_absolute_error: {mae:.4f}")
    st.write(f"Mean_squared_error: {mse:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"R²: {r2:.4f}")
    st.write(f"Adjusted R²: {adj_r2:.4f}")
    return mae,mse, rmse, r2, adj_r2


# 1. LINEAR REGRESSION MODEL
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# Train
y_test_pred_lr = lr_model.predict(X_test)
mae_lr, mse_lr, rmse_lr, r2_lr, adj_r2_lr = evaluate_model(y_test, y_test_pred_lr, "Linear Regression")

# Test
y_test_pred_lr = lr_model.predict(X_test)
mae_test_lr, mse_test_lr, rmse_test_lr, r2_test_lr, adj_r2_test_lr = evaluate_model(y_test, y_test_pred_lr, "Linear Regression (Test)")

# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=y_test_pred_lr, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted Cash Flow")
plt.xlabel("Actual Cash Flow")
plt.ylabel("Predicted Cash Flow")
plt.grid(True)
st.pyplot(plt)

# Cross-Validation with significant features
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lr_model, X[significant_features], y, cv=kfold, scoring='r2')
st.write("Cross-Validation R² Scores:", scores)
st.write("Average R² Score:", scores.mean())
# Observation: Cross-validation confirms consistent performance across folds.

plt.figure(figsize=(7,5))
sns.barplot(x=list(range(1, len(scores)+1)), y=scores, palette='Blues_r')
plt.axhline(y=scores.mean(), color='red', linestyle='--', label=f'Mean R² = {scores.mean():.4f}')
plt.title("Cross-Validation R² Scores (Significant Features)")
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# 2. RANDOM FOREST REGRESSION
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Train
y_train_pred_rf = rf_model.predict(X_train)
mae_train_rf, mse_train_rf, rmse_train_rf, r2_train_rf, adj_r2_train_rf = evaluate_model(y_train, y_train_pred_rf, "Random Forest Regression (Train)")

# Test
y_test_pred_rf = rf_model.predict(X_test)
mae_test_rf, mse_test_rf, rmse_test_rf, r2_test_rf, adj_r2_test_rf = evaluate_model(y_test, y_test_pred_rf, "Random Forest Regression (Test)")

# K-Fold Cross-Validation (RF)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_scores = cross_val_score(rf_model, X[significant_features], y, cv=kfold, scoring='r2')

st.write(f"Random Forest CV R² Scores: {rf_cv_scores}")
st.write(f"Average CV R² (RF): {rf_cv_scores.mean():.4f}\n")

# Visualization: RF CV Scores
plt.figure(figsize=(7,5))
sns.barplot(x=list(range(1, len(rf_cv_scores)+1)), y=rf_cv_scores, palette='Blues_r')
plt.axhline(y=rf_cv_scores.mean(), color='red', linestyle='--', label=f'Mean R² = {rf_cv_scores.mean():.4f}')
plt.title("Random Forest: 5-Fold Cross-Validation R² Scores")
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.legend()
plt.grid(True)
st.pyplot(plt)

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Train
y_train_pred_xgb = xgb_model.predict(X_train)
mae_train_xgb, mse_train_xgb, rmse_train_xgb, r2_train_xgb, adj_r2_train_xgb = evaluate_model(y_train, y_train_pred_xgb, "XGBoost Regression (Train)")

# Test
y_test_pred_xgb = xgb_model.predict(X_test)
mae_test_xgb, mse_test_xgb, rmse_test_xgb, r2_test_xgb, adj_r2_test_xgb = evaluate_model(y_test, y_test_pred_xgb, "XGBoost Regression (Test)")

# K-Fold Cross-Validation (XGB)
xgb_cv_scores = cross_val_score(xgb_model, X[significant_features], y, cv=kfold, scoring='r2')

st.write(f"XGBoost CV R² Scores: {xgb_cv_scores}")
st.write(f"Average CV R² (XGB): {xgb_cv_scores.mean():.4f}\n")
# Observation: XGBoost maintains high and consistent cross-validation R².

# Visualization: XGB CV Scores
plt.figure(figsize=(7,5))
sns.barplot(x=list(range(1, len(xgb_cv_scores)+1)), y=xgb_cv_scores, palette='Greens_r')
plt.axhline(y=xgb_cv_scores.mean(), color='red', linestyle='--', label=f'Mean R² = {xgb_cv_scores.mean():.4f}')
plt.title("XGBoost: 5-Fold Cross-Validation R² Scores")
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# 📈 Model Comparison Summary
# ----------------------------
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MAE': [mae_lr, mae_rf, mae_xgb],
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb],
    'R²': [r2_lr, r2_rf, r2_xgb],
    'Adjusted R²': [adj_r2_lr, adj_r2_rf, adj_r2_xgb]
})


st.write("\nModel Performance Summary:")
st.write(results.round(4))
# Observation: Model comparison table shows performance ranking.

# Visualization: Compare R² across models
plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='R²', data=results, palette='viridis')
plt.title("Model R² Comparison")
plt.ylabel("R² Score")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
st.pyplot(plt)
# Observation: XGBoost model achieves the highest R² score.

# Visualization: Compare RMSE across models
plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='RMSE', data=results, palette='coolwarm')
plt.title("Model RMSE Comparison (Lower is Better)")
plt.ylabel("RMSE")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
st.pyplot(plt)
# Observation: Random Forest and XGBoost exhibit lowest prediction errors.