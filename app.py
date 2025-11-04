import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

st.set_page_config(page_title="💰 Cash Flow Prediction Dashboard", layout="wide")
st.title("💰 Cash Flow Prediction using ML Models")

uploaded_file = st.file_uploader("📂 Upload your Cashflow CSV file", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    st.write("**Dataset Shape:**", data.shape)

    numeric_cols = [
        'Revenue', 'COGS', 'Operating_Expenses','Depreciation_Amortization',
        'Change_in_Inventory','Accounts_Receivable', 'Accounts_Payable', 'Taxes_Paid',
        'CapEx', 'Asset_Sale_Proceeds', 'Investments_Bought',
        'Investments_Sold','Interest_Received',
        'Debt_Raised', 'Debt_Repaid', 'Interest_Paid',
        'Equity_Issued', 'Dividends_Paid', 'Net_Cash_Flow'
    ]
    categorical_cols = ['Company_Name', 'Month']

    for col in categorical_cols:
        data[col].fillna('Unknown', inplace=True)
    for col in numeric_cols:
        data[col].fillna(data[col].median(), inplace=True)

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

    data['Operating_Cash_Flow'] = (
        (data['Revenue'] - data['COGS'] - data['Operating_Expenses'])
        + data['Depreciation_Amortization']
        - (data['Change_in_Inventory'] + data['Accounts_Receivable'] - data['Accounts_Payable'])
        - data['Taxes_Paid']
    )

    data['Investing_Cash_Flow'] = (
        (-data['CapEx']) + data['Asset_Sale_Proceeds']
        - data['Investments_Bought'] + data['Investments_Sold']
        + data['Interest_Received']
    )

    data['Financing_Cash_Flow'] = (
        data['Debt_Raised'] - data['Debt_Repaid']
        - data['Interest_Paid'] + data['Equity_Issued']
        - data['Dividends_Paid']
    )

    data['Cash_Flow'] = (
        data['Operating_Cash_Flow'] + data['Investing_Cash_Flow'] + data['Financing_Cash_Flow']
    )

    st.write("✅ Feature Engineering Completed!")
    st.dataframe(data[['Company_Name', 'Month', 'Operating_Cash_Flow', 'Investing_Cash_Flow', 'Financing_Cash_Flow', 'Cash_Flow']].head())

    engineered_features = ['Operating_Cash_Flow', 'Investing_Cash_Flow', 'Financing_Cash_Flow']
    numeric_cols.extend(engineered_features)

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cols = pd.DataFrame(
        encoder.fit_transform(data[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )
    data_encoded = pd.concat([data.drop(columns=categorical_cols).reset_index(drop=True),
                              encoded_cols.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])

    X = data_encoded[['Operating_Cash_Flow', 'Investing_Cash_Flow', 'Financing_Cash_Flow']]
    y = data_encoded['Net_Cash_Flow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_model(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - X_train.shape[1] - 1)
        st.write(f"**{model_name} Results:**")
        st.write(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | Adjusted R²: {adj_r2:.4f}")
        return mae, rmse, r2, adj_r2

    with st.spinner("Training Models..."):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)

    st.success("✅ Model training complete!")

    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost"],
        "MAE": [evaluate_model(y_test, lr_pred, "Linear Regression")[0],
                evaluate_model(y_test, rf_pred, "Random Forest")[0],
                evaluate_model(y_test, xgb_pred, "XGBoost")[0]],
        "RMSE": [evaluate_model(y_test, lr_pred, "Linear Regression")[1],
                 evaluate_model(y_test, rf_pred, "Random Forest")[1],
                 evaluate_model(y_test, xgb_pred, "XGBoost")[1]],
        "R²": [evaluate_model(y_test, lr_pred, "Linear Regression")[2],
               evaluate_model(y_test, rf_pred, "Random Forest")[2],
               evaluate_model(y_test, xgb_pred, "XGBoost")[2]]
    })

    st.subheader("📈 Model Performance Summary")
    st.dataframe(results.round(4))

    st.subheader("📊 R² Score Comparison")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x='Model', y='R²', data=results, palette='viridis', ax=ax)
    plt.title("Model Comparison Based on R²")
    st.pyplot(fig)

    best_model_name = results.loc[results['R²'].idxmax(), 'Model']
    st.subheader(f"🔍 Actual vs Predicted Net Cash Flow ({best_model_name})")

    if best_model_name == "Linear Regression":
        best_pred = lr_pred
    elif best_model_name == "Random Forest":
        best_pred = rf_pred
    else:
        best_pred = xgb_pred

    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.scatterplot(x=y_test, y=best_pred, alpha=0.6, color="green", ax=ax2)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("Actual Net Cash Flow")
    plt.ylabel("Predicted Net Cash Flow")
    st.pyplot(fig2)

else:
    st.info("👆 Please upload a Cashflow CSV file to begin analysis.")
