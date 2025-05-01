import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/YOGESH/OneDrive/Documents/sales_data.csv')

df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df.fillna(0, inplace=True)

features = ['store_id', 'sku_id', 'price', 'promotion', 'day', 'month', 'year']
target = 'units_sold'
X = df[features]
y = df[target]

weights = np.where(df['promotion'] == 1, 1.5, 1.0)

X_train, X_test, y_train, y_test, w_train, w_test, df_train, df_test = train_test_split(
    X, y, weights, df, test_size=0.2, random_state=42
)

models = {
    'Linear Regression': LinearRegression(),
    'Bagging (Linear Regression)': BaggingRegressor(estimator=LinearRegression(), n_estimators=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

print("Model Performance:")
for name, model in models.items():
    model.fit(X_train, y_train, sample_weight=w_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

best_model = models['XGBoost']
best_model.fit(X_train, y_train, sample_weight=w_train)
y_pred_final = best_model.predict(X_test)

prediction_df = df_test[['date', 'store_id', 'sku_id']].copy()
prediction_df['actual_units_sold'] = y_test.values
prediction_df['predicted_units_sold'] = y_pred_final
prediction_df.to_csv('detailed_predictions.csv', index=False)

forecast_summary = prediction_df.copy()
forecast_summary['month_year'] = forecast_summary['date'].dt.to_period('M')
summary = forecast_summary.groupby(['store_id', 'sku_id', 'month_year'])['predicted_units_sold'].sum().reset_index()
summary.rename(columns={'predicted_units_sold': 'total_forecasted_demand'}, inplace=True)
summary.to_csv('forecast_summary.csv', index=False)

threshold = forecast_summary['predicted_units_sold'].quantile(0.90)
insights = summary[summary['total_forecasted_demand'] > threshold]

insights['insight'] = insights.apply(
    lambda row: f"🔺 Increase production for SKU {row['sku_id']} in Store {row['store_id']} for {row['month_year']} due to high forecasted demand.",
    axis=1
)

overstock_risk = summary[summary['total_forecasted_demand'] < summary['total_forecasted_demand'].quantile(0.10)]
overstock_risk['insight'] = overstock_risk.apply(
    lambda row: f"📉 Reduce stock of SKU {row['sku_id']} in Store {row['store_id']} for {row['month_year']} due to low forecasted demand.",
    axis=1
)

promotions_impact = df.groupby(['sku_id', 'promotion'])['units_sold'].mean().reset_index()
promotions_impact['insight'] = promotions_impact.apply(
    lambda row: f"📈 Sales of SKU {row['sku_id']} are higher during promotions - consider promotional pricing."
                if row['promotion'] == 1 else "",
    axis=1
)

all_insights = pd.concat([insights[['sku_id', 'store_id', 'month_year', 'insight']],
                          overstock_risk[['sku_id', 'store_id', 'month_year', 'insight']],
                          promotions_impact[['sku_id', 'insight']]], ignore_index=True)

all_insights.to_csv('inventory_insights.csv', index=False)

all_insights['scenario_weight'] = all_insights.apply(
    lambda row: 1.5 if 'Increase production' in row['insight'] else 1.0, axis=1
)

ranked_insights = all_insights.sort_values(by=['scenario_weight', 'sku_id'], ascending=False)

top_10_insights = ranked_insights.head(10)

print("\nTop 10 Scenario-Weighted Insights:")
for idx, row in top_10_insights.iterrows():
    print(f"🔮 {row['insight']}")

plt.figure(figsize=(10, 5))
sns.lineplot(x=range(100), y=y_test.values[:100], label='Actual')
sns.lineplot(x=range(100), y=y_pred_final[:100], label='Predicted')
plt.title('Actual vs Predicted Sales using XGBoost')
plt.xlabel('Sample Index')
plt.ylabel('Units Sold')
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Files generated:")
print("- detailed_predictions.csv")
print("- forecast_summary.csv")
print("- inventory_insights.csv")
