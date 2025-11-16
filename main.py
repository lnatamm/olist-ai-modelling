import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Carregar datasets
order_items = pd.read_csv('data/olist_order_items_dataset.csv')
products = pd.read_csv('data/olist_products_dataset.csv')
orders = pd.read_csv('data/olist_orders_dataset.csv')
customers = pd.read_csv('data/olist_customers_dataset.csv')
sellers = pd.read_csv('data/olist_sellers_dataset.csv')

# Feature engineering: volume e peso
products['volume'] = products['product_length_cm'] * products['product_height_cm'] * products['product_width_cm']
products['product_category_name'] = products['product_category_name'].fillna('Indefinido')

# Merge dos dados
data = order_items.merge(products, on='product_id').merge(orders, on='order_id').merge(customers, on='customer_id').merge(sellers, on='seller_id')

# Features categóricas e numéricas
data['same_state'] = (data['customer_state'] == data['seller_state']).astype(int)
data['customer_state'] = data['customer_state'].astype('category').cat.codes
data['seller_state'] = data['seller_state'].astype('category').cat.codes
data['product_category'] = data['product_category_name'].astype('category').cat.codes

features = ['volume', 'product_weight_g', 'customer_state', 'seller_state', 'same_state', 'product_category']
X = data[features].dropna()
y = data.loc[X.index, 'freight_value']

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predições e avaliação
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
tolerance = 0.2  # 20% de tolerância
accuracy = (abs(y_test - y_pred) / y_test <= tolerance).mean() * 100
print(f"R² Score (Teste): {r2_score(y_test, y_pred):.4f}")
print(f"MAE: R$ {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: R$ {rmse:.2f}")
print(f"Acertos (±20%): {accuracy:.2f}%")
print(f"Features: {features}")