import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Normalização dos dados (necessário para PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA para redução de dimensionalidade
# Mantendo 95% da variância dos dados
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Dimensões originais: {X_train_scaled.shape[1]}")
print(f"Dimensões após PCA: {X_train_pca.shape[1]}")
print(f"Variância explicada: {pca.explained_variance_ratio_.sum():.4f}")
print(f"Features originais: {features}")
print()

# Definir os 3 modelos para comparação
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeRegressor(max_depth=15, min_samples_leaf=2, ccp_alpha=0.0, random_state=42)
}

# Dicionário para armazenar resultados
results = {}

# Treinar e avaliar cada modelo
for model_name, model in models.items():
    print(f"{'='*60}")
    print(f"Modelo: {model_name}")
    print(f"{'='*60}")
    
    # Treinar modelo
    model.fit(X_train_pca, y_train)
    
    # Predições
    y_pred = model.predict(X_test_pca)
    
    # Calcular as 3 métricas obrigatórias
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    
    # Calcular acurácia com tolerância
    tolerance = 0.2  # 20% de tolerância
    accuracy = (abs(y_test - y_pred) / y_test <= tolerance).mean() * 100
    
    # Armazenar resultados
    results[model_name] = {
        'R²': r2,
        'MAE': mae,
        'RMSE': rmse,
        'Acurácia (±20%)': accuracy
    }
    
    # Exibir métricas
    print(f"R² Score: {r2:.4f}")
    print(f"MAE (Mean Absolute Error): R$ {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): R$ {rmse:.2f}")
    print(f"Predições dentro de ±20%: {accuracy:.2f}%")
    print()

# Comparação final dos modelos
print(f"{'='*60}")
print("COMPARAÇÃO DOS MODELOS")
print(f"{'='*60}")
print(f"{'Modelo':<25} {'R²':<10} {'MAE':<12} {'RMSE':<12} {'Acurácia':<10}")
print(f"{'-'*60}")
for model_name, metrics in results.items():
    print(f"{model_name:<25} {metrics['R²']:<10.4f} R$ {metrics['MAE']:<9.2f} R$ {metrics['RMSE']:<9.2f} {metrics['Acurácia (±20%)']:<9.2f}%")

# Identificar o melhor modelo baseado em R²
best_model = max(results.items(), key=lambda x: x[1]['R²'])
print(f"\n🏆 Melhor modelo (por R²): {best_model[0]} com R² = {best_model[1]['R²']:.4f}")