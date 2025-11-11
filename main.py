import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
olist_customers_dataset = pd.read_csv('data/olist_customers_dataset.csv')
olist_geolocation_dataset = pd.read_csv('data/olist_geolocation_dataset.csv')
olist_order_items_dataset = pd.read_csv('data/olist_order_items_dataset.csv')
olist_order_payments_dataset = pd.read_csv('data/olist_order_payments_dataset.csv')
olist_order_reviews_dataset = pd.read_csv('data/olist_order_reviews_dataset.csv')
olist_orders_dataset = pd.read_csv('data/olist_orders_dataset.csv')
olist_products_dataset = pd.read_csv('data/olist_products_dataset.csv')
olist_sellers_dataset = pd.read_csv('data/olist_sellers_dataset.csv')
product_category_name_translation = pd.read_csv('data/product_category_name_translation.csv')

# Aprendizado Supervisionado
# Regressão Linear OLS (Base Model)
# Preço do Frete X Volume, Peso, Estado, Quantidade de Itens, Categoria do Produto
olist_products_dataset['volume'] = olist_products_dataset['product_length_cm'] * olist_products_dataset['product_height_cm'] * olist_products_dataset['product_width_cm']

# Preencher valores NaN com "Indefinido"
olist_products_dataset['product_category_name'].fillna('Indefinido', inplace=True)

# One-Hot Encoding da categoria do produto
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
category_encoded = encoder.fit_transform(olist_products_dataset[['product_category_name']])
category_encoded_df = pd.DataFrame(
    category_encoded, 
    columns=encoder.get_feature_names_out(['product_category_name']),
    index=olist_products_dataset.index
)

# Concatenar as colunas codificadas com o dataset original
olist_products_dataset = pd.concat([olist_products_dataset, category_encoded_df], axis=1)

# Merge dos datasets para obter o preço do frete
order_items_with_products = olist_order_items_dataset.merge(
    olist_products_dataset, 
    on='product_id', 
    how='inner'
)

# Selecionar features: volume e todas as colunas de categorias
category_columns = [col for col in olist_products_dataset.columns if col.startswith('product_category_name_')]
features = ['volume'] + category_columns

# Remover linhas com valores NaN nas features ou target
X = order_items_with_products[features].dropna()
y = order_items_with_products.loc[X.index, 'freight_value']

# Aplicar PCA para redução de dimensionalidade (80% de variância)
pca = PCA(n_components=0.8)
X_pca = pca.fit_transform(X)

print(f"Features originais: {len(features)}")
print(f"Componentes principais após PCA: {pca.n_components_}")
print(f"Variância explicada: {pca.explained_variance_ratio_.sum():.4f}")

# Criar e treinar o modelo de regressão linear com dados reduzidos
model = linear_model.LinearRegression()
model.fit(X_pca, y)

print(f"R² Score: {model.score(X_pca, y):.4f}")

print(f"Coeficientes - Volume: {model.coef_[0]:.6f}")

print("Customers Dataset:")
print(olist_customers_dataset.head())
print("Geolocation Dataset:")
print(olist_geolocation_dataset.head())
print("Order Items Dataset:")
print(olist_order_items_dataset.head())
print("Order Payments Dataset:")
print(olist_order_payments_dataset.head())
print("Order Reviews Dataset:")
print(olist_order_reviews_dataset.head())
print("Orders Dataset:")
print(olist_orders_dataset.head())
print("Products Dataset:")
print(olist_products_dataset.head())
print("Sellers Dataset:")
print(olist_sellers_dataset.head())
print("Product Category Name Translation Dataset:")
print(product_category_name_translation.head())

print(olist_products_dataset.columns)