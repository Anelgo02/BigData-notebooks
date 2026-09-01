# %% 
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

rng = np.random.default_rng(42)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

# Teniamo solo alcune colonne per semplicita'
cols_to_keep = ['mean radius', 'mean texture', 'mean smoothness', 'mean symmetry', 'target']
df = df[cols_to_keep].copy()

# Colonna categorica sintetica, 3 livelli, distribuzione non uniforme (come nella realtà)
regioni = rng.choice(['Nord', 'Centro', 'Sud'], size=len(df), p=[0.5, 0.3, 0.2])
df['regione'] = regioni

# Missing artificiali su due colonne numeriche (~8% ciascuna)
for col in ['mean radius', 'mean smoothness']:
    mask = rng.random(len(df)) < 0.08
    df.loc[mask, col] = np.nan

# %%
print(df.shape)
print(df.dtypes)
print(df.isnull().mean().sort_values(ascending=False) * 100)
print(df.head())


# %% FASE 1: Effettuiamo lo split stratificato 

from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, stratify=y, random_state=42)


# %% IMPUTAZIONE (senza Sklearn)
# correlazione con le altre numeriche
print(X_train[['mean radius', 'mean texture', 'mean smoothness', 'mean symmetry']].corr())

# la media di 'mean radius' cambia in base a 'regione'?
print(X_train.groupby('regione')['mean radius'].median())

print(X_train[['mean radius', 'mean smoothness']].skew())
print(X_train.groupby('regione')['mean smoothness'].median())

# %% La distribuzione di mean radius non è simmetrica, useremo la mediana
# La mediana con groupby non cambia molto, non c'e' un reale motivo per usare groupby

# Si effettua il calcolo una sola volta prima e poi lo si usa => prevenire DATA LEAKAGE
median_radius = X_train['mean radius'].median() 
median_smoothness = X_train['mean smoothness'].median() 

X_train['mean radius'] = X_train['mean radius'].fillna(median_radius) 
X_test['mean radius'] = X_test['mean radius'].fillna(median_radius) 

X_train['mean smoothness'] = X_train['mean smoothness'].fillna(median_smoothness) 
X_test['mean smoothness'] = X_test['mean smoothness'].fillna(median_smoothness)

print(X_train.isnull().sum())
print(X_test.isnull().sum())


# FASE 2: Per effettuare l'imputazione in modo semplice si può usare simpleImputer di sklearn
# %% IMPUTAZIONE
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
cols_to_impute = ['mean radius','mean smoothness']

imputer.fit(X_train[cols_to_impute])

X_train[cols_to_impute] = imputer.transform(X_train[cols_to_impute])
X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

print(X_train.isnull().sum())
print(X_test.isnull().sum())

# %%
# Prossimo step: econding variabile Regione (da categorica a numerica)
from sklearn.preprocessing import OneHotEncoder

cat_cols = ['regione']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
encoder.fit(X_train[cat_cols])

# trasforma solo la parte categorica
train_encoded = encoder.transform(X_train[cat_cols])
test_encoded = encoder.transform(X_test[cat_cols])

# ricostruisci i nomi di colonna generati automaticamente (es. 'regione_Nord', 'regione_Sud')
encoded_cols = encoder.get_feature_names_out(cat_cols)
print(encoded_cols)

train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=X_test.index)

# riunisci con le colonne numeriche originali, eliminando la colonna 'regione' testuale
X_train = pd.concat([X_train.drop(columns=cat_cols), train_encoded_df], axis=1)
X_test = pd.concat([X_test.drop(columns=cat_cols), test_encoded_df], axis=1)

print(X_train.head())

# SCALING
# %%
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

continuous_cols = ['mean radius', 'mean texture', 'mean smoothness', 'mean symmetry' ] 

X_train[continuous_cols] = sc.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = sc.transform(X_test[continuous_cols])

print(X_train.head())

# %%
