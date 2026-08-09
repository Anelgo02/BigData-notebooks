# %% 

import sklearn

import pandas as pd
import numpy as np




# Import DATASET
data = pd.read_csv('heart_disease_uci.csv', sep=';')
df = data.copy(deep=True)

# %%
# Zeri sentinella: in questo dataset alcuni missing sono codificati come 0.
# chol=0 mg/dl e trestbps=0 mmHg sono fisiologicamente impossibili → NaN.
# NB: oldpeak=0 è legittimo (nessun sottoslivellamento ST), non va toccato.
print("Zeri sospetti prima:", (df[['chol', 'trestbps']] == 0).sum().to_dict())
df[['chol', 'trestbps']] = df[['chol', 'trestbps']].replace(0, np.nan)
print("Zeri sospetti dopo: ", (df[['chol', 'trestbps']] == 0).sum().to_dict())

# Analisi esplorativa per missing values
# %%
# Drop delle colonne non informative e che falserebbero i valori missing
X = df.drop(columns=['num', 'id'])


# Per colonne (%)
print("=== Sparsità Colonne Iniziale (%) ===")
print((X.isna().mean() * 100).sort_values(ascending=False))

# Per righe (%)
print("\n=== Sparsità Righe Iniziale (%) ===")
print((X.isna().mean(axis=1) * 100).sort_values(ascending=False))

# %%
# Le colonne 'ca' e 'thal' superano la soglia critica del 50%
# procediamo al drop delle colonne
X = X.drop(columns=['ca', 'thal'])

print("=== Sparsità Colonne dopo Drop 'ca' e 'thal' (%) ===")
print((X.isna().mean() * 100).sort_values(ascending=False))

# %%
# Identificazione dinamica e rimozione delle righe con percentuale missing > 40%
missing_rows_pct = X.isna().mean(axis=1) * 100
rows_to_drop = missing_rows_pct[missing_rows_pct > 40].index

X = X.drop(index=rows_to_drop)

print("=== Sparsità Righe Residua (%) ===")
print((X.isna().mean(axis=1) * 100).sort_values(ascending=False))
print(f"\nNuova dimensione del dataset: {X.shape}")

# %%
# Train Split Test
from sklearn.model_selection import train_test_split

# Estraiamo y a partire da X, cosi' non vi e' alcun disallineamento
y = df.loc[X.index, 'num']

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42, test_size=0.2)

# BLOCCO PREPROCESSING: IMPUTAZIONE -> ENCODING
# USIAMO PIPELINE + COLUMN TRANSFORMER

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Utilizziamo Pipeline + ColumnTransformer per un preprocessing eterogeneo


# 1.Identifichiamo le colonne su cui dobbiamo lavorare
cat_cols = X_train.select_dtypes(include=['object', 'bool', 'category']).columns
num_cols = X_train.columns.drop(cat_cols)

# Salviamo gli indici originali per poi ricomporre il DataFrame
# UPDATE : Non serve se si usa preprocessor.set_output(transform='pandas)
"""X_train_idx = X_train.index
X_test_idx = X_test.index"""

# 2. Prepariamo le pipeline per i due tipi di dati

# Pipeline Categorica
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary'))
])

# Pipeline Numerica
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Uniamo tutto nel column transformer

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

preprocessor.set_output(transform='pandas')

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(X_train_processed.shape, X_test_processed.shape)
print("NaN residui:", X_train_processed.isna().sum().sum())
print(list(X_train_processed.columns))
print(y_train.value_counts().sort_index())

# %% STANDARDIZZAZIONE

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# lo scaling va effettuato solo sulle features continue
cont_cols = [c for c in X_train_processed.columns if X_train_processed[c].nunique() > 2]

X_train_s = X_train_processed.copy()
X_test_s = X_test_processed.copy()

X_train_s[cont_cols] = scaler.fit_transform(X_train_processed[cont_cols])
X_test_s[cont_cols] = scaler.transform(X_test_processed[cont_cols])

print(X_train_s[cont_cols].mean().round(3))
print(X_train_s[cont_cols].std().round(3))


# %% ANALISI DELLE FEATURES MULTICOLLINEARI

import matplotlib.pyplot as plt
import seaborn as sns


# 1. Matrice di correlazione
corr_matrix = X_train_s.corr().abs()

plt.figure(figsize=(14,10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    vmin=-1, vmax=1,
    linewidths=0.5
)

plt.title("Matrice di correlazione tra le features (POST-PROCESSING)")
plt.tight_layout()
plt.show()

# 2. Prendiamo il triangolo superiore per evitare duplicati
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 3. Individuazione delle features con corr > 0.85

threshold = 0.85
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

print(f"Feature fortemente correlate da rimuovere (soglia > {threshold}):", to_drop)

# 4. Drop delle feature collineari sia da train che da test
X_train_s = X_train_s.drop(columns=to_drop)
X_test_s = X_test_s.drop(columns=to_drop)

print("Nuovo numero di feature dopo il drop multicollineare:", X_train_s.shape[1])

# %% FEATURE SELECTION

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# 1. Istanziamo il selettore per estrarre le top-k feature più informative (es. k=10)
selector = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42))
selector.set_output(transform='pandas')

X_train_final = selector.fit_transform(X_train_s, y_train)
X_test_final = selector.transform(X_test_s)

# Visualizzazione delle feature selezionate e dei rispettivi F-scores
selected_features = list(X_train_final.columns)
scores = pd.Series(
    selector.estimator_.feature_importances_,
    index=X_train_s.columns
    ).sort_values(ascending=False
)

print("Features selezionate", selected_features)
print(scores.head(10))


# %%
