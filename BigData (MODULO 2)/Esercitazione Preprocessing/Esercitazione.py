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

# %% MULTICOLLINEARITÀ
import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix = X_train_s.corr().abs()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='Reds', vmin=0, vmax=1,
            square=True, linewidths=0.3)
plt.title("Matrice di correlazione (|r|) — training set")
plt.tight_layout(); plt.show()

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

threshold = 0.85
coppie = [(i, j, upper_tri.loc[i, j])
          for i in upper_tri.index for j in upper_tri.columns
          if pd.notna(upper_tri.loc[i, j]) and upper_tri.loc[i, j] > threshold]
print(f"Coppie con |r| > {threshold}:")
for a, b, v in sorted(coppie, key=lambda t: -t[2]):
    print(f"  {a:35s} — {b:35s}  r={v:.3f}")

# NOTA: Pearson è bivariata: non intercetta la dipendenza lineare residua dei
# gruppi one-hot multi-classe, che sommano a 1 per riga. Con un modello lineare
# non regolarizzato servirebbe drop='first' o un'analisi VIF.

to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > threshold)]
print("\nRimosse:", to_drop)

X_train_nc = X_train_s.drop(columns=to_drop)   # nomi nuovi: cella ri-eseguibile
X_test_nc  = X_test_s.drop(columns=to_drop)
print("Feature residue:", X_train_nc.shape[1])

# %% FEATURE SELECTION — EMBEDDED (RandomForest, importanze MDI)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Binarizzazione: le classi 3 (75) e 4 (21) sono troppo rare per una
# selezione supervisionata stabile sul multiclasse.
y_train_bin = (y_train > 0).astype(int)
y_test_bin  = (y_test  > 0).astype(int)

# EMBEDDED: la selezione è un sottoprodotto di un singolo training.
# threshold='mean' (default) tiene le feature con importanza > media.
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
)
selector.set_output(transform='pandas')

X_train_final = selector.fit_transform(X_train_nc, y_train_bin)
X_test_final  = selector.transform(X_test_nc)

scores = pd.Series(selector.estimator_.feature_importances_,
                   index=X_train_nc.columns).sort_values(ascending=False)

print(f"Selezionate {X_train_final.shape[1]}/{X_train_nc.shape[1]}:")
print(list(X_train_final.columns))
print("\nRanking importanze (MDI):")
print(scores.round(4))
# NOTA: MDI favorisce le feature ad alta cardinalità (le continue hanno più
# split candidati delle dummy binarie). permutation_importance sarebbe meno
# distorto, al costo di N ri-valutazioni del modello.

print("\nColonne train==test:", list(X_train_final.columns) == list(X_test_final.columns))