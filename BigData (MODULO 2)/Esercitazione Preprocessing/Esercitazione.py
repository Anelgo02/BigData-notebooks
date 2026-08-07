# %% 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Import DATASET
data = pd.read_csv('heart_disease_uci.csv', sep=';')
df = data.copy(deep=True)

# Analisi esplorativa per missing values

# Per colonne (%)
print("=== Sparsità Colonne Iniziale (%) ===")
print((df.isna().mean() * 100).sort_values(ascending=False))

# Per righe (%)
print("\n=== Sparsità Righe Iniziale (%) ===")
print((df.isna().mean(axis=1) * 100).sort_values(ascending=False))

# %%
# Le colonne 'ca' e 'thal' superano la soglia critica del 50%
# procediamo al drop delle colonne
df = df.drop(columns=['ca', 'thal'])

print("=== Sparsità Colonne dopo Drop 'ca' e 'thal' (%) ===")
print((df.isna().mean() * 100).sort_values(ascending=False))

# %%
# Identificazione dinamica e rimozione delle righe con percentuale missing > 40%
missing_rows_pct = df.isna().mean(axis=1) * 100
rows_to_drop = missing_rows_pct[missing_rows_pct > 40].index

df = df.drop(index=rows_to_drop)

print("=== Sparsità Righe Residua (%) ===")
print((df.isna().mean(axis=1) * 100).sort_values(ascending=False))
print(f"\nNuova dimensione del dataset: {df.shape}")

# %%