

# %% Quesito 1: split, imputazione, encoding, scaling, multicollinearità, feature selection
import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/train.csv")


# %% ============ Analisi Esplorativa ==============

# 1. Dimensioni del dataset
print(df.shape)

# 2. tipi di dato per colonna, dice subito cosa è numerico e cosa è testo
print(df.dtypes)

# 3. percentuale di missing per colonna, ordinata dal peggiore
missing_pct = df.isnull().mean().sort_values(ascending=False) * 100
print(missing_pct)

# 4. cardinalità: quanti valori distinti ha ogni colonna
print(df.nunique().sort_values(ascending=False))

# 5. statistiche descrittive delle colonne numeriche
print(df.describe())

# 6. anteprima dei valori per farti un'idea del contenuto
print(df.head())
print(df.tail())




# %% ====================================================


# Split 90/10 stratificato, PRIMA di ogni fit imputazione o scaling

train_df, val_df = train_test_split(df, test_size=0.10, random_state=42, stratify=df['Survived'])

# %% Drop delle colonne troppo sparse o non informative nella forma grezza
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']

# Drop=True scarta il vecchio indice e non crea una colonna nuova `index`
train_df = train_df.drop(columns=cols_to_drop).reset_index(drop=True)
val_df = val_df.drop(columns=cols_to_drop).reset_index(drop=True)

# ===== IMPUTAZIONE ==========

# 1. Imputazione Age condizionata da Sex e Pclass, fittata solo sul training
age_medians = train_df.groupby(['Pclass', 'Sex'])['Age'].median()
# age_medians sara' una Series con doppio indice ['Pclass', 'Sex']

def impute_age(row, medians):
    if pd.isna(row['Age']):
        # Qui gli sto dicendo di ritornare la mediana tramite i due indici per accedere all'elemento
        # della tabella
        return medians.loc[row['Pclass'], row['Sex']]
    return row['Age']
# axis=1 applica riga per riga 
train_df['Age'] = train_df.apply(lambda r: impute_age(r, age_medians), axis=1)
val_df['Age'] = val_df.apply(lambda r: impute_age(r, age_medians), axis=1)



# 2. Imputazione `Embarked` con la moda del training
m = train_df['Embarked'].mode()
embarked_mode = m.iloc[0] if not m.empty else 'S' 
# Poichè mode() restituisce una Series nel caso in cui il primo elemento 
# sia vuoto evitiamo l'errore con un fallback in 'S'
train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode) 
val_df['Embarked'] = val_df['Embarked'].fillna(embarked_mode) 


# ======== ENCODING ==========

# encoding Sex: binaria, mappa manuale
sex_map = {'male':0, 'female':1}
train_df["Sex"] = train_df['Sex'].map(sex_map)
val_df["Sex"] = val_df['Sex'].map(sex_map)

# encoding Embarked: one-hot, fittato sulle categorie viste nel training

# pd.get_dummies prende una colonna categorica e la sostituisce con tante
# colonne binarie quanti sono i valori distinti presenti in quella colonna.

# Facendo encoding one-hot si verifica la Dummy Variable Trap:
# Multicollinearita' strutturale, ovvero le tre variabili Emb_ sono linearmente
# dipendenti.

# SOLUZIONE -> drop_first
train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Emb', drop_first=True)
val_df = pd.get_dummies(val_df, columns=['Embarked'], prefix='Emb', drop_first=True)

# allineamento colonne: se nel val set manca una categoria rara vista solo nel train
val_df = val_df.reindex(columns=train_df.columns, fill_value=0)

# Multicollinearita'
"""
Parch e SibSp concettualmente misurano la stessa cosa, non e' un caso
che siano correlate a 0.41, nonostante il valore sia sotto la soglia di 0.60 
scelta per individuare multicollinearita' decidiamo di intervenire lo stesso. 

La combinazione lineare naturale e' la dimensione totale del nucleo familiare a bordo
"""

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
val_df['FamilySize'] = val_df['SibSp'] + val_df['Parch'] + 1

# Il +1 conta il passeggero stesso, altrimenti un viaggiatore solitario 
# risulterebbe con FamilySize = 0, che è concettualmente sbagliato,
# la dimensione minima di un nucleo che include se stessi è 1.

train_df = train_df.drop(columns=['SibSp', 'Parch'])
val_df = val_df.drop(columns=['SibSp', 'Parch'])

print(train_df.dtypes)
print(train_df.head())

# %% ============= SCALING E MATRICE DI CORRELAZIONE

# Effettuiamo lo scaling delle featurse continue, non includiamo le binarie come Embarked e Sex
target_col = 'Survived'
feature_cols = [c for c in train_df.columns if c != target_col]
binary_cols = ['Sex'] + [c for c in feature_cols if c.startswith('Emb_')]
continuous_cols = [c for c in feature_cols if c not in binary_cols]

print(feature_cols)
print(continuous_cols)
print(binary_cols)

scaler = StandardScaler()
train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols])
val_df[continuous_cols] = scaler.transform(val_df[continuous_cols])

# la matrice di correlazione si calcola su tutte le features tranne l'etichetta
corr_matrix = train_df[feature_cols].corr()
print(corr_matrix.round(2))

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice di correlazione feature (training set)')
plt.show()


# %% ===============FEATURE SELECTION================
"""
procedere  alla  feature  selection attraverso  una  tecnica embedded  
che  impieghi  un  classificatore  come modello. La scelta del modello
embedded è lasciata al candidato, tenendo conto che il problema è di 
classificazione binaria. """

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_val = val_df[feature_cols]
y_val = val_df[target_col]

embedded_model = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)
embedded_model.fit(X_train, y_train)

coef = pd.Series(embedded_model.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
print(coef)

selector = SelectFromModel(estimator=embedded_model, prefit=True, threshold='median')
selected_mask = selector.get_support()
selected_features = [f for f,m in zip(feature_cols, selected_mask) if m]
print(selected_features)

# %%
