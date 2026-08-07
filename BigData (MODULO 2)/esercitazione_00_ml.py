import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# PARTE 1: Caricamento dati e teoria degli Stimatori 

print("---PARTE 1: Dati e Stimatori ---")
# Carichiamo il dataset 
X, y = load_breast_cancer(return_X_y=True)

#Estraiamo una singola feature, prima colonna per esempio 
feature_0 = X[:, 0]

#Calcoliamo lo stimatore campionario della media (valore atteso) e della varianza
#La media campionaria è uno stimatore non polarizzato 

media_campionaria = np.mean(feature_0)

#Usiamo ddof=1 per lo stimatore della varianza non polarizzato (m-1 al denominatore)

varianza_campionaria = np.var(feature_0, ddof = 1)

std_error = np.sqrt(varianza_campionaria)