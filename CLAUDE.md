# Contesto del progetto

Corso di Big Data — Modulo 2. L'utente si chiama Angelo e sta imparando l'analisi dati con Python/pandas/sklearn.
Approccio preferito: guidare senza dare le risposte complete, fare domande per stimolare il ragionamento.

## Notebook in corso

`MODULO 2/esercitazione_01.ipynb` — Analisi del dataset **California Housing** (sklearn).

Il notebook `esercitazione_00.ipynb` copre già:
1. Caricamento e ispezione del dataset
2. Analisi distribuzioni (istogrammi, skewness, curtosi, trasformazione log)
3. Correlazione e multicollinearità (heatmap, feature selection)
4. Train/test split
5. Pipeline con StandardScaler + LinearRegression
6. Analisi dei residui
7. Cross-validation e confronto OLS / Ridge / Lasso

## Stato esercizi in `esercitazione_01.ipynb`

- [x] **Esercizio 1** — Esplora `AveOccup`: skewness (97.6 → 2.12 dopo log), curtosi (12418 → 34.64 dopo log), istogramma comparativo originale vs log. Completato.
- [ ] **Esercizio 2** — Feature selection: rimuovi `AveBedrms` (ridondante con `AveRooms`), riaddestra la pipeline, confronta R² e RMSE con e senza quella colonna.
- [ ] **Esercizio 3** — Tuning alpha di Ridge: prova alpha in [0.01, 0.1, 1, 10, 100], grafico MSE_cv vs alpha.
- [ ] **Esercizio 4** — Aggiungi `RandomForestRegressor` al confronto modelli in cross-validation.
- [ ] **Esercizio 5** — Analisi geografica: scatter plot Latitude/Longitude colorato da `MedHouseVal`.

## Prossimo passo

Iniziare **Esercizio 2**: rimuovere `AveBedrms` dal DataFrame, ricostruire la pipeline StandardScaler + LinearRegression, confrontare R² e RMSE prima e dopo la rimozione.
