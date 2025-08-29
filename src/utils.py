import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

def naive_lag1_global(series, frac_train=0.8, ylabel="Visitors"):
    """
    Naive lag-1 baseline forecaster.
    series: pd.Series indexed by date, e.g. daily total visitors.
    Returns (r2, rmse).
    """
    series = series.sort_index()
    n_train = int(len(series) * frac_train)
    train, test = series.iloc[:n_train], series.iloc[n_train:]

    # Predictions = yesterday’s value
    preds = np.empty(len(test), dtype=float)
    preds[0] = train.iloc[-1]
    if len(test) > 1:
        preds[1:] = test.values[:-1]

    # Metrics
    r2 = r2_score(test, preds)
    rmse = mean_squared_error(test, preds)
    print(f"Naive Lag-1 → R²={r2:.3f}, RMSE={rmse:.2f}")

    # --- Plot ---
    plt.figure(figsize=(12, 4))
    plt.plot(train.index, train.values, label="Train", color="blue")
    plt.plot(test.index, test.values, label="Test (actual)", color="black")
    plt.plot(test.index, preds, "--", label="Naive lag-1 pred", color="red")

    plt.title(f"Naive Lag-1 Baseline\nR²={r2:.3f}, RMSE={rmse:.2f}")
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    return r2, rmse, test, pd.Series(preds, index=test.index)

# Code credit: Adapted from 
# https://learning.oreilly.com/library/view/introduction-to-machine/9781449369880/

def eval_on_features(features, target, regressor, n_train_frac=0.8, sales_data=False, 
                     ylabel='Counts', feat_names="Default", impute=True):
    """
    Evaluate a regression model on a given set of features and target.

    Splits chronologically: first n_train_frac portion = train, rest = test.
    Fits regressor, prints R², and plots actual vs predictions.
    """
    # Compute train size from fraction
    n_train = int(len(features) * n_train_frac)

    # Split the features and target data into training and test sets
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]

    if impute:
        simp = SimpleImputer()
        X_train = simp.fit_transform(X_train)
        X_test = simp.transform(X_test)
    
    # Fit the model on the training data
    regressor.fit(X_train, y_train)

    # Print R^2 scores
    print("Train-set R^2: {:.2f}".format(regressor.score(X_train, y_train)))
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))

    # Predictions
    y_pred_train = regressor.predict(X_train)
    y_pred = regressor.predict(X_test)

    # Plotting
    plt.figure(figsize=(10, 3))
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, n_train + len(y_test)), y_test, "-", label="test")
    plt.plot(range(n_train), y_pred_train, "--", label="prediction train")
    plt.plot(range(n_train, n_train + len(y_test)), y_pred, "--", label="prediction test")

    title = regressor.__class__.__name__ + "\n Features= " + feat_names
    plt.title(title)
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Time index")
    plt.ylabel(ylabel)
    plt.tight_layout()
    
# ---

def naive_lag1_per_store(df, frac_train=0.8, ylabel="Visitors"):
    """
    Simple naive lag-1 baseline per store.
    df must have: ['air_store_id','visit_date','visitors'].
    Returns (r2, rmse).
    """
    df = df.sort_values(["air_store_id","visit_date"])

    # Collect per-store splits
    y_tr_list, y_te_list, pred_tr_list, pred_te_list = [], [], [], []
    for sid, g in df.groupby("air_store_id"):
        n_train = int(len(g) * frac_train)
        if n_train < 1 or n_train >= len(g):  
            continue

        train = g.iloc[:n_train]
        test  = g.iloc[n_train:].copy()

        # predictions
        pred_test = test["visitors"].shift(1).fillna(train["visitors"].iloc[-1])

        # Save
        y_tr_list.append(train["visitors"])
        y_te_list.append(test["visitors"])
        pred_tr_list.append(train["visitors"].shift(1).fillna(method="bfill"))  # trivial for train
        pred_te_list.append(pred_test)

    # Recombine across stores
    y_train = pd.concat(y_tr_list, axis=0)
    y_test  = pd.concat(y_te_list, axis=0)
    pred_train = pd.concat(pred_tr_list, axis=0)
    pred_test  = pd.concat(pred_te_list, axis=0)

    # Metrics
    r2   = r2_score(y_test, pred_test)
    rmse = mean_squared_error(y_test, pred_test)
    print(f"Naive lag-1 per-store → R²={r2:.3f}, RMSE={rmse:.2f}")

    # Plot concatenated style
    plt.figure(figsize=(12,4))
    plt.plot(range(len(y_train)), y_train, label="Train")
    plt.plot(range(len(y_train), len(y_train)+len(y_test)), y_test, "-", label="Test")
    plt.plot(range(len(y_train)), pred_train, "--", label="Pred train")
    plt.plot(range(len(y_train), len(y_train)+len(y_test)), pred_test, "--", label="Pred test")
    plt.title(f"Naive Lag-1 per-store baseline\nR²={r2:.3f}, RMSE={rmse:.2f}")
    plt.xlabel("Concatenated per-store time index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    return r2, rmse

def eval_on_features_per_store(features, target, groups, regressor, frac_train=0.8, ylabel="Counts", feat_names="Default"):
    """
    Train/test evaluation for per-store time-series.
    Each store gets its own chronological split (first frac_train = train).
    Then recombine for one global model fit.
    """
    from sklearn.impute import SimpleImputer
    
    # Split each store separately
    X_tr_list, X_te_list, y_tr_list, y_te_list = [], [], [], []
    for sid in np.unique(groups):
        mask = (groups == sid)
        Xg, yg = features[mask], target[mask]

        n_tr = int(len(Xg) * frac_train)
        if n_tr <= 0 or n_tr >= len(Xg):
            continue  # skip too-short stores

        X_tr_list.append(Xg.iloc[:n_tr])
        X_te_list.append(Xg.iloc[n_tr:])
        y_tr_list.append(yg.iloc[:n_tr])
        y_te_list.append(yg.iloc[n_tr:])

    # Recombine
    X_train = pd.concat(X_tr_list, axis=0)
    X_test  = pd.concat(X_te_list, axis=0)
    y_train = pd.concat(y_tr_list, axis=0)
    y_test  = pd.concat(y_te_list, axis=0)

    # Impute
    simp = SimpleImputer()
    X_train = simp.fit_transform(X_train)
    X_test  = simp.transform(X_test)

    # Fit
    regressor.fit(X_train, y_train)

    # Score
    print("Train-set R^2: {:.2f}".format(regressor.score(X_train, y_train)))
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))

    # Predict
    y_pred_train = regressor.predict(X_train)
    y_pred_test  = regressor.predict(X_test)

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(range(len(y_train)), y_train, label="Train")
    plt.plot(range(len(y_train), len(y_train)+len(y_test)), y_test, "-", label="Test")
    plt.plot(range(len(y_train)), y_pred_train, "--", label="Pred train")
    plt.plot(range(len(y_train), len(y_train)+len(y_test)), y_pred_test, "--", label="Pred test")
    plt.title(regressor.__class__.__name__ + "\nPer-store split • Features=" + str(feat_names))
    plt.legend()
    plt.xlabel("Concatenated per-store time index")
    plt.ylabel(ylabel)
    plt.tight_layout()