import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# Code credit: Adapted from 
# https://learning.oreilly.com/library/view/introduction-to-machine/9781449369880/

def eval_on_features(features, target, regressor, n_train=184, sales_data=False, 
                     ylabel='Counts', 
                     feat_names="Default", 
                     impute=True):
    """
    Evaluate a regression model on a given set of features and target.

    This function splits the data into training and test sets, fits the 
    regression model to the training data, and then evaluates and plots 
    the performance of the model on both the training and test datasets.

    Parameters:
    -----------
    features : array-like
        Input features for the model.
    target : array-like
        Target variable for the model.
    regressor : model object
        A regression model instance that follows the scikit-learn API.
    n_train : int, default=184
        The number of samples to be used in the training set.
    sales_data : bool, default=False
        Indicates if the data is sales data, which affects the plot ticks.
    ylabel : str, default='Rentals'
        The label for the y-axis in the plot.
    feat_names : str, default='Default'
        Names of the features used, for display in the plot title.
    impute : bool, default=True
        whether SimpleImputer needs to be applied or not

    Returns:
    --------
    None
        The function does not return any value. It prints the R^2 score
        and generates a plot.
    """

    # Split the features and target data into training and test sets
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]

    if impute:
        simp = SimpleImputer()
        X_train = simp.fit_transform(X_train)
        X_test = simp.transform(X_test)
    
    # Fit the model on the training data
    regressor.fit(X_train, y_train)

    # Print R^2 scores for training and test datasets
    print("Train-set R^2: {:.2f}".format(regressor.score(X_train, y_train)))
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))

    # Predict target variable for both training and test datasets
    y_pred_train = regressor.predict(X_train)
    y_pred = regressor.predict(X_test)

    # Plotting
    plt.figure(figsize=(10, 3))

    # If not sales data, adjust x-ticks for dates (assumes datetime format)
    if not sales_data: 
        plt.xticks(range(0, len(X), 8), plt.xticks.strftime("%a %m-%d"), rotation=90, ha="left")

    # Plot training and test data, along with predictions
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, "-", label="test")
    plt.plot(range(n_train), y_pred_train, "--", label="prediction train")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, "--", label="prediction test")

    # Set plot title, labels, and legend
    title = regressor.__class__.__name__ + "\n Features= " + feat_names
    plt.title(title)
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel(ylabel)

# ---

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
    plt.show()