import numpy as np
import pandas as pd
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def OLS(returns, factRet):
    """
    OLS factor model.
    Returns:
        mu      : (n,1) expected returns
        Q       : (n,n) covariance matrix
        R2      : (n,1) in-sample R² for each asset
        R2_adj  : (n,1) adjusted R² for each asset
    """
    T, p = factRet.shape
    n = returns.shape[1]

    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)  # shape (T, p+1)
    B = np.linalg.solve(X.T @ X, X.T @ returns)  # shape (p+1, n)

    a = B[0, :]
    V = B[1:, :]

    Y_hat = X @ B  # shape (T, n)
    residuals = returns.values - Y_hat
    sigma_ep = 1 / (T - p - 1) * np.sum(residuals ** 2, axis=0)
    D = np.diag(sigma_ep)

    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D
    Q = (Q + Q.T) / 2  # ensure symmetry

    # Compute R² and Adjusted R²
    ss_res = np.sum(residuals ** 2, axis=0)
    ss_tot = np.sum((returns.values - returns.values.mean(axis=0)) ** 2, axis=0)
    R2 = 1 - ss_res / ss_tot

    k = p  # number of predictors (exclude intercept)
    R2_adj = 1 - (ss_res / (T - k - 1)) / (ss_tot / (T - 1))

    return mu, Q, R2.reshape((n, 1)), R2_adj.reshape((n, 1))

def FF3(returns, factRet):
    """
    Fama-French 3-Factor model.
    Uses only the first 3 columns of the factor returns.
    Returns:
        mu       : (n,1) expected returns
        Q        : (n,n) covariance matrix
        R2       : (n,1) in-sample R²
        R2_adj   : (n,1) adjusted R²
    """
    T, n = returns.shape
    X_ff3 = factRet.iloc[:, :3].values  # use first 3 factors
    p = X_ff3.shape[1]

    X = np.concatenate([np.ones((T, 1)), X_ff3], axis=1)  # intercept + 3 factors
    B = np.linalg.solve(X.T @ X, X.T @ returns)  # shape (p+1, n)

    a = B[0, :]
    V = B[1:, :]

    Y_hat = X @ B
    residuals = returns.values - Y_hat
    sigma_ep = 1 / (T - p - 1) * np.sum(residuals ** 2, axis=0)
    D = np.diag(sigma_ep)

    f_bar = np.expand_dims(X_ff3.mean(axis=0), 1)
    F = np.cov(X_ff3, rowvar=False)
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D
    Q = (Q + Q.T) / 2

    # R² and Adjusted R²
    ss_res = np.sum(residuals ** 2, axis=0)
    ss_tot = np.sum((returns.values - returns.values.mean(axis=0)) ** 2, axis=0)
    R2 = 1 - ss_res / ss_tot
    k = p  # 3 factors (no intercept counted)
    R2_adj = 1 - (ss_res / (T - k - 1)) / (ss_tot / (T - 1))

    return mu, Q, R2.reshape((n, 1)), R2_adj.reshape((n, 1))


def cross_validate_lasso(X, y, lam_values, k=4):
    """
    Perform deterministic K-fold cross-validation to select the best lambda for Lasso.

    Inputs:
        X           : NumPy array or DataFrame (T x p), factor matrix (already normalized)
        y           : Pandas Series (T,), excess return for one asset
        lam_values  : list/array of candidate lambda values
        k           : number of CV folds (default 4)

    Returns:
        best_lambda : lambda value with lowest average validation error
    """
    # Ensure X is a NumPy array
    if isinstance(X, pd.DataFrame):
        X = X.values

    T = len(y)
    fold_size = T // k
    indices = np.arange(T)  # No shuffling for deterministic split

    avg_val_errors = []

    for lam in lam_values:
        val_errors = []

        for j in range(k):
            val_idx = indices[j * fold_size : (j + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y.iloc[train_idx].values
            y_val = y.iloc[val_idx].values

            beta = cp.Variable(X.shape[1])
            objective = cp.Minimize(cp.sum_squares(X_train @ beta - y_train) + lam * cp.norm1(beta))
            problem = cp.Problem(objective)
            problem.solve(solver=cp.OSQP)

            y_pred = X_val @ beta.value
            val_error = np.mean((y_val - y_pred) ** 2)
            val_errors.append(val_error)

        avg_error = np.mean(val_errors)
        avg_val_errors.append(avg_error)

    # Find best lambda (smallest CV error)
    min_index = np.argmin(avg_val_errors)
    best_lambda = lam_values[min_index]

    return best_lambda

def LASSO(returns, factRet, lam):
    """
    Lasso factor regression.
    Returns:
        mu       : (n,1) expected excess returns
        Q        : (n,n) covariance matrix
        R2       : (n,1) in-sample R²
        R2_adj   : (n,1) adjusted R²
    """
    T, n = returns.shape
    p = factRet.shape[1]

    # Design matrix with intercept
    X = np.column_stack([np.ones(T), factRet.values])  # shape (T, p+1)

    # Containers
    B_hat = np.zeros((p + 1, n))
    residuals = np.zeros((T, n))
    R2 = np.zeros((n, 1))
    R2_adj = np.zeros((n, 1))

    for j in range(n):
        y = returns.iloc[:, j].values

        beta = cp.Variable(p + 1)
        objective = cp.Minimize(cp.sum_squares(X @ beta - y) + lam * cp.norm1(beta))
        prob = cp.Problem(objective)
        prob.solve(solver=cp.OSQP)

        b = beta.value
        B_hat[:, j] = b
        y_hat = X @ b
        res = y - y_hat
        residuals[:, j] = res

        # R² and adjusted R²
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        R2[j, 0] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Count non-zero coefficients excluding intercept
        k = np.sum(np.abs(b[1:]) > 1e-6)
        if ss_tot > 0 and T - k - 1 > 0:
            R2_adj[j, 0] = 1 - (ss_res / (T - k - 1)) / (ss_tot / (T - 1))
        else:
            R2_adj[j, 0] = np.nan

    # Compute mu and Q
    a = B_hat[0, :]
    V = B_hat[1:, :]

    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar

    sigma_ep = np.var(residuals, axis=0)
    Q = V.T @ F @ V + np.diag(sigma_ep)
    Q = (Q + Q.T) / 2

    return mu, Q, R2, R2_adj


def BSS(returns, factorRet, K):
    """
    Best Subset Selection (BSS) via MIQP.

    Inputs:
        returns   : (T x n) DataFrame of excess returns
        factorRet : (T x p) DataFrame of factor returns (excluding RF)
        K         : max number of predictors allowed (excluding intercept)

    Returns:
        mu      : (n, 1) expected excess returns
        Q       : (n, n) covariance matrix
        R2      : (n, 1) in-sample R²
        R2_adj  : (n, 1) in-sample adjusted R²
    """
    T, n = returns.shape
    p = factorRet.shape[1]
    X = np.column_stack([np.ones(T), factorRet.values])  # T x (p+1)

    B_all = np.zeros((n, p + 1))
    residuals = np.zeros((T, n))
    R2 = np.zeros((n, 1))
    R2_adj = np.zeros((n, 1))

    for i in range(n):
        r_i = returns.iloc[:, i].values
        B = cp.Variable(p + 1)
        y = cp.Variable(p + 1, boolean=True)
        M = 5.0  # Big-M constant

        constraints = [
            B <=  M * y,
            B >= -M * y,
            cp.sum(y[1:]) <= K  # exclude intercept
        ]

        obj = cp.Minimize(cp.sum_squares(r_i - X @ B))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GUROBI)

        b_hat = B.value
        B_all[i, :] = b_hat
        y_hat = X @ b_hat
        res = r_i - y_hat
        residuals[:, i] = res

        # Compute R² and adjusted R²
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((r_i - np.mean(r_i)) ** 2)
        R2[i, 0] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Count number of nonzero coefficients excluding intercept
        k = np.sum(np.abs(b_hat[1:]) > 1e-6)
        if ss_tot > 0 and T - k - 1 > 0:
            R2_adj[i, 0] = 1 - (ss_res / (T - k - 1)) / (ss_tot / (T - 1))
        else:
            R2_adj[i, 0] = np.nan

    # Compute mu
    mean_factors = factorRet.mean(axis=0).values  # (p,)
    mu = B_all[:, 0] + B_all[:, 1:] @ mean_factors
    mu = mu.reshape((n, 1))

    # Compute Q
    cov_factors = np.cov(factorRet.values, rowvar=False)
    B_factors = B_all[:, 1:]
    Q_factor = B_factors @ cov_factors @ B_factors.T
    var_resid = np.var(residuals, axis=0)
    Q = Q_factor + np.diag(var_resid)
    Q = (Q + Q.T) / 2

    return mu, Q, R2, R2_adj

## Helper function that returns the best number of components
def select_n_components_purged_cv(
    returns: pd.DataFrame,
    factorRet: pd.DataFrame,
    k_list: list,
    train_size: int,
    test_size: int,
    purge_size: int
) -> int:
    """
    Purged time-series CV to choose PCA n_components.

    Dynamically rescales train/purge/test sizes if the full window
    doesn't fit, preserving relative proportions.
    """
    T = len(returns)
    # original sum
    S = train_size + purge_size + test_size
    if S > T:
        # scale down proportions
        scale = T / S
        tr = max(1, int(train_size * scale))
        pu = int(purge_size * scale)
        te = max(1, int(test_size * scale))
        # ensure sum <= T
        while tr + pu + te > T:
            # reduce purge first, then test
            if pu > 0:
                pu -= 1
            elif te > 1:
                te -= 1
            else:
                tr -= 1
        train_size_eff, purge_size_eff, test_size_eff = tr, pu, te
    else:
        train_size_eff, purge_size_eff, test_size_eff = train_size, purge_size, test_size

    losses = {k: [] for k in k_list}
    start = 0
    while True:
        train_end = start + train_size_eff
        test_start = train_end + purge_size_eff
        test_end = test_start + test_size_eff
        if test_end > T:
            break

        R_train = returns.iloc[start:train_end]
        F_train = factorRet.iloc[start:train_end]
        R_test  = returns.iloc[test_start:test_end]
        F_test  = factorRet.iloc[test_start:test_end]

        # skip empty
        if len(R_train)==0 or len(R_test)==0:
            break

        for k in k_list:
            pca = PCA(n_components=k).fit(F_train.values)
            Z_tr = pca.transform(F_train.values)
            Z_te = pca.transform(F_test.values)
            X_tr = np.hstack([np.ones((len(Z_tr),1)), Z_tr])
            B = np.linalg.solve(X_tr.T @ X_tr, X_tr.T @ R_train.values)
            X_te = np.hstack([np.ones((len(Z_te),1)), Z_te])
            R_pred = X_te @ B
            losses[k].append(np.mean((R_test.values - R_pred)**2))

        start += test_size_eff

    avg_loss = {k: (np.mean(v) if v else np.inf) for k,v in losses.items()}
    best_k = min(avg_loss, key=avg_loss.get)
    return best_k


def PCA_model(
    returns: pd.DataFrame,
    factorRet: pd.DataFrame,
    n_components: int = None,
    history_months: int = 60,
    train_size: int = 48,
    test_size: int = 6,
    purge_size: int = 1,
    k_list: list = None
):
    """
    Constructs a PCA-based factor model.

    Parameters:
    -----------
    returns     : (T x n) DataFrame of asset excess returns
    factorRet   : (T x p) DataFrame of factor returns (excess)
    n_components: int, number of principal components to retain

    Returns:
    --------
    mu  : ndarray (n x 1) expected excess returns
    Q   : ndarray (n x n) covariance matrix of asset returns
    R2  : ndarray (n x 1) in-sample R^2 per asset
    """
    # Auto-tune n_components if not provided
    if n_components is None:
        # Restrict tuning data to the first history_months of the calibration window
        hist_len = min(history_months, len(returns))
        R_hist = returns.iloc[:hist_len, :]
        F_hist = factorRet.iloc[:hist_len, :]
        p = factorRet.shape[1]
        if k_list is None:
            k_list = list(range(1, p+1))
        n_components = select_n_components_purged_cv(
            R_hist, F_hist, k_list,
            train_size, test_size, purge_size
        )
        print(f"Tuned # of PCs: {n_components}")

    # 1) Fit PCA on the raw factors
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(factorRet.values)  # (T, k)

    # 2) Build regression matrix [intercept, PCs]
    T, n = returns.shape
    X = np.hstack([np.ones((T,1)), Z])      # (T, k+1)

    # 3) OLS estimation of intercept & loadings
    B = np.linalg.solve(X.T @ X, X.T @ returns.values)
    alpha = B[0, :]                         # (n,)
    beta = B[1:, :]                         # (k, n)

    # 4) Residuals and R²
    eps = returns.values - X @ B
    R2 = np.zeros((n,1))
    for i in range(n):
        y = returns.values[:, i]
        yhat = X @ B[:, i]
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        R2[i,0] = 1 - ss_res/ss_tot if ss_tot>0 else np.nan

    # 5) Expected returns: alpha + beta.T @ mean(PC)
    z_bar = np.mean(Z, axis=0).reshape(-1,1)
    mu = alpha.reshape(-1,1) + beta.T @ z_bar

    # 5) Adjusted R²: 1 - (1-R2)*(T-1)/(T-k-1)
    df_total = T - 1
    df_resid = T - (n_components + 1)
    adjR2 = np.zeros((n,1))
    for i in range(n):
        adjR2[i,0] = 1 - (1 - R2[i,0]) * df_total / df_resid

    # 6) Covariance assembly
    # Compute factor‐covariance; for k=1 this might come back as a scalar
    Fz = np.cov(Z, rowvar=False)
    # Ensure Fz is at least 2D: if it's a scalar, turn it into a 1×1 matrix
    Fz = np.atleast_2d(Fz)
    Qf     = beta.T @ Fz @ beta
    var_eps= np.var(eps, axis=0)
    Q      = Qf + np.diag(var_eps)
    Q      = (Q + Q.T) / 2
    return mu, Q, R2, adjR2

if __name__ == "__main__":
    # Load prices and factor returns
    prices = pd.read_csv("MMF1921_AssetPrices_3.csv", index_col=0, parse_dates=True)
    factorRet = pd.read_csv("MMF1921_FactorReturns_3.csv", index_col=0, parse_dates=True)

    # Compute monthly returns from prices
    assetRet = prices.pct_change().dropna()

    # Compute excess returns: subtract RF (risk-free rate)
    RF = factorRet['RF']  # Usually RF is in percent
    excessRet = assetRet.sub(RF, axis=0)  # auto-aligns on index

    # Drop RF column from factors before using them in OLS
    factors = factorRet.drop(columns=['RF'])

    # Align time indices
    common_idx = excessRet.index.intersection(factors.index)
    excessRet = excessRet.loc[common_idx]
    factors = factors.loc[common_idx]

    # Hyper parameter tuning for PCA
    history_months = 60
    train_size     = 48
    test_size      = 6
    purge_size     = 3
    p               = factors.shape[1]
    k_list         = list(range(1, p+1))

    # tune on the first 60 months of (aligned) excess returns & raw factors
    best_k = select_n_components_purged_cv(
        excessRet.iloc[:history_months, :],
        factors.iloc[:history_months, :],
        k_list,
        train_size,
        test_size,
        purge_size
    )
    print(f"Tuned # of PCs: {best_k}")


    # Run OLS, FF3
    mu_ols, Q_ols, R2_ols, R2adj_ols = OLS(excessRet, factorRet)
    mu_ff3, Q_ff3, R2_ff3, R2adj_ff3 = FF3(excessRet, factorRet)
    mu_lasso, Q_lasso, R2_lasso, R2adj_lasso = LASSO(excessRet, factorRet, lam=0.01)
    mu_bss, Q_bss, R2_bss, R2adj_bss = BSS(excessRet, factorRet, K=3)

    ''' 
    # Run lambda 
    # Step 1: Choose lambda grid
    lam_grid = np.logspace(-3, 0, 10)
    y = excessRet.iloc[:, 0]  # This is a Series of shape (T,)
    X = np.column_stack([np.ones(len(factors)), ((factors - factors.mean()) / factors.std()).values])
    # Step 2: Run cross-validation once across all assets
    best_lam = cross_validate_lasso(X, y, lam_grid)
    '''

    # LASSO result
    print("\n--- LASSO Results ---")
    #print("Best lambda:", best_lam)
    print("mu shape:", mu_lasso.shape)
    print("Average R²:", np.mean(R2_lasso))

    # Print output summaries
    print("\n--- OLS Results ---")
    print("mu shape:", mu_ols.shape)
    print("Q shape:", Q_ols.shape)
    print("R2 shape:", R2_ols.shape)
    print("Average R²:", np.mean(R2_ols))
    
    print("\n--- Fama-French 3-Factor Results ---")
    print("mu shape:", mu_ff3.shape)
    print("Q shape:", Q_ff3.shape)
    print("R2 shape:", R2_ff3.shape)
    print("Average R²:", np.mean(R2_ff3))

    K = 3  # or try 2, 4, etc.
    print("mu shape:", mu_bss.shape)          # should be (33, 1)
    print("Q shape:", Q_bss.shape)            # should be (33, 33)
    print("R² shape:", R2_bss.shape)          # should be (33, 1)
    print("Average R²:", np.mean(R2_bss))     # summary of fit

