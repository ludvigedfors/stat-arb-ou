import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def load_data(A = 'GOOG', B = 'GOOGL', start_date='2020-01-01', end_date='2023-01-01'):
    # Download historical data for the two stocks
    data = yf.download([A, B], start=start_date, end=end_date)
    # print(data.head())  # Display the first few rows of data
    close_price = data['Close']
    # print(close_price.head())  # Display the first few rows of close_price
    return close_price  

def engle_granger_test(data, significance=0.05): 
    '''Good if i know B is dependent on A
    not if i dont know which is dependent on which
    otherwise run both ways'''
    if data is None:
        raise ValueError("Data not loaded. Please call load_data() first.")

    t1, t2 = data.columns[0], data.columns[1]

    # Run both directions
    _, p1, _ = coint(data[t1], data[t2])
    _, p2, _ = coint(data[t2], data[t1])

    coint1 = p1 < significance
    coint2 = p2 < significance

    return {
        "cointegrated": (coint1, coint2),
        "p_value": (p1, p2),
        "min_p_value": min(p1, p2),
    }


def johansen_test(data: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1, significance_level: str = "5%"):
    """more robust test for cointegration, 
    can handle multiple time series and 
    does not require specifying which series is dependent on the other.
    Needs large sample though
    det_order = 0, assume no trend in the data
    k_ar_diff = 1 for daily closing prices, 0 for intra day returns are basically noise
    """
    sig_index = {"1%": 2, "5%": 1, "10%": 0}[significance_level]

    result = coint_johansen(data, det_order, k_ar_diff)

    trace_stats = result.lr1  # trace statistics
    trace_cvs = result.cvt[:, sig_index]  # critical values for trace

    max_eig_stats = result.lr2  # max eigenvalue statistics
    max_eig_cvs = result.cvm[:, sig_index]

    # # Count cointegrating relations using the trace test
    # n_coint_trace = sum(t > c for t, c in zip(trace_stats, trace_cvs))
    # n_coint_eigen = sum(t > c for t, c in zip(max_eig_stats, max_eig_cvs))

    # Proper sequential rank determination
    n_coint_trace = 0
    for stat, cv in zip(trace_stats, trace_cvs):
        if stat > cv:
            n_coint_trace += 1
        else:
            break

    n_coint_eigen = 0
    for stat, cv in zip(max_eig_stats, max_eig_cvs):
        if stat > cv:
            n_coint_eigen += 1
        else:
            break

    return {
        "trace_statistics": trace_stats,
        "trace_critical_values": trace_cvs,
        "max_eigen_statistics": max_eig_stats,
        "max_eigen_critical_values": max_eig_cvs,
        "cointegrating_vectors": result.evec,
        "n_cointegrating_trace": n_coint_trace,
        "n_cointegrating_eigen": n_coint_eigen,
    }
    
    
def cointegration_test(data, test_type = 1, print_results=True):
    if data is None:
        raise ValueError("Data not loaded. Please call load_data() first.")

    # choose the pair with the strongest cointegration signal
    log_data = np.log(data)

    best_pair = None
    best_score = -np.inf
    best_stats = None

    for i, ticker1 in enumerate(log_data.columns[:-1]):
        for ticker2 in log_data.columns[i+1:]:
            pair = log_data[[ticker1, ticker2]].dropna()

            if test_type == 1:
                # Engle-Granger: smaller p-value = stronger evidence
                stats = engle_granger_test(pair)
                score = -min(stats["p_value"])
            elif test_type == 2:
                # Johansen: more cointegrating relations and bigger gap to critical value = stronger evidence
                joh = johansen_test(pair)
                score = joh["n_cointegrating_trace"] + np.mean(
                    np.maximum(joh["trace_statistics"] - joh["trace_critical_values"], 0)
                )
            else:
                raise ValueError("test_type must be 1 (Engle-Granger) or 2 (Johansen)")

            if score > best_score:
                best_score = score
                best_pair = (ticker1, ticker2, pair)

    if best_pair is None:
        raise ValueError("No valid pair found for cointegration analysis.")

    ticker1, ticker2, pair_data = best_pair

    log_pair = np.log(pair_data).dropna()

    # OLS: ticker2_t = alpha + beta * ticker1_t + epsilon_t
    A = log_pair[ticker1].to_numpy()
    B = log_pair[ticker2].to_numpy()
    X = np.column_stack([np.ones(len(A)), A])

    alpha, beta = np.linalg.lstsq(X, B, rcond=None)[0]
    epsilon = B - (alpha + beta * A)

    result = pd.DataFrame(
        {
            "log_A": A,
            "log_B": B,
            "fitted_B": alpha + beta * A,
            "epsilon": epsilon,
        },
        index=log_pair.index,
    )

    if print_results:
            if test_type == 1:
                result = engle_granger_test(data)
                print("Engle-Granger Cointegration Test Results")
                print("=" * 50)
                print(f"Cointegrated (A on B): {result['cointegrated'][0]}, p-value: {result['p_value'][0]:.6f}")
                print(f"Cointegrated (B on A): {result['cointegrated'][1]}, p-value: {result['p_value'][1]:.6f}")
    
            elif test_type == 2:
                result = johansen_test(data)
                print("Johansen Cointegration Test Results")
                print("=" * 50)
                for i in range(len(data.columns)):
                    print(
                        f"# of cointegrating relations <= {i}: trace stat = {result['trace_statistics'][i]:.2f}, "
                        f"critical value = {result['trace_critical_values'][i]:.2f}, "
                        f"reject = {result['trace_statistics'][i] > result['trace_critical_values'][i]}")
                    
                print(f"\nNumber of cointegrating relations (trace): "f"{result['n_cointegrating_trace']}")
    
                print(f"Number of cointegrating relations (max eigenvalue): "f"{result['n_cointegrating_eigen']}")
    

    return {
        "Ticker A": ticker1,
        "Ticker B": ticker2,
        "Intercept": alpha,
        "HedgeRatio": beta,
        "Spread": epsilon,
        "Result": result,
        "selection_score": best_score}
    
