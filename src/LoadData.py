import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen



def load_data(stocks, start_date='2025-10-01', end_date='2026-10-01'):
    data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=True)
    close_price = data['Close']
    return close_price


def engle_granger_test(data, significance=0.05):
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


def johansen_test(data, det_order = 0, k_ar_diff = 1, significance_level = "5%"):
    sig_index = {"1%": 2, "5%": 1, "10%": 0}[significance_level]

    result = coint_johansen(data, det_order, k_ar_diff)

    trace_stats = result.lr1
    trace_cvs = result.cvt[:, sig_index]

    max_eig_stats = result.lr2
    max_eig_cvs = result.cvm[:, sig_index]

    # Sequential rank determination
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

def ornstein_uhlenbeck_half_life(spread):
    # calculate Ornstein-Uhlenbeck half-life on the resulting spread
    # dS_t = theta * (mu - S_t) * dt + sigma * dW_t
    # {without noise} -> d(S_t-mu)/dt = -theta * (S_t-mu) 
    # -> S_t-mu = (S_0-mu) * exp(-theta * t)
    # t_half = ln(2)/theta
    # dS ~ DeltaS -> DeltaS = theta*(mu-S_t-1) * 1 + e
    # OLS beta = -theta 

        s_lag = spread[:-1]
        s_diff = np.diff(spread)
        reg = np.linalg.lstsq(np.column_stack([np.ones(len(s_lag)), s_lag]), s_diff, rcond=None)[0]
        half_life = -np.log(2) / reg[1] if reg[1] < 0 else np.inf
        return half_life


def cointegration_test(data, test_type=2, print_results=True):
    if data is None:
        raise ValueError("Data not loaded. Please call load_data() first.")

    log_data = np.log(data)

    best_pair = None
    best_score = -np.inf
    best_stats = None

    for i, ticker1 in enumerate(log_data.columns[:-1]):
        for ticker2 in log_data.columns[i+1:]:
            pair = log_data[[ticker1, ticker2]].dropna()

            if test_type == 1:
                stats = engle_granger_test(pair)
                score = -stats["min_p_value"]
            elif test_type == 2:
                joh = johansen_test(pair, det_order=1, k_ar_diff=1, significance_level="5%")

                # demand rank = 1 for trace and eig
                trace_rank_1 = (joh["trace_statistics"][0] > joh["trace_critical_values"][0]) and \
                            (joh["trace_statistics"][1] < joh["trace_critical_values"][1])
                
                eigen_rank_1 = (joh["max_eigen_statistics"][0] > joh["max_eigen_critical_values"][0]) and \
                            (joh["max_eigen_statistics"][1] < joh["max_eigen_critical_values"][1])

                evec = joh["cointegrating_vectors"][:, 0]
                beta_candidate = -evec[0] / evec[1] if evec[1] != 0 else 0

                if trace_rank_1 and eigen_rank_1 and beta_candidate > 0:
                    # normalised score 
                    score = (joh["trace_statistics"][0] / joh["trace_critical_values"][0])
                else:
                    score = -np.inf
                stats = joh
            else:
                raise ValueError("test_type must be 1 (Engle-Granger) or 2 (Johansen)")

            if score > best_score:
                best_score = score
                best_pair = (ticker1, ticker2, pair)
                best_stats = stats

    if best_pair is None or best_score == -np.inf:
        raise ValueError("No valid cointegrated pair found.")

    ticker1, ticker2, pair_data = best_pair

    log_pair = pair_data

    A = log_pair[ticker1].to_numpy()
    B = log_pair[ticker2].to_numpy()

    # use Johansen's cointegrating vector when test_type==2 to maintain directional symmetry
    if test_type == 1:
        # OLS regression: B = alpha + beta * A
        X = np.column_stack([np.ones(len(A)), A])
        alpha, beta = np.linalg.lstsq(X, B, rcond=None)[0]
        epsilon = B - (alpha + beta * A)
    elif test_type == 2:
        # Johansen eigenvector: w1 * A + w2 * B = stationary spread
        evec = best_stats["cointegrating_vectors"][:, 0]
        beta = -evec[0] / evec[1]
        raw_spread = B - beta * A
        alpha = float(np.mean(raw_spread))
        epsilon = raw_spread - alpha
    else:
        raise ValueError("test_type must be 1 (Engle-Granger) or 2 (Johansen)")
    
    half_life = ornstein_uhlenbeck_half_life(epsilon)

    result_df = pd.DataFrame(
        {
            "log_A": A,
            "log_B": B,
            "fitted_B": alpha + beta * A,
            "epsilon": epsilon,
        },
        index=log_pair.index,
    )
    spread_series = pd.Series(epsilon, index=log_pair.index, name="spread")

    # print the precomputed results of the best pair
    if print_results:
        print("=" * 50)
        print(f"Selected Pair: {ticker1} and {ticker2}")
        print("=" * 50)
        if test_type == 1:
            print(f"Engle-Granger Cointegration Test Results")
            print(f"Cointegrated ({ticker1} on {ticker2}): {best_stats['cointegrated'][0]}, p-value: {best_stats['p_value'][0]:.6f}")
            print(f"Cointegrated ({ticker2} on {ticker1}): {best_stats['cointegrated'][1]}, p-value: {best_stats['p_value'][1]:.6f}")
        elif test_type == 2:
            print(f"Johansen Cointegration Test Results")
            for i in range(2):
                t_stat = best_stats['trace_statistics'][i]
                c_val = best_stats['trace_critical_values'][i]
                print(f"# of cointegrating relations <= {i}: trace stat = {t_stat:.2f}, critical value = {c_val:.2f}, reject = {t_stat > c_val}")
            print(f"Cointegrating Rank: {best_stats['n_cointegrating_trace']}")
        print(f"Half-life of Mean Reversion: {half_life:.2f} days")

    return {
        "Ticker A": ticker1,
        "Ticker B": ticker2,
        "Intercept": alpha,
        "HedgeRatio": beta,
        "Spread": spread_series,
        "HalfLife": half_life,
        "Result": result_df,
        "selection_score": best_score,
    }