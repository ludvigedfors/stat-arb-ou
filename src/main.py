import LoadData
import SignalGeneration
import BackTest
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # main()
    # test()
    # pairs: KO/PEP, GOOG/GOOGL
    pair = LoadData.load_data(A='KO', B='PEP')
    cointegration_result = LoadData.cointegration_test(pair)
    # print(f"Intercept = {cointegration_result['Intercept']:.6f}")
    # print(f"Hedge Ratio  = {cointegration_result['HedgeRatio']:.6f}")
    #print(cointegration_result["Result"].head())


    # Generate Signal
    signals = SignalGeneration.generate_signal(
        spread=cointegration_result["Spread"],
        half_life=cointegration_result["HalfLife"],
        entry_z=2.0, exit_z=0.0, stop_z=3.5)

    # Backtest & Review Metrics
    results = BackTest.backtest(
        signals = signals,
        pair_log_prices = np.log(pair),
        tickA = cointegration_result["Ticker A"],
        tickB = cointegration_result["Ticker B"],
        beta=cointegration_result["HedgeRatio"],
        transaction_cost=0.0,
    )

    print("\n--- Strategy Performance Metrics ---")
    for k, v in results["metrics"].items():
        print(f"{k}: {v:.2f}")

    # Plot
    BackTest.plot_backtest_results(results["backtest_df"])
    
