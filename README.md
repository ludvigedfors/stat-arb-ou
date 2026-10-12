# Stat arb. 

Python project, statistical arbitrage using Ornstein-Uhlenbeck process

Pick two stock believed to be cointegrated, 

examples:

Estimation of cointegration by OLS
    A,B log price
    B_t = alpha + beta*A_t + epsiln_t ; beta - hedge ratio, alpha - intercept, epsilon_t reps the spread.

    Is run on price levels NOT returns, as we need stationary residuals for pair-arbitrage trading to work.

Test residuals for stationarity (ADF or johansen?)

    dEpsilon_t = GammaEpsilon_t-1 + sum_i=1^p (delta_i * dEpsilon_t-i) + u_t


