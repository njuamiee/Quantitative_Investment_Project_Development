# 2026-10-11 research protocol

This protocol is recorded before the new runs. Earlier results remain frozen.

## Objective and evidence standard

Investigate stock-selection systems with annualized Sharpe > 1 and annualized
Calmar > 2, at 5 bp one-way assumed total transaction cost. A threshold is an
acceptance target, never a reason to edit returns, omit losing periods, or tune
on the outer evaluation. Report every registered candidate and failure.

The 2024Q4–2026Q3 history has already been inspected in earlier work. Walk-forward
results on it are retrospective out-of-time estimates, not a pristine holdout.
October data will be recorded separately; immature labels are not scored.

## Scope

1. Read-only SFTP inventory and file-vintage comparison with October 8.
2. Download required changed data into a new snapshot, with hashes and failures.
3. Forecast 1, 2, 5, 10 and 20 trading-day horizons. For h>=2, signal at close t,
   entry open t+1, target exit close t+h. For h=1, next-open to following-open
   prediction/holding avoids an impossible same-day sale under T+1.
4. Training-only causal factor screening; linear, tree, ranking, ensemble models.
5. HMM filtered probabilities: volatility, trend, liquidity and dispersion;
   2/3-state alternatives, risk budgets and conditional factor weighting.
6. Compare HMM against unscaled, fixed exposure, volatility and trend controls.
7. Risk management: causal volatility sizing, diversification, drawdown responses;
   no future known drawdown or full-sample smoothed HMM probabilities.
8. Intraday factor research and a restricted-expression LLM proposed factor library.
9. Research Qlib, RD-Agent, financial foundation models and Jev with primary sources.
10. Produce code, reproducible runs, missing-file inventory, teacher task mapping,
    and a complete explanatory document with explicit completion status.

## Missing evidence

Do not fabricate historical universe membership, delisting, corporate-action cash
settlement, share tradability, field release times, proprietary platform access or
external model credentials. Document conditional experiments separately from fully
verified real-trading claims. Do not place trades or modify the remote server.

## Metrics

Use daily equity including cash. Annualized return uses actual elapsed calendar
time; Sharpe uses 252 sessions and both 0% and 1.5% annual risk-free sensitivities.
Calmar = annualized return / absolute maximum drawdown. Include the initial capital
in the equity peak. Report exposure, turnover, cash days, completed-trade win rate,
payoff ratio, benchmark excess and return confidence intervals. The user requires
both Sharpe > 1 and Calmar > 2; a result failing either is not a passing model.

## Subsequent exploratory amendments (recorded separately)

The initial validation-selection procedure did not meet both thresholds. The
subsequent risk refinement (540 accounts) and forecast/holding extension (216
accounts) used already inspected history to develop candidates. Their winners
are explicitly retrospective discoveries, not successful tests under the initial
selection protocol. Both the initial failures and later grids remain available.

On October 12, a subperiod audit identified recent-year losses behind full-history
passing ratios. A further registered follow-up tests 63/126/252-session supervised
training windows with monthly updates, and 126 versus 504-session HMM windows.
It tests 480 accounts for forecast horizons 1/5/20, requiring full-history, recent
year and recent half-year thresholds jointly for a historical stability screen.
This screen itself uses inspected periods and cannot replace future validation.
The supplementary signal diagnosis uses fixed existing candidates, without
selecting new parameters from the diagnostic results.
