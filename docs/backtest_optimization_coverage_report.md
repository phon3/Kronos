# Backtest Optimization Coverage Report

## Executive Answer

The optimizer covered every major backtest category, but it did **not** exhaustively cross every Web UI setting, model, dataset, and timeframe.

The trustworthy final experiments used:

- Market: BTC/USD
- Entry timeframe: 1h
- Data: `data/BTC_USD_1h.csv`
- Kronos model: `btc_usd_1h_prod`
- Macro data for multi-timeframe mode: `data/BTC_USD_1d.csv`
- Source-data range: 2025-01-04 16:00 through 2025-07-31 23:00
- Common OOS start: 2025-05-30 12:00
- Initial capital: $100,000
- Fees: 0.1% per transaction
- Slippage: 0.05% adverse per execution
- Validation: common timestamp split plus three chronological OOS folds
- Candidate gates: at least 10 reported OOS exits and non-negative excess return over buy-and-hold

No trustworthy final optimizer run has compared `btc_usd_15m_prod`, `btc_usd_1h_prod`, and `btc_usd_1d_prod`. There is not yet an evidence-based best timeframe or best production model.

## Combined Confirmation Correction

The initial Combined implementation confirmed Kronos only on raw nonzero trend-event bars. It ignored the held trend `position` that carries the larger regime through neutral bars. This contradicted the intended rule that confirmation remains active until trend reversal.

After correcting Combined to use the held trend regime, the OOS diagnostics were:

- OOS bars: 1,460
- Kronos held long: 745 bars
- Kronos held short: 715 bars
- Raw trend long: 369 bars
- Raw trend short: 132 bars
- Raw trend neutral: 959 bars
- Held trend long: 1,038 bars
- Held trend short: 422 bars
- Combined long agreement: 503 bars
- Combined short agreement: 180 bars
- Flat/no agreement: 777 bars
- Combined entries: 70
- Complete exits: 70

The corrected trade count validates the expectation that Kronos signals should trigger repeatedly while the larger trend remains active. It also supersedes the earlier 29-exit Combined candidate.

Corrected OOS returns:

| Risk profile | OOS return | Buy-and-hold | Exits | Qualified |
|---|---:|---:|---:|---|
| Conservative | 0.38% | 11.75% | 70 | No |
| Balanced | 0.93% | 11.75% | 70 | No |
| Aggressive | 4.36% | 11.75% | 70 | No |
| DeepTrade | 0.89% | 11.75% | 71 reported exit events | No |

Thus the corrected Combined strategy has adequate activity but does not produce OOS alpha.

## Web UI Setting Coverage

| Web UI setting | Executed coverage | Status | Notes |
|---|---|---|---|
| Strategy preset / signal mode | Trend, Kronos, Combined, Multi-Timeframe | Covered | All four modes were evaluated independently. |
| Model | `btc_usd_1h_prod` | Partial | No final 15m or 1d model comparison. |
| Device | CUDA on RTX 4090 | Fixed | Device changes runtime, not strategy semantics. |
| Samples | 1 and 3 | Covered selectively | Three-sample forecasts did not produce a benchmark-beating candidate. |
| Signal threshold | 0.0005, 0.00075, 0.001, 0.00125, 0.0015; broader exploratory values through 0.005 | Covered targeted range | Final promising region was around 0.0015. |
| Lookback | 256 in final trustworthy run | Partial | 256/512 were explored earlier, but that matrix was superseded by sampling and evaluation-alignment fixes. |
| Prediction length | 48 in final trustworthy run | Partial | 24/48 were explored earlier, but that matrix was superseded. |
| Data file | `BTC_USD_1h.csv` | Partial | No matched 15m or 1d model/data optimization yet. |
| Max data bars | 5,000 in final runs; 1,024 and 2,048 diagnostics | Covered selectively | Final OOS period was approximately two months. |
| Macro data file | `BTC_USD_1d.csv` | Covered for one configuration | Used only for Multi-Timeframe mode. |
| Initial capital | $100,000 | Fixed | Return percentages should be mostly invariant absent nonlinear constraints. |
| Position size | 10%, 25%, 40%, 50% | Covered | Targeted risk run focused on 25%, 40%, and 50%. |
| Stop loss | 4%, 8%, 12% | Covered | Targeted risk run focused on 8% and 12%. |
| Single take profit | None, 8%, 12%, 20% | Covered | Targeted risk run used None, 12%, and 20%. |
| Loss multiplier | 1.0x, 1.5x | Covered | 1.5x materially increases risk after losses. |
| Multi-target TP | Three schedules | Covered selectively | See schedules below. |
| Daily loss limit | 2%, 3%, 5% | Covered | Targeted run focused on 3% and 5%. |
| Trend period | 30, 60 | Covered selectively | Period 60 dominated the promising Combined results. |
| Trend minimum bars | 3, 5 | Covered selectively | Both frequently produced behaviorally identical signals. |
| Trend break extension | 0.03, 0.05 | Covered selectively | 0.03 dominated the promising results. |
| Trend limit extension | 0.01, 0.02 | Covered selectively | Both often produced identical behavior. |
| Walk-forward validation | 70/30 timestamp split plus 3 OOS folds | Covered with fixed structure | Other train ratios and fold counts were not swept. |
| Macro period | 10 | Fixed | Broader macro periods were not run in the final Multi-Timeframe comparison. |
| Macro break extension | 0.03 | Fixed | Not broadly swept. |
| Macro limit extension | 0.03 | Fixed | Not broadly swept. |
| Macro minimum bars | 3 | Fixed | Not broadly swept. |

## Multi-Target TP Schedules Tested

1. 50% at +3%, 50% at +6%
2. 50% at +4%, 50% at +8%
3. 33% at +5%, 33% at +10%, 34% at +15%

The earlier targeted multi-target result of 11.98% and no-fixed-TP result of 12.07% were generated by raw-event-only Combined confirmation. They are superseded by the held-regime correction and must not be used as live-test candidates.

Under corrected held-regime confirmation, the bundled DeepTrade profile returned 0.89% OOS versus 11.75% buy-and-hold and did not qualify.

## Combined Mode Coverage

Combined was tested in two final stages:

### Inference and signal stage

- Lookback: 256
- Prediction length: 48
- Samples: 1 and 3
- Thresholds: 0.0005 through 0.0015
- Trend period: 60
- Break extension: 0.03
- Limit extension: 0.01 and 0.02
- Minimum bars: 3 and 5
- Four bundled risk profiles

Historical outcome before the held-regime correction:

- Sample count 1 produced one marginal behavior.
- Sample count 3 produced no benchmark-beating candidate.
- The marginal result is superseded and is no longer a valid candidate.

### Targeted risk stage

- 144 combinations
- Position size: 25%, 40%, 50%
- Stop loss: 8%, 12%
- Take profit: None, 12%, 20%
- Loss multiplier: 1.0x, 1.5x
- Daily loss limit: 3%, 5%
- Three multi-target schedules

Historical outcome before the held-regime correction:

- Two marginal behaviors qualified under raw-event-only confirmation.
- Both required 50% sizing and 1.5x loss escalation.
- Both are superseded by the corrected 70-entry Combined result, which underperformed buy-and-hold in every tested bundled risk profile.

## Why So Few Trades?

### OOS duration is limited

The final source set contained 5,000 hourly bars. The OOS period began on 2025-05-30 and ended near 2025-07-30/31, approximately two months. A two-month test cannot produce a large number of independent medium-term trend trades without using a very sensitive signal.

### Combined mode requires two simultaneous conditions

A Combined entry requires:

1. Kronos predicted movement to exceed the signal threshold; and
2. The held trend regime to agree with the predicted direction.

The earlier implementation incorrectly required a raw nonzero trend event on the same bar, which produced only 29 exits. Held-regime confirmation now produces 70 complete entries/exits for the diagnostic configuration. Kronos-only still changes direction much more frequently, but those high-frequency configurations lose money after costs.

### Trend settings create persistent movements

Trend periods of 30 or 60 bars and minimum movement lengths of 3 or 5 bars intentionally suppress short-lived changes. Trend-only configurations reported 1–10 exits. The apparent 20.72% Trend result had only five exits and was rejected.

### Multi-timeframe adds another filter

Multi-Timeframe mode requires entry-timeframe and 1d macro trends to agree. It reported only 1–7 exits. Its best return had three exits and only one profitable OOS fold, so it was rejected.

### Multi-target exits affect the count

`total_trades` currently counts both complete closes and `PARTIAL_CLOSE` events. A three-stage take-profit can therefore contribute multiple reported trades from one original position. The 30 reported exits in the best multi-target configuration are not necessarily 30 independent entries. This means its statistical sample size is weaker than the headline count suggests.

### Strict quality gates remove attractive low-sample results

Candidates must:

- Have at least 10 reported OOS exits;
- Beat buy-and-hold OOS;
- Survive confidence adjustment toward a target of 30 trades;
- Be evaluated across three chronological folds;
- Absorb adverse fees and slippage.

This correctly rejected:

- Trend: 20.72% with five exits;
- Multi-Timeframe: 10.94% with three exits;
- Historical raw-event Combined variants returning 14–20% with fewer than ten exits;
- Corrected held-regime Combined variants with 70 exits but substantial benchmark underperformance;
- Kronos-only variants with many trades but negative performance.

## Mode Summary

| Mode | Data | Model / macro | Best observed result | Trade evidence | Qualified behavior |
|---|---|---|---:|---|---|
| Combined | BTC/USD 1h | `btc_usd_1h_prod` | 4.36% vs 11.75% hold after held-regime correction | 70 complete exits | No |
| Kronos only | BTC/USD 1h | `btc_usd_1h_prod` | Approximately -1.04% best | Up to 398 exits | No |
| Trend only | BTC/USD 1h | None | 20.72% vs 9.51% hold | Only 5 exits; >=10-exit variants underperformed | No |
| Multi-Timeframe | BTC/USD 1h | BTC/USD 1d macro | 10.94% vs 9.51% hold | Only 3 exits, 1 profitable fold | No |

## What Was Not Covered

The following still require separate experiments:

1. `btc_usd_15m_prod` against matching 15m data.
2. `btc_usd_1d_prod` against matching 1d data.
3. Final post-fix lookback comparison beyond 256.
4. Final post-fix prediction-length comparison beyond 48.
5. Temperature and top-p grids.
6. Long-only versus short-only behavior.
7. Fee and slippage stress grids.
8. Broader macro period, extension, and minimum-bar grids.
9. Alternative train ratios and more rolling market-regime windows.
10. Full-history runs beyond the most recent 5,000 hourly bars.

## Current Conclusion

The request to cover every major category was met, but the executed search was targeted rather than exhaustive. It is incorrect to claim that every Web UI value, model, dataset, and timeframe was tested.

Within the tested 1h model/data set, no mode currently has a qualified behavior after correcting Combined to use held trend regimes. Combined now has adequate trade activity but materially underperforms buy-and-hold. The next evidence-producing work is matched 15m and 1d model/data evaluation, longer regime coverage, and reconsideration of how Kronos forecast paths are converted into directional signals—not relaxing ranking gates or selecting low-trade high-return configurations.
