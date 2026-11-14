#!/usr/bin/env python3
"""Minute-by-minute 4-hour-ahead quantile forecasts of Polymarket Yes prices.

Reads every case file ``<id>.in`` from $MAKEFUN_INPUT_DIR (else /input/) and
writes one forecast ``<id>.out`` per case to $MAKEFUN_OUTPUT_DIR (else
/output/).  Each output is {"forecasts": [{"id": <market id>, "q": [[5
quantiles] x 240 rows]}]} at the levels [0.05, 0.25, 0.5, 0.75, 0.95].

Model
-----
Per market, a random walk in logit space, centred on the checkpoint price:

* Let x be the price at the checkpoint and lx = logit(x) (prices are clamped
  into [0.001, 0.999] before the transform so 0 and 1 stay finite).
* sigma is the population standard deviation of the market's own one-minute
  logit-price changes over its last 240 minutes (fewer than 10 changes, e.g.
  a market that only just started trading, falls back to a default of 0.04).
  Before taking the standard deviation, the changes are winsorized at the
  99th percentile of their absolute values in the window: prices move in
  steps with occasional jumps, and a single jump (the Iran market once fell
  from 0.49 to 0.275 in one minute) would otherwise set the spread for the
  next four hours even though the market is otherwise quiet.  Winsorizing
  keeps the estimate tied to the market's typical minute-to-minute noise
  while still letting a market that jumps often keep a wide spread, since
  its 99th percentile is itself large.  Sigma is then floored at 0.005 and
  capped at 0.03 logit units per sqrt-minute, so a market that has sat still
  for hours still gets a sane spread.
* For minute h = 1..240 after the checkpoint, the tau-quantile is
  sigmoid(lx + z_tau * sigma * sqrt(h)), clamped to [0, 1] and rounded to
  6 decimals, where z_tau is the standard-normal quantile at tau.

Why logit space: prediction-market prices live in [0, 1] and move in steps
with occasional jumps.  A walk in logit space keeps every quantile inside
(0, 1) by construction, widens the spread in price terms when the price is
near 0.5 and narrows it near the boundaries, and lets each market's own
recent volatility — not one global constant — set the spread.  The median
stays at the checkpoint price: a traded price is already the crowd's best
estimate, so no drift is imposed.

Determinism: no randomness, no wall-clock, no network; the same input
always produces the same bytes.  Pure standard library.
"""

import json
import math
import os
import statistics

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at the five levels above.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

WINDOW = 240          # minutes of history used to estimate sigma
MIN_DIFFS = 10        # need at least this many 1-min changes, else default
DEFAULT_SIGMA = 0.04  # logit units per sqrt-minute, for very short histories
SIGMA_FLOOR = 0.005   # logit units per sqrt-minute
SIGMA_CAP = 0.03      # logit units per sqrt-minute
LOGIT_EPS = 0.001     # clamp prices into [eps, 1 - eps] before logit


def logit(p):
    p = min(1.0 - LOGIT_EPS, max(LOGIT_EPS, p))
    return math.log(p / (1.0 - p))


def sigmoid(u):
    if u >= 0:
        return 1.0 / (1.0 + math.exp(-u))
    e = math.exp(u)
    return e / (1.0 + e)


def estimate_sigma(prices):
    """Per-market 1-minute volatility in logit space, from recent history.

    The 1-minute logit changes over the last 240 minutes are winsorized at
    the 99th percentile of their absolute values before the standard
    deviation is taken, so one jump does not set the spread for the next
    four hours (see the module docstring).
    """
    n = len(prices)
    if n < 2:
        return DEFAULT_SIGMA
    start = max(0, n - WINDOW - 1)
    diffs = [logit(prices[i + 1]) - logit(prices[i])
             for i in range(start, n - 1)]
    if len(diffs) < MIN_DIFFS:
        return DEFAULT_SIGMA
    abs_diffs = sorted(abs(d) for d in diffs)
    cap = abs_diffs[int(0.99 * (len(abs_diffs) - 1))]
    diffs = [max(-cap, min(cap, d)) for d in diffs]
    sigma = statistics.pstdev(diffs)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast_market(prices, horizon):
    """240 rows x 5 quantiles for one market, median at the checkpoint price."""
    x = prices[-1] if prices else 0.5
    lx = logit(x)
    sigma = estimate_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        row = []
        for z in Z:
            q = sigmoid(lx + z * spread)
            q = min(1.0, max(0.0, q))
            row.append(round(q, 6))
        rows.append(row)
    return rows


def forecast_case(case):
    horizon = int(case.get("horizon_minutes", 240))
    forecasts = []
    for market in case["markets"]:
        prices = market["prices"]
        forecasts.append({"id": market["id"],
                          "q": forecast_market(prices, horizon)})
    return {"forecasts": forecasts}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":"), sort_keys=True)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
