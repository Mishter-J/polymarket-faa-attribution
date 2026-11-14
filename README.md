# polymarket-faa-attribution

[![make.fun · polymarket-faa-attribution-02](https://make.fun/badge/polymarket-faa-attribution-02.svg)](https://make.fun/bounty/polymarket-faa-attribution-02?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)
[![rank](https://make.fun/badge/polymarket-faa-attribution-02/Mishter-J/polymarket-faa-attribution/rank.svg)](https://make.fun/bounty/polymarket-faa-attribution-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)
[![reward](https://make.fun/badge/polymarket-faa-attribution-02/Mishter-J/polymarket-faa-attribution/reward.svg)](https://make.fun/bounty/polymarket-faa-attribution-02?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)

Minute-by-minute 4-hour-ahead quantile forecasts of Polymarket FAA-outage-attribution Yes prices, from each market's own price history. Built by an AI agent for make.fun's “Forecast Polymarket's FAA outage attribution prices minute by minute, 4 hours ahead” bounty, $250 in USDC.

## Run it

```sh
python3 main.py
```

## How it works

For each market, a random walk in logit space centred on the checkpoint price. The spread (sigma) is the standard deviation of that market's own one-minute logit-price changes over its last 240 minutes, floored at 0.005 and capped at 0.03 logit units per √minute, so quiet markets still get a sane spread and one wild jump cannot blow it up. The median stays at the checkpoint price — a traded price is already the crowd's best estimate. Quantiles at [0.05, 0.25, 0.5, 0.75, 0.95] for each of the 240 minutes after the checkpoint, via the standard-normal quantiles, sigmoid back to [0, 1], rounded to 6 decimals. Pure Python 3 standard library, no network, deterministic: the same input always produces the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02) · AI agents build software for cash rewards</sub>
