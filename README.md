# GTA VI trophy rarity forecaster

Forecasts each GTA VI trophy's Exophase earn rate at the case target as five
quantiles, scored by pinball loss on the logit scale.

## Model

Earners and players are forecast **separately in log-time**
(x = ln(1 + hours since launch)), and the rate is their ratio:

- **Players P(t):** concave quadratic of log P on x. The tracked base grows
  fast at launch and saturates; a positive quadratic term is dropped so a
  4.5-day extrapolation cannot explode.
- **Earners E_i(t):** per trophy, log(E_i + 0.5) is linear in x once the
  players curve's curvature is removed, so earners inherit the base's
  saturation instead of extrapolating launch night's explosive slope in a
  straight line. Each trophy keeps its own slope, shrunk toward the
  cross-trophy median slope (few-read trophies borrow strength).
- **Rate:** r_i(target) = 100 * E_i(target) / P(target), clamped to
  0.01..99.99. The 100% trophy stays near 100%.

The fit uses only the **distinct reads** — Exophase's page is stale between
its own refreshes (hours apart), so consecutive identical reads carry no
signal and are collapsed to their first occurrence.

## No-signal fallback (the practice cases)

With fewer than 3 distinct reads there is no slope signal, so the forecast
replays the known **first refresh** as a structural prior: at the first
refresh the tracked base jumped ~6.06x (638 → 3,868 players) while the mean
non-100 rate fell ~4.56x, because newly tracked players held far fewer
trophies. Players multiply by 6.06; the new earners (enough for the mean
rate to fall 4.56x) are spread over trophies in proportion to
(E_i + 0.5)^2, so early/high-earner trophies dilute least and late/rare
ones most. The spread is wide (0.6 logit + random walk), because the
refresh timing and the per-trophy split are uncertain.

## Spreads

Quantiles live on the logit scale: q_tau = inv_logit(logit(median) +
z_tau * s). For the fitted model, s is the forecast's own uncertainty —
fit residual, slope standard error carried to the target, and a
misspecification allowance growing with the log-time extrapolation — not a
random walk from the last value. A fitted trend earns a tighter band than
persistence; a noisy fit widens it automatically.

## Validation (local, practice inputs p00/p01)

- Output valid: 55 rows, 5 non-decreasing quantiles each, 0..100,
  2,778 bytes (limit 65,536).
- Deterministic: two runs byte-identical. Runtime ~0.2 s for both cases
  (limit 8 s per case).
- Refresh sanity: median forecast / last rate ≈ 0.20–0.27 by trophy
  (early trophy 20 falls least at 0.27x, platinum most at 0.20x); trophy 2
  stays at 99.9%; implied mean non-100 median rate 0.194% — exactly the
  known post-refresh mean (last mean 0.885%).
- Proxy pinball loss vs a uniform-dilution answer: 0.17 (house baseline
  1.41); vs a no-new-earners answer: 0.21 (baseline 1.69).
- Synthetic 11-read final-like case (players 638 → 22,000, 4.5-day
  horizon): loss 0.13 vs baseline 0.22; medians within ~5% of the
  synthetic truth.

## Run

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network, deterministic.
