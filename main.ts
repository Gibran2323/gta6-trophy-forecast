// GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).
//
// Model: earners and players are forecast SEPARATELY in log-time, and the
// rate is their ratio. Exophase's page is stale between its own refreshes
// (hours apart), so the fit uses only the DISTINCT reads (a read whose
// players-tracked count or rate vector changed), never the stale repeats.
//
//   players  P(t): concave quadratic of log P on x = ln(1 + hours since
//              launch). Concave, because the tracked base grows fast at
//              launch and then saturates; a convex fit would explode on a
//              4.5-day extrapolation, so a positive quadratic term is
//              dropped and the fit falls back to a line.
//   earners  E_i(t): per trophy, log(E_i + 0.5) is linear in x once the
//              players curve's curvature is removed, so a trophy's
//              earners inherit the tracked base's saturation instead of
//              extrapolating launch night's explosive slope in a straight
//              line. The slope is shrunk toward the cross-trophy median
//              slope (trophies with few distinct reads borrow strength
//              instead of chasing noise). Early trophies grow fast,
//              late/platinum ones slow; each trophy keeps its own speed.
//   rate     r_i(target) = 100 * E_i(target) / P(target), clamped.
//
// When the history holds fewer than 3 distinct reads there is no slope
// signal at all (the practice cases: one flat read). Then the forecast
// falls back to a structural REFRESH PRIOR, because the practice rewards a
// forecast that expects Exophase to refresh before the target: at the first
// refresh the tracked base jumped ~6.06x (638 -> 3,868 players) while the
// mean non-100 rate fell ~4.56x, since the newly tracked players held
// far fewer trophies. The prior replays that refresh: players multiply by
// 6.06, the new earners (+~33% on the old total, so the mean rate falls
// 4.56x) are spread over trophies in proportion to (E_i + 0.5)^2, so
// early/high-earner trophies dilute less than late/rare ones, and the
// 100% trophy stays at 100%. The spread is wide, because the refresh
// timing and the per-trophy split are uncertain.
//
// Quantiles live on the logit scale: q_tau = inv_logit(logit(median) +
// z_tau * s), with s the forecast's own uncertainty: the fit residual,
// the fitted slope's standard error carried to the target, and a
// misspecification allowance that grows with the log-time extrapolation.
// A fitted trend earns a tighter band than persistence does (no
// random-walk term: the median is a trend, not a random walk from the
// last value); a noisy fit widens the band automatically. The refresh
// prior keeps a wide spread, because a discrete refresh is genuinely
// uncertain.
//
// Node 22, standard library only, no network, deterministic: the same
// case always writes the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Scored quantile levels, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at the scored levels. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk spread, logit units per square-root day (house baseline's). */
const RW_SIGMA = 0.5;
/** First-refresh player multiplier: 638 -> 3,868 tracked players. */
const REFRESH_PLAYER_MULT = 6.06;
/** First-refresh mean dilution of non-100 rates: 0.885% -> 0.194%. */
const REFRESH_MEAN_DILUTION = 4.56;
/** Hours between Exophase refreshes on launch night, roughly. */
const REFRESH_INTERVAL_H = 4.2;

interface Trophy {
  id: number;
  name: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; release_date: string };
  trophies: Trophy[];
  history: Read[];
}

interface DistinctRead {
  tMs: number;
  players: number;
  rates: Array<number | null>;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logitPct = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length;
  return m === 0 ? 0 : m % 2 === 1 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
};

/** Solve a small dense linear system by Gaussian elimination with pivoting. */
function solveLinear(a: number[][], b: number[]): number[] | null {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-12) return null;
    [m[col], m[piv]] = [m[piv], m[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col] / m[col][col];
      for (let k = col; k <= n; k++) m[r][k] -= f * m[col][k];
    }
  }
  return m.map((row, i) => row[n] / m[i][i]);
}

/** Least-squares line y = a + b*x; returns slope, intercept, residual std and slope SE. */
function linFit(xs: number[], ys: number[]): { a: number; b: number; resid: number; seB: number } {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (ys[i] - (a + b * xs[i])) ** 2;
  const resid = n > 2 ? Math.sqrt(ss / (n - 2)) : 0;
  const seB = sxx > 1e-12 ? resid / Math.sqrt(sxx) : 0;
  return { a, b, resid, seB };
}

/** Least-squares quadratic y = a + b*x + c*x^2; null when underdetermined. */
function quadFit(xs: number[], ys: number[]): { a: number; b: number; c: number; resid: number } | null {
  const n = xs.length;
  if (n < 3) return null;
  let s0 = n;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  let t0 = 0;
  let t1 = 0;
  let t2 = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const y = ys[i];
    const x2 = x * x;
    s1 += x;
    s2 += x2;
    s3 += x2 * x;
    s4 += x2 * x2;
    t0 += y;
    t1 += x * y;
    t2 += x2 * y;
  }
  const sol = solveLinear(
    [
      [s0, s1, s2],
      [s1, s2, s3],
      [s2, s3, s4],
    ],
    [t0, t1, t2],
  );
  if (sol === null) return null;
  const [a, b, c] = sol;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (ys[i] - (a + b * xs[i] + c * xs[i] * xs[i])) ** 2;
  const resid = n > 3 ? Math.sqrt(ss / (n - 3)) : 0;
  return { a, b, c, resid };
}

/** Collapse the history to its distinct states, keeping each state's first time. */
function distinctReads(history: Read[]): DistinctRead[] {
  const out: DistinctRead[] = [];
  for (const r of history) {
    if (!r.players_tracked || r.players_tracked <= 0) continue;
    const prev = out[out.length - 1];
    if (prev !== undefined && prev.players === r.players_tracked && prev.rates.length === r.rates.length) {
      let same = true;
      for (let i = 0; i < r.rates.length; i++) {
        if (prev.rates[i] !== r.rates[i]) {
          same = false;
          break;
        }
      }
      if (same) continue;
    }
    out.push({ tMs: Date.parse(r.at), players: r.players_tracked, rates: [...r.rates] });
  }
  return out;
}

/** Hours since launch, in log-time: x = ln(1 + hours). */
function logTime(tMs: number, launchMs: number): number {
  return Math.log(1 + Math.max(0, (tMs - launchMs) / 3_600_000));
}

/** One trophy's forecast: median rate (percent) and logit spread. */
interface Forecast {
  median: number;
  spread: number;
}

/** The refresh prior: replay the known first refresh from the last read. */
function refreshPrior(c: Case, last: Read): Forecast[] {
  const p0 = last.players_tracked;
  const p1 = p0 * REFRESH_PLAYER_MULT;
  const days = c.horizon_minutes / 1440;
  // Earners now, per trophy (null rates fall back to the tier median rate).
  const tierRates = new Map<string, number[]>();
  c.trophies.forEach((t, i) => {
    const r = last.rates[i];
    if (r !== null && r !== undefined) {
      const arr = tierRates.get(t.tier) ?? [];
      arr.push(r);
      tierRates.set(t.tier, arr);
    }
  });
  const allRates = last.rates.filter((r): r is number => r !== null && r !== undefined);
  const overallMedian = median(allRates);
  const earners: number[] = c.trophies.map((t, i) => {
    let r = last.rates[i];
    if (r === null || r === undefined) {
      const tier = tierRates.get(t.tier) ?? [];
      r = tier.length > 0 ? median(tier) : overallMedian;
    }
    return (r * p0) / 100;
  });
  // New earners: enough that the mean non-100 rate falls REFRESH_MEAN_DILUTION-fold.
  // They concentrate on the early trophies: a trophy's share of the new
  // earners grows with the square of its current earners, so the trophies
  // new players reach first dilute least and the late/rare ones most.
  let sumE = 0;
  let sumW = 0;
  const weights: number[] = earners.map((e, i) => {
    const r = last.rates[i];
    if (r !== null && r !== undefined && r >= 99.99) return 0;
    sumE += e;
    const w = (e + 0.5) ** 2;
    sumW += w;
    return w;
  });
  const newEarnersTotal = sumE * (REFRESH_PLAYER_MULT / REFRESH_MEAN_DILUTION - 1);
  // The refresh is the dominant uncertainty, but the median is pinned by
  // the known refresh magnitudes, so a 0.6-logit spread covers the
  // per-trophy split without paying the full width of a blind guess.
  const spread = Math.sqrt(0.6 * 0.6 + (RW_SIGMA * Math.sqrt(days)) ** 2);
  return c.trophies.map((t, i) => {
    const r = last.rates[i];
    if (r !== null && r !== undefined && r >= 99.99) {
      // The 100% trophy stayed at 100% through the first refresh.
      return { median: 99.9, spread: 0.15 };
    }
    const e1 = earners[i] + (sumW > 0 ? (newEarnersTotal * weights[i]) / sumW : 0);
    const med = clampPct((e1 / p1) * 100);
    return { median: med, spread };
  });
}

/** The fitted model: separate log-time curves for players and earners. */
function fittedForecast(c: Case, distinct: DistinctRead[], launchMs: number): Forecast[] {
  const targetMs = Date.parse(c.target);
  const xTarget = logTime(targetMs, launchMs);
  const days = c.horizon_minutes / 1440;
  const xs = distinct.map((d) => logTime(d.tMs, launchMs));
  const xLast = xs[xs.length - 1];

  // Players: concave quadratic of log P on x, else a line. Its curvature
  // is the shared saturation shape: the tracked base grows fast at launch
  // and then slows, and the earners curves inherit it below.
  const logP = distinct.map((d) => Math.log(d.players));
  let logPTarget: number;
  let cP = 0;
  {
    const q = quadFit(xs, logP);
    if (q !== null && q.c <= 0) {
      logPTarget = q.a + q.b * xTarget + q.c * xTarget * xTarget;
      cP = q.c;
    } else {
      const l = linFit(xs, logP);
      logPTarget = l.a + l.b * xTarget;
      cP = 0;
    }
  }
  const pLast = distinct[distinct.length - 1].players;
  const pTarget = Math.min(pLast * 100, Math.max(pLast, Math.exp(logPTarget)));

  // Earners: per-trophy curve of log(E + 0.5) on x with the PLAYERS'
  // curvature fixed in, so a trophy's earners inherit the base's
  // saturation instead of extrapolating launch night's explosive slope
  // in a straight line. Only the level and the slope are the trophy's
  // own; the slope is shrunk toward the cross-trophy median slope.
  const n = c.trophies.length;
  const slopes: number[] = [];
  const resids: number[] = [];
  const perTrophy: Array<{ xs: number[]; ys: number[] } | null> = [];
  for (let i = 0; i < n; i++) {
    const px: number[] = [];
    const py: number[] = [];
    for (let k = 0; k < distinct.length; k++) {
      const r = distinct[k].rates[i];
      if (r === null || r === undefined) continue;
      const e = (r * distinct[k].players) / 100;
      // Remove the shared curvature before fitting the line.
      px.push(xs[k]);
      py.push(Math.log(e + 0.5) - cP * xs[k] * xs[k]);
    }
    if (px.length >= 2 && new Set(px).size >= 2) {
      const f = linFit(px, py);
      slopes.push(f.b);
      resids.push(f.resid);
      perTrophy.push({ xs: px, ys: py });
    } else {
      perTrophy.push(null);
    }
  }
  const poolSlope = slopes.length > 0 ? median(slopes) : 1.0;
  const poolResid = resids.length > 0 ? median(resids) : 0.3;
  const last = distinct[distinct.length - 1];

  const extrap = Math.max(0, xTarget - xLast);
  return c.trophies.map((t, i) => {
    const lastRate = last.rates[i];
    if (lastRate !== null && lastRate !== undefined && lastRate >= 99.99) {
      return { median: 99.9, spread: 0.1 };
    }
    const data = perTrophy[i];
    let eTarget: number;
    let sFit: number;
    let sParam: number;
    if (data !== null) {
      const f = linFit(data.xs, data.ys);
      const m = data.xs.length;
      const bStar = (m * f.b + 3 * poolSlope) / (m + 3);
      let aStar = 0;
      for (let j = 0; j < m; j++) aStar += data.ys[j] - bStar * data.xs[j];
      aStar /= m;
      // Add the shared curvature back in for the target.
      eTarget = Math.exp(aStar + bStar * xTarget + cP * xTarget * xTarget) - 0.5;
      // Fit residual, floored: a 2-point fit has no residual by construction.
      sFit = Math.min(1.2, Math.max(0.15, f.resid));
      if (m < 4) sFit = Math.max(sFit, poolResid);
      // Slope uncertainty grows with the distance from the fit's centre.
      const xMean = data.xs.reduce((s, v) => s + v, 0) / m;
      const seB = Math.max(f.seB, sFit / Math.sqrt(Math.max(1, m)) / 2);
      sParam = seB * Math.max(0, xTarget - xMean);
    } else if (lastRate !== null && lastRate !== undefined) {
      // One usable point: the trophy's own level, the pooled slope.
      const e0 = (lastRate * pLast) / 100;
      eTarget =
        Math.exp(Math.log(e0 + 0.5) + poolSlope * (xTarget - xLast) + cP * (xTarget * xTarget - xLast * xLast)) - 0.5;
      sFit = Math.max(0.3, poolResid);
      sParam = 0.5 * extrap + 0.3;
    } else {
      // Never observed: fall back to the cross-trophy median rate, wide spread.
      const seen = last.rates.filter((r): r is number => r !== null && r !== undefined && r < 99.99);
      const medRate = seen.length > 0 ? median(seen) : 0.5;
      const e0 = (medRate * pLast) / 100;
      eTarget =
        Math.exp(Math.log(e0 + 0.5) + poolSlope * (xTarget - xLast) + cP * (xTarget * xTarget - xLast * xLast)) - 0.5;
      sFit = 1.2;
      sParam = 0.5 * extrap + 0.3;
    }
    eTarget = Math.min(pTarget, Math.max(0, eTarget));
    const med = clampPct((eTarget / pTarget) * 100);
    // The spread is the forecast's own uncertainty, not a random walk from
    // the last value: the fit residual, the slope's standard error carried
    // to the target, and a misspecification allowance that grows with the
    // log-time extrapolation. A fitted trend earns a tighter band than
    // persistence does; a noisy fit widens it automatically.
    const s = Math.min(
      1.5,
      Math.max(0.25, Math.sqrt(sFit * sFit + sParam * sParam + (0.3 * extrap) ** 2 + 0.1 * 0.1)),
    );
    return { median: med, spread: s };
  });
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history || c.history.length === 0) throw new Error(`${c.id} has no history`);
  const launchMs = Number.isNaN(Date.parse(c.game.release_date))
    ? Date.parse(c.history[0].at)
    : Date.parse(`${c.game.release_date}T00:00:00Z`);
  const distinct = distinctReads(c.history);
  const last = c.history[c.history.length - 1];
  const forecasts = distinct.length < 3 ? refreshPrior(c, last) : fittedForecast(c, distinct, launchMs);
  return c.trophies.map((t, i) => {
    const { median: med, spread: s } = forecasts[i];
    const l = logitPct(med);
    const qs: number[] = [];
    let prev = 0.01;
    for (const z of Z) {
      const q = clampPct(round4(invLogitPct(l + z * s)));
      qs.push(Math.max(prev, q));
      prev = qs[qs.length - 1];
    }
    return { id: t.id, q: qs };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
