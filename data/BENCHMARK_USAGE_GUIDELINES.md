# Benchmark Usage Guidelines

Use this benchmark in a way that keeps results meaningful and comparable. These guidelines help avoid overfitting, temporal errors, and other common mistakes.

---

## 1. Do Not Overfit to the Benchmark

**Rule:** Treat the benchmark as a **held-out test set**. Do not tune your predictor on benchmark data.

- **Do not** iterate on your method using benchmark scores, then report those same scores as "final" performance.
- **Do** use a separate development/validation set (e.g. another date range or regime) for tuning and model selection. Run the benchmark only when you are done changing your method.
- **Optional:** If you must try many variants, reserve one benchmark regime (e.g. "falling") as a final test and use the others only for development; or use a single final run and report that.

**Why:** Repeated runs or tuning on the same benchmark inflate reported performance and make comparisons unfair.

---

## 2. Respect Temporal Ordering (No Look-Ahead)

**Rule:** At prediction time `T_0`, your predictor may use **only** information available strictly **before** `T_0`.

- Use **strict `<`** (not `<=`) for any time boundary when querying snapshots or historical blocks. See [RESEARCH_BEST_PRACTICES.md](../RESEARCH_BEST_PRACTICES.md#temporal-ordering-rules) and [TEMPORAL_ORDERING_GUARANTEES.md](../TEMPORAL_ORDERING_GUARANTEES.md).
- **Snapshot at T_0:** Only include transactions with `first_seen_timestamp < T_0`.
- **Historical blocks for "past 24h":** Only include blocks with `block_timestamp < T_0` (and within the lookback window).
- If you use the provided pipeline and `SnapshotWindow`/`target_time` correctly, temporal ordering is enforced; if you add custom queries or features, double-check them.

**Why:** Using future or "current moment" data that would not be available in production causes look-ahead bias and overstates accuracy.

---

## 3. Train/Test and Regime Discipline

**Rule:** Keep training, validation, and benchmark evaluation separate by **time** and/or **regime**.

- **Time split:** If you train on past data, evaluate on strictly later dates. Do not train on dates that overlap the benchmark evaluation window.
- **Regime split:** The benchmark has rising, stable, and falling regimes. Do not train on the same regime you are evaluating on unless you explicitly report it as "same-regime" and also report "unseen regime" results.
- **Cross-regime:** For a robust claim, report performance on **all three** regimes (rising, stable, falling); avoid cherry-picking the regime where your method looks best.

**Why:** Training on test data or on the same regime inflates scores; reporting only one regime can be misleading.

---

## 4. Report Fully and Consistently

**Rule:** Report enough detail for others to reproduce and compare.

- **Configuration:** Dataset name, date range, `start_time`, `num_trials`, `simulation_interval`, and target window (e.g. 24h, 1h).
- **Metrics:** Report the same metrics the benchmark uses (e.g. success rate, overpayment, MAE if applicable) for each regime you evaluate.
- **Single run vs multiple:** If you run the benchmark once, say so. If you run multiple times (e.g. different seeds), report mean and spread (e.g. std or min–max).
- **Code/data:** Note the predictor version and, if you changed anything, how the benchmark was run (script, config, or command).

**Why:** Incomplete reporting makes it hard to compare methods and to spot overfitting or mistakes.

---

## 5. Avoid Multiple-Comparison and Selection Bias

**Rule:** If you try many predictors or hyperparameters, do not report only the best run as if it were the only one.

- **Many variants:** Use a separate validation set to choose the best variant; then run the benchmark **once** (or a fixed number of times) for the chosen variant and report that. Alternatively, report results for all variants and clearly label which was selected and how.
- **Multiple regimes:** Report results for every regime you evaluate; do not only show the regime where your method wins.
- **Statistical claims:** If you claim "method A is better than B," use an appropriate comparison (e.g. same trials, same regime, same metrics) and, if possible, note uncertainty (e.g. confidence intervals or variance across runs).

**Why:** Picking the best of many runs or regimes without correction inflates reported performance and can be misleading.

---

## 6. Use the Benchmark as Intended

- **Target windows:** The benchmark supports different target windows (e.g. confirm within 1h, 24h, 3d). Use the same target and evaluation window for all methods you compare.
- **Block index:** Ensure the dataset has a block index (or complete block data) for the evaluation period so inclusion checks are valid. See the standard benchmark plan for setup.
- **Regimes:** Use the designated rising, stable, and falling datasets (and dates) so that results are comparable across studies.

**Why:** Consistent setup and definitions make benchmark numbers comparable and reproducible.

---

## Quick Checklist

Before reporting benchmark results, confirm:

- [ ] Predictor was **not** tuned on the benchmark (or on the same regime you are reporting).
- [ ] All time boundaries use **strict `<`** (no look-ahead); custom code has been checked.
- [ ] Training data does **not** overlap the benchmark evaluation period (and ideally not the same regime).
- [ ] Results are reported for **all** regimes (or clearly scoped) with **full config** and **metrics**.
- [ ] If many variants were tried, **selection** is clearly described and **reported** results are for a fixed, chosen setup.

For full detail on temporal rules and validation, see [RESEARCH_BEST_PRACTICES.md](../RESEARCH_BEST_PRACTICES.md) and [TEMPORAL_ORDERING_GUARANTEES.md](../TEMPORAL_ORDERING_GUARANTEES.md).
