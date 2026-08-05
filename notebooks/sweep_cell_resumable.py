#@title Run every baseline -- RESUMABLE, memory-bounded
# Replaces the original sweep cell. Safe to re-run after a crash: completed
# (dataset, tool, rate, seed) combinations are skipped, not recomputed.
import gc, logging, os, time
import pandas as pd, numpy as np

# Great Expectations builds an ephemeral context and a temp docs directory on
# every validate() call. Across 75 sweep iterations these accumulate and
# exhaust Colab RAM. Quieten it and force collection between runs.
for noisy in ("great_expectations", "great_expectations.data_context"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

RATES   = [0.01, 0.05, 0.10, 0.20, 0.30]   #@param
N_SEEDS = 5                                 #@param {type:"integer"}
SEEDS   = seed_list(20260803, N_SEEDS)

CKPT = METRICS / "baseline_comparison_checkpoint.csv"
COLS = ["dataset","tool","rate","seed","status","seconds",
        "precision","recall","f1","n_detected"]

if CKPT.exists():
    done_df = pd.read_csv(CKPT)
    done = set(zip(done_df.dataset, done_df.tool,
                   done_df.rate.round(4), done_df.seed))
    print(f"resuming: {len(done_df)} rows already complete")
else:
    done_df = pd.DataFrame(columns=COLS)
    done = set()
    done_df.to_csv(CKPT, index=False)
    print("starting fresh")


def _append(rows):
    """Append immediately so a crash never costs more than one iteration."""
    pd.DataFrame(rows, columns=COLS).to_csv(CKPT, mode="a", header=False, index=False)


t0 = time.time()
for name, clean in CLEAN.items():
    rules = RULES[name]
    nat = natural_defect_keys_full(clean, SPECS[name], rules.columns())
    print(f"\n{name}: excluding {len(nat):,} naturally defective cells")

    for rate in RATES:
        for seed in SEEDS:
            pending = [t for t in BASELINES
                       if (name, t, round(rate, 4), seed) not in done]
            if not pending:
                continue

            corrupted, gt = PLANS[name]().run(clean, rate, seed)
            res = run_baselines(corrupted, rules, tools=pending)

            rows = []
            for tool, r in res.items():
                if r["detections"] is None:
                    rows.append(dict(dataset=name, tool=tool, rate=rate, seed=seed,
                                     status=r["status"], seconds=np.nan,
                                     precision=np.nan, recall=np.nan, f1=np.nan,
                                     n_detected=0))
                    continue
                o = score_restricted(gt, r["detections"], nat).attrs["overall"]
                rows.append(dict(dataset=name, tool=tool, rate=rate, seed=seed,
                                 status="ok", seconds=r["seconds"],
                                 precision=o["precision"], recall=o["recall"],
                                 f1=o["f1"], n_detected=len(r["detections"])))
            _append(rows)

            # release everything before the next iteration
            del corrupted, gt, res, rows
            gc.collect()

        print(f"  rate {rate}: done ({time.time()-t0:.0f}s)")

bench = pd.read_csv(CKPT)
bench.to_csv(METRICS / "baseline_comparison_raw.csv", index=False)
print(f"\n{len(bench)} rows total in {time.time()-t0:.0f}s")
print(bench.groupby(['dataset','tool']).size().to_string())
