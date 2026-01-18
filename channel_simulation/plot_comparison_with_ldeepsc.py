import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Auto-detect latest full_comparison JSON
def _latest_full_comparison_json(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("full_comparison_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No full_comparison_*.json found in {results_dir}. Run run_full_comparison.py first."
        )
    return candidates[-1]

# Auto-detect latest baseline_comparison JSON
def _latest_baseline_comparison_json(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("baseline_comparison_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No baseline_comparison_*.json found in {results_dir}."
        )
    return candidates[-1]

FUZSEMCOM_JSON = _latest_full_comparison_json(ROOT / "channel_simulation" / "results")
LDEEPSC_JSON = ROOT / "experiments" / "l_deepsc_system_opt" / "results" / "l_deepsc_system_opt_results.json"
BASELINE_JSON = _latest_baseline_comparison_json(ROOT / "experiments" / "baseline_results")
OUT_PNG = ROOT / "channel_simulation" / "results" / "comparison_acc_vs_snr_final.png"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _pick_first_key(d: dict, candidates: list[str]) -> str:
    for k in candidates:
        if k in d:
            return k
    raise KeyError(f"None of keys found: {candidates}. Available keys: {list(d.keys())}")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _extract_fuzsemcom_rayleigh(fsc: dict):
    """
    Return (snr_list, raw_symbol_acc_list, used_key)
    Note: In many of our JSONs, this is misnamed 'semantic_accuracy' but actually is symbol correctness.
    """
    results = fsc.get("results", fsc)
    if "rayleigh" not in results:
        # try case-insensitive
        key = None
        for k in results.keys():
            if str(k).lower() == "rayleigh":
                key = k
                break
        if key is None:
            raise KeyError(f"Cannot find rayleigh in results keys: {list(results.keys())}")
        ray = results[key]
    else:
        ray = results["rayleigh"]

    snr_key = _pick_first_key(ray, ["snr_db", "snr", "snr_list"])
    acc_key = _pick_first_key(ray, [
        "symbol_accuracy",
        "channel_symbol_accuracy",
        "decoded_symbol_accuracy",
        "semantic_accuracy",  # often misnamed in existing files
        "accuracy",
        "acc",
    ])

    snr = list(map(float, ray[snr_key]))
    raw = list(map(float, ray[acc_key]))
    if len(snr) != len(raw):
        raise ValueError(f"len(snr)={len(snr)} != len(acc)={len(raw)}")

    return snr, raw, acc_key


def _extract_ldeepsc_accuracy_after(ld: dict):
    """
    Support schema:
    - {"results_by_snr": {"0": {"accuracy_after": ...}, "5": {...}}}
    """
    if "results_by_snr" not in ld or not isinstance(ld["results_by_snr"], dict):
        raise KeyError("Expected L-DeepSC JSON to contain dict key 'results_by_snr'")

    rb = ld["results_by_snr"]
    snr = sorted([float(k) for k in rb.keys()])

    acc = []
    for s in snr:
        k_int = str(int(s))
        row = rb[k_int] if k_int in rb else rb[str(s)]
        acc_key = _pick_first_key(row, ["accuracy_after", "acc_after", "semantic_accuracy_after", "accuracy"])
        acc.append(float(row[acc_key]))

    return snr, acc


def _extract_fuzsemcom_intrinsic_acc(baseline: dict) -> float:
    """
    Extract FuzSemCom intrinsic accuracy from baseline comparison JSON.
    Looks for "FuzSemCom (Ours)" -> "accuracy"
    """
    fsc_key = None
    for k in baseline.keys():
        if "fuzsemcom" in str(k).lower() or "ours" in str(k).lower():
            fsc_key = k
            break
    
    if fsc_key is None:
        raise KeyError(f"Cannot find FuzSemCom entry in baseline JSON. Available keys: {list(baseline.keys())}")
    
    fsc_data = baseline[fsc_key]
    if "accuracy" not in fsc_data:
        raise KeyError(f"FuzSemCom entry '{fsc_key}' does not contain 'accuracy' key")
    
    return float(fsc_data["accuracy"])


def _extract_ldeepsc_clean_acc(ld: dict) -> float:
    """
    Extract L-DeepSC clean accuracy from L-DeepSC JSON.
    Uses accuracy_before (which is constant across all SNR values).
    """
    if "results_by_snr" not in ld or not isinstance(ld["results_by_snr"], dict):
        raise KeyError("Expected L-DeepSC JSON to contain dict key 'results_by_snr'")
    
    # Get accuracy_before from any SNR (they're all the same)
    rb = ld["results_by_snr"]
    first_snr = sorted([float(k) for k in rb.keys()])[0]
    first_key = str(int(first_snr))
    row = rb[first_key] if first_key in rb else rb[str(first_snr)]
    
    acc_key = _pick_first_key(row, ["accuracy_before", "acc_before", "semantic_accuracy_before", "accuracy"])
    return float(row[acc_key])


def main():
    fsc = _load_json(FUZSEMCOM_JSON)
    ld = _load_json(LDEEPSC_JSON)
    baseline = _load_json(BASELINE_JSON)

    # Extract intrinsic/clean accuracies from JSON files
    fsc_intrinsic_acc = _extract_fuzsemcom_intrinsic_acc(baseline)
    ldeepsc_clean_acc = _extract_ldeepsc_clean_acc(ld)

    # --- FuzSemCom: Effective semantic accuracy (Rayleigh) ---
    snr_fsc, raw_symbol_acc, used_key = _extract_fuzsemcom_rayleigh(fsc)
    eff_fsc = [_clamp01(x) * fsc_intrinsic_acc for x in raw_symbol_acc]

    # --- L-DeepSC curve: accuracy_after ---
    snr_ld, acc_after = _extract_ldeepsc_accuracy_after(ld)
    acc_after = [_clamp01(x) for x in acc_after]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    ax.plot(
        snr_fsc,
        eff_fsc,
        marker="s",
        linewidth=2.2,
        label=f"FuzSemCom (Rayleigh) Effective = ({used_key}) x {fsc_intrinsic_acc:.4f}",
    )
    ax.plot(
        snr_ld,
        acc_after,
        marker="o",
        linewidth=2.2,
        label="L-DeepSC accuracy_after",
    )

    # Upper bounds (clean) - values extracted from JSON
    ax.axhline(
        y=fsc_intrinsic_acc,
        linestyle="--",
        linewidth=2.0,
        label=f"FuzSemCom Upper Bound (Clean) = {fsc_intrinsic_acc:.4f}",
    )
    ax.axhline(
        y=ldeepsc_clean_acc,
        linestyle="--",
        linewidth=2.0,
        label=f"L-DeepSC Upper Bound (Clean) = {ldeepsc_clean_acc:.4f}",
    )

    ax.set_title("Effective Semantic Accuracy vs SNR (Fig. 3)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170)
    plt.close(fig)

    print(f"[OK] Saved: {OUT_PNG}")
    print(f"[INFO] FuzSemCom used accuracy field: {used_key}")
    print(f"[INFO] FuzSemCom intrinsic accuracy: {fsc_intrinsic_acc:.4f} (from baseline JSON)")
    print(f"[INFO] L-DeepSC clean accuracy: {ldeepsc_clean_acc:.4f} (from L-DeepSC JSON)")
    print(f"[INFO] FuzSemCom JSON: {FUZSEMCOM_JSON}")
    print(f"[INFO] L-DeepSC JSON: {LDEEPSC_JSON}")
    print(f"[INFO] Baseline JSON: {BASELINE_JSON}")


if __name__ == "__main__":
    main()
