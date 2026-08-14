"""Compare EEGNet with EA + EEGNet LOSO results."""

from pathlib import Path

import pandas as pd


BASELINE_PATH = Path(
    "results/eegnet_loso/eegnet_loso_results.csv"
)

EA_PATH = Path(
    "results/eegnet_ea_loso/eegnet_ea_loso_results.csv"
)

OUTPUT_PATH = Path(
    "results/eegnet_alignment_comparison.csv"
)


def main() -> None:
    baseline = pd.read_csv(BASELINE_PATH)
    aligned = pd.read_csv(EA_PATH)

    baseline = baseline[
        [
            "held_out_subject",
            "accuracy_percent",
            "kappa",
        ]
    ].rename(
        columns={
            "accuracy_percent": "eegnet_accuracy_percent",
            "kappa": "eegnet_kappa",
        }
    )

    aligned = aligned[
        [
            "held_out_subject",
            "accuracy_percent",
            "kappa",
        ]
    ].rename(
        columns={
            "accuracy_percent": "ea_eegnet_accuracy_percent",
            "kappa": "ea_eegnet_kappa",
        }
    )

    comparison = baseline.merge(
        aligned,
        on="held_out_subject",
        validate="one_to_one",
    )

    comparison["accuracy_change"] = (
        comparison["ea_eegnet_accuracy_percent"]
        - comparison["eegnet_accuracy_percent"]
    )

    comparison["kappa_change"] = (
        comparison["ea_eegnet_kappa"]
        - comparison["eegnet_kappa"]
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(comparison.to_string(index=False))

    print()
    print(
        "EEGNet mean accuracy: "
        f"{comparison['eegnet_accuracy_percent'].mean():.4f}%"
    )

    print(
        "EA + EEGNet mean accuracy: "
        f"{comparison['ea_eegnet_accuracy_percent'].mean():.4f}%"
    )

    print(
        "Mean accuracy change: "
        f"{comparison['accuracy_change'].mean():+.4f} "
        "percentage points"
    )

    print()
    print(f"Saved comparison: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
