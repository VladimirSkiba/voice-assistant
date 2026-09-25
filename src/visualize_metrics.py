"""Build charts from the voice assistant metrics CSV."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_FIELDS = ("audio_sec", "asr_sec", "rtf", "llm_sec", "tts_sec", "e2e_sec")


def load_metrics(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"engine", *METRIC_FIELDS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"В CSV отсутствуют колонки: {', '.join(sorted(missing))}"
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                parsed = {"engine": row["engine"] or "unknown"}
                parsed.update({
                    field: float(row[field]) for field in METRIC_FIELDS
                })
            except (TypeError, ValueError) as error:
                print(f"[WARNING] Пропущена строка {line_number}: {error}")
                continue
            rows.append(parsed)

    if not rows:
        raise ValueError("CSV не содержит корректных строк с метриками")
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_visualization(rows: list[dict], output_path: Path) -> None:
    by_engine = defaultdict(list)
    for row in rows:
        by_engine[row["engine"]].append(row)

    engines = list(by_engine)
    colors = plt.get_cmap("tab10").colors
    engine_colors = {
        engine: colors[index % len(colors)]
        for index, engine in enumerate(engines)
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Voice Assistant Metrics", fontsize=16, fontweight="bold")

    latency_fields = ("asr_sec", "llm_sec", "tts_sec", "e2e_sec")
    latency_labels = ("ASR", "LLM", "TTS", "E2E")
    x_positions = list(range(len(engines)))
    bar_width = 0.18
    for offset, (field, label) in enumerate(zip(latency_fields, latency_labels)):
        values = [mean([row[field] for row in by_engine[engine]]) for engine in engines]
        positions = [x + (offset - 1.5) * bar_width for x in x_positions]
        axes[0, 0].bar(positions, values, width=bar_width, label=label)
    axes[0, 0].set_title("Средние задержки по этапам")
    axes[0, 0].set_ylabel("Секунды")
    axes[0, 0].set_xticks(x_positions, engines)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.25)

    rtf_values = [mean([row["rtf"] for row in by_engine[engine]]) for engine in engines]
    axes[0, 1].bar(engines, rtf_values, color=[engine_colors[engine] for engine in engines])
    axes[0, 1].set_title("Средний RTF")
    axes[0, 1].set_ylabel("ASR time / audio duration")
    axes[0, 1].grid(axis="y", alpha=0.25)

    for engine in engines:
        e2e_values = [row["e2e_sec"] for row in by_engine[engine]]
        axes[1, 0].plot(
            range(1, len(e2e_values) + 1),
            e2e_values,
            marker="o",
            label=engine,
            color=engine_colors[engine],
        )
    axes[1, 0].set_title("E2E по тестам")
    axes[1, 0].set_xlabel("Номер теста для движка")
    axes[1, 0].set_ylabel("Секунды")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    for engine in engines:
        engine_rows = by_engine[engine]
        axes[1, 1].scatter(
            [row["audio_sec"] for row in engine_rows],
            [row["e2e_sec"] for row in engine_rows],
            label=engine,
            color=engine_colors[engine],
            s=55,
            alpha=0.85,
        )
    axes[1, 1].set_title("Длительность аудио и E2E")
    axes[1, 1].set_xlabel("Аудио, секунды")
    axes[1, 1].set_ylabel("E2E, секунды")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    by_engine = defaultdict(list)
    for row in rows:
        by_engine[row["engine"]].append(row)

    print("\nСводка по движкам:")
    print("engine\ttests\tavg_asr_sec\tavg_rtf\tavg_e2e_sec")
    for engine, engine_rows in by_engine.items():
        print(
            f"{engine}\t{len(engine_rows)}\t"
            f"{mean([row['asr_sec'] for row in engine_rows]):.2f}\t"
            f"{mean([row['rtf'] for row in engine_rows]):.2f}\t"
            f"{mean([row['e2e_sec'] for row in engine_rows]):.2f}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "metrics.csv",
        help="Путь к CSV с метриками",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "metrics_visualization.png",
        help="Путь для PNG с графиками",
    )
    args = parser.parse_args()

    rows = load_metrics(args.input)
    build_visualization(rows, args.output)
    print_summary(rows)
    print(f"\nГрафики сохранены: {args.output}")


if __name__ == "__main__":
    main()
