from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reler.evaluation import get_tasks

DEFAULT_INPUT_DIR = "results/mteb"
DEFAULT_BENCHMARK = "MTEB(eng, v1, subset)"
SUPPORTED_VIEWS = ("run", "type", "task", "retrieval", "subset")

RETRIEVAL_TASK_TYPE = "Retrieval"


@dataclass
class RunSummary:
    run_id: str
    result_dir: Path
    benchmark: str
    task_types: dict[str, str]
    task_scores: dict[str, float]
    task_subset_scores: dict[str, dict[str, float]]
    errors: list[str]

    @property
    def task_count(self) -> int:
        return len(self.task_scores)

    @property
    def task_mean_pct(self) -> float | None:
        if not self.task_scores:
            return None
        return sum(self.task_scores.values()) / len(self.task_scores) * 100.0

    @property
    def type_scores(self) -> dict[str, float]:
        grouped_scores: dict[str, list[float]] = defaultdict(list)
        for task_name, score in self.task_scores.items():
            grouped_scores[self.task_types[task_name]].append(score)

        return {
            task_type: sum(scores) / len(scores)
            for task_type, scores in grouped_scores.items()
        }

    @property
    def retrieval_task_scores(self) -> dict[str, float]:
        return {
            task_name: score
            for task_name, score in self.task_scores.items()
            if self.task_types.get(task_name) == RETRIEVAL_TASK_TYPE
        }

    @property
    def retrieval_mean_pct(self) -> float | None:
        scores = list(self.retrieval_task_scores.values())
        if not scores:
            return None
        return sum(scores) / len(scores) * 100.0

    @property
    def type_mean_pct(self) -> float | None:
        scores = list(self.type_scores.values())
        if not scores:
            return None
        return sum(scores) / len(scores) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize MTEB results from a single result directory or recursively "
            f"scan a root directory (default: {DEFAULT_INPUT_DIR})."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help=(
            "A single MTEB result directory, or a root directory that contains "
            f"multiple result directories. Default: {DEFAULT_INPUT_DIR}"
        ),
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        default=DEFAULT_BENCHMARK,
        help=f"Benchmark name used to validate task files. Default: {DEFAULT_BENCHMARK}",
    )
    parser.add_argument(
        "--views",
        default="run,type",
        help=(
            "Comma-separated views to print/export. Supported values: "
            f"{', '.join(SUPPORTED_VIEWS)}. Default: run,type"
        ),
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help=(
            "If set, export one CSV per selected view into this directory. "
            "Example: --csv-dir results/mteb/_summary"
        ),
    )
    parser.add_argument(
        "--sort-runs-by",
        choices=("name", "task_mean", "type_mean"),
        default="task_mean",
        help="How to sort runs in the summary output. Default: task_mean",
    )
    return parser.parse_args()


def load_task_catalog(benchmark: str) -> dict[str, str]:
    tasks = get_tasks(names=None, languages=None, benchmark=benchmark)
    return {task.metadata.name: task.metadata.type for task in tasks}


def is_valid_task_file(filename: str, task_catalog: dict[str, str]) -> bool:
    return filename.endswith(".json") and filename[:-5] in task_catalog


def list_valid_task_files(directory: Path, task_catalog: dict[str, str]) -> list[Path]:
    if not directory.is_dir():
        return []
    task_files = [
        directory / entry.name
        for entry in directory.iterdir()
        if entry.is_file() and is_valid_task_file(entry.name, task_catalog)
    ]
    return sorted(task_files)


def discover_result_dirs(input_path: Path, task_catalog: dict[str, str]) -> list[Path]:
    input_path = input_path.resolve()
    direct_task_files = list_valid_task_files(input_path, task_catalog)
    if direct_task_files:
        return [input_path]

    result_dirs: list[Path] = []
    for root, dirnames, filenames in os.walk(input_path):
        dirnames[:] = [dirname for dirname in dirnames if not dirname.startswith(".")]
        if any(is_valid_task_file(filename, task_catalog) for filename in filenames):
            result_dirs.append(Path(root))
            dirnames[:] = []

    return sorted(result_dirs)


def parse_task_subset_scores(payload: dict[str, Any]) -> dict[str, float]:
    scores = payload.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("missing or invalid `scores` field")

    _, split_scores = next(iter(scores.items()))
    if not isinstance(split_scores, list) or not split_scores:
        raise ValueError("first eval split does not contain a non-empty score list")

    subset_scores = {
        str(entry.get("hf_subset", "default")): float(entry["main_score"])
        for entry in split_scores
        if "main_score" in entry
    }
    if not subset_scores:
        raise ValueError("no `main_score` found in the first eval split")
    return subset_scores


def parse_task_score(payload: dict[str, Any]) -> float:
    subset_scores = parse_task_subset_scores(payload)
    return sum(subset_scores.values()) / len(subset_scores)


def make_run_id(result_dir: Path, input_path: Path) -> str:
    def simplify_run_name(raw: str) -> str:
        normalized = raw.lstrip("/")
        first_segment = normalized.split("/", 1)[0] if normalized else ""
        if not first_segment:
            first_segment = Path(raw).name
        return first_segment

    resolved_result_dir = result_dir.resolve()
    resolved_input_path = input_path.resolve()
    try:
        relative_path = resolved_result_dir.relative_to(resolved_input_path)
    except ValueError:
        relative_path = resolved_result_dir

    relative_str = relative_path.as_posix()
    if relative_str not in ("", "."):
        return simplify_run_name(relative_str)

    if result_dir.parent.name:
        return simplify_run_name(result_dir.parent.name)
    return simplify_run_name(result_dir.name)


def summarize_result_dir(
    result_dir: Path,
    input_path: Path,
    benchmark: str,
    task_catalog: dict[str, str],
) -> RunSummary:
    task_scores: dict[str, float] = {}
    task_subset_scores: dict[str, dict[str, float]] = {}
    errors: list[str] = []

    for task_file in list_valid_task_files(result_dir, task_catalog):
        task_name = task_file.stem
        try:
            with task_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            subset_scores = parse_task_subset_scores(payload)
            task_subset_scores[task_name] = subset_scores
            task_scores[task_name] = sum(subset_scores.values()) / len(subset_scores)
        except Exception as exc:
            errors.append(f"{task_name}: {exc}")

    task_types = {task_name: task_catalog[task_name] for task_name in task_scores}
    return RunSummary(
        run_id=make_run_id(result_dir, input_path),
        result_dir=result_dir.resolve(),
        benchmark=benchmark,
        task_types=task_types,
        task_scores=task_scores,
        task_subset_scores=task_subset_scores,
        errors=errors,
    )


def missing_tasks(summary: RunSummary, task_catalog: dict[str, str]) -> list[str]:
    return sorted(
        task_name for task_name in task_catalog if task_name not in summary.task_scores
    )


def format_score(score_pct: float | None) -> str:
    if score_pct is None:
        return "N/A"
    return f"{score_pct:.2f}"


def make_run_rows(
    summaries: list[RunSummary],
    task_catalog: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "run": summary.run_id,
                "tasks_found": summary.task_count,
                "tasks_missing": len(task_catalog) - summary.task_count,
                "types_found": len(summary.type_scores),
                "mean_task_score": round(summary.task_mean_pct or 0.0, 2),
                "mean_type_score": round(summary.type_mean_pct or 0.0, 2),
                "retrieval": round(summary.retrieval_mean_pct or 0.0, 2),
                "errors": len(summary.errors),
                "result_dir": summary.result_dir.as_posix(),
            }
        )
    return rows


def make_retrieval_rows(summaries: list[RunSummary]) -> list[dict[str, Any]]:
    """Return a stable per-task retrieval breakdown for each run."""
    task_names = sorted(
        {
            task_name
            for summary in summaries
            for task_name in summary.retrieval_task_scores
        }
    )
    rows = []
    for summary in summaries:
        scores = summary.retrieval_task_scores
        row: dict[str, Any] = {"run": summary.run_id}
        for task_name in task_names:
            row[task_name] = (
                round(scores[task_name] * 100.0, 2) if task_name in scores else None
            )
        row["avg"] = round(summary.retrieval_mean_pct or 0.0, 2)
        rows.append(row)
    return rows


def print_retrieval_view(summaries: list[RunSummary]) -> None:
    rows = make_retrieval_rows(summaries)
    if not rows or len(rows[0]) <= 3:
        print("\n[Retrieval] no retrieval tasks in these results")
        return
    columns = [(key, key) for key in rows[0]]
    display = [
        {
            key: (
                format_score(value)
                if isinstance(value, float)
                else ("-" if value is None else value)
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    print("\n[Retrieval per task]")
    print(build_table(display, columns))


def make_type_rows(summaries: list[RunSummary]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        grouped_tasks: dict[str, list[str]] = defaultdict(list)
        for task_name, task_type in summary.task_types.items():
            grouped_tasks[task_type].append(task_name)

        mean_task_score = round(summary.task_mean_pct or 0.0, 2)
        mean_type_score = round(summary.type_mean_pct or 0.0, 2)
        for task_type, mean_score in sorted(
            summary.type_scores.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            rows.append(
                {
                    "run": summary.run_id,
                    "type": task_type,
                    "task_count": len(grouped_tasks[task_type]),
                    "mean_score": round(mean_score * 100.0, 2),
                    "mean_task_score": mean_task_score,
                    "mean_type_score": mean_type_score,
                }
            )
    return rows


def make_type_pivot_rows(summaries: list[RunSummary]) -> list[dict[str, Any]]:
    all_types = sorted(
        {task_type for summary in summaries for task_type in summary.type_scores}
    )
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row: dict[str, Any] = {"run": summary.run_id}
        for task_type in all_types:
            score = summary.type_scores.get(task_type)
            row[task_type] = round(score * 100.0, 2) if score is not None else ""

        row["mean_task_score"] = round(summary.task_mean_pct or 0.0, 2)
        row["mean_type_score"] = round(summary.type_mean_pct or 0.0, 2)
        rows.append(row)
    return rows


def make_task_rows(summaries: list[RunSummary]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for task_name, score in sorted(
            summary.task_scores.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            rows.append(
                {
                    "run": summary.run_id,
                    "task": task_name,
                    "type": summary.task_types[task_name],
                    "score": round(score * 100.0, 2),
                }
            )
    return rows


def make_subset_rows(summaries: list[RunSummary]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for task_name, subset_scores in summary.task_subset_scores.items():
            for subset, score in subset_scores.items():
                rows.append(
                    {
                        "run": summary.run_id,
                        "task": task_name,
                        "subset": subset,
                        "score": round(score * 100.0, 2),
                    }
                )
    return rows


def build_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "(empty)"

    headers = [header for _, header in columns]
    body = [[str(row[key]) for key, _ in columns] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        for index in range(len(columns))
    ]

    def render_row(values: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in body)
    return "\n".join(lines)


def print_run_summary(
    summaries: list[RunSummary], task_catalog: dict[str, str]
) -> None:
    rows = make_run_rows(summaries, task_catalog)
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                **row,
                "mean_task_score": format_score(row["mean_task_score"]),
                "mean_type_score": format_score(row["mean_type_score"]),
                "retrieval": format_score(row["retrieval"]),
            }
        )

    print("\n[Run Summary]")
    print(
        build_table(
            display_rows,
            [
                ("run", "Run"),
                ("tasks_found", "Tasks"),
                ("tasks_missing", "Missing"),
                ("types_found", "Types"),
                ("mean_task_score", "Mean(Task)"),
                ("mean_type_score", "Mean(Type)"),
                ("retrieval", "Retr."),
                ("errors", "Errors"),
                ("result_dir", "Result Dir"),
            ],
        )
    )

    print("\n[Missing Tasks]")
    for summary in summaries:
        missed = missing_tasks(summary, task_catalog)
        if missed:
            print(f"- {summary.run_id}: {len(missed)} missing")
            print(f"  {', '.join(missed)}")
        else:
            print(f"- {summary.run_id}: none")

    error_summaries = [summary for summary in summaries if summary.errors]
    if error_summaries:
        print("\n[Parse Errors]")
        for summary in error_summaries:
            print(f"- {summary.run_id}")
            for error in summary.errors:
                print(f"  {error}")


def print_grouped_view(
    summaries: list[RunSummary],
    view: str,
) -> None:
    if view == "retrieval":
        print_retrieval_view(summaries)
        return
    if view == "type":
        print("\n[Type Summary]")
        for summary in summaries:
            rows = [
                {
                    "type": row["type"],
                    "task_count": row["task_count"],
                    "mean_score": format_score(row["mean_score"]),
                    "mean_task_score": format_score(row["mean_task_score"]),
                    "mean_type_score": format_score(row["mean_type_score"]),
                }
                for row in make_type_rows([summary])
            ]
            print(f"\n{summary.run_id}")
            print(
                build_table(
                    rows,
                    [
                        ("type", "Type"),
                        ("task_count", "Tasks"),
                        ("mean_score", "Mean Score"),
                        ("mean_task_score", "Mean(Task)"),
                        ("mean_type_score", "Mean(Type)"),
                    ],
                )
            )
        return

    if view == "task":
        print("\n[Task Summary]")
        for summary in summaries:
            rows = [
                {
                    "task": row["task"],
                    "type": row["type"],
                    "score": format_score(row["score"]),
                }
                for row in make_task_rows([summary])
            ]
            print(f"\n{summary.run_id}")
            print(
                build_table(
                    rows,
                    [
                        ("task", "Task"),
                        ("type", "Type"),
                        ("score", "Score"),
                    ],
                )
            )
        return

    if view == "subset":
        print("\n[Subset Summary]")
        for summary in summaries:
            rows = [
                {**row, "score": format_score(row["score"])}
                for row in make_subset_rows([summary])
            ]
            print(f"\n{summary.run_id}")
            print(
                build_table(
                    rows,
                    [
                        ("task", "Task"),
                        ("subset", "Subset"),
                        ("score", "Score"),
                    ],
                )
            )


def export_csv(
    csv_dir: Path,
    view: str,
    rows: list[dict[str, Any]],
) -> Path:
    csv_dir.mkdir(parents=True, exist_ok=True)
    output_path = csv_dir / f"{view}_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
    return output_path


def sort_summaries(summaries: list[RunSummary], sort_by: str) -> list[RunSummary]:
    if sort_by == "name":
        return sorted(summaries, key=lambda summary: summary.run_id)

    if sort_by == "type_mean":
        return sorted(
            summaries,
            key=lambda summary: (
                -(
                    summary.type_mean_pct
                    if summary.type_mean_pct is not None
                    else float("-inf")
                ),
                summary.run_id,
            ),
        )

    return sorted(
        summaries,
        key=lambda summary: (
            -(
                summary.task_mean_pct
                if summary.task_mean_pct is not None
                else float("-inf")
            ),
            summary.run_id,
        ),
    )


def parse_views(raw_views: str) -> list[str]:
    views = [view.strip() for view in raw_views.split(",") if view.strip()]
    invalid_views = [view for view in views if view not in SUPPORTED_VIEWS]
    if invalid_views:
        raise ValueError(
            f"unsupported views: {', '.join(invalid_views)}; "
            f"supported views: {', '.join(SUPPORTED_VIEWS)}"
        )
    if not views:
        raise ValueError("at least one view must be selected")
    return views


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    views = parse_views(args.views)
    task_catalog = load_task_catalog(args.benchmark)
    result_dirs = discover_result_dirs(input_path, task_catalog)
    if not result_dirs:
        raise RuntimeError(
            f"no valid result directories found under {input_path.resolve()} "
            f"for benchmark {args.benchmark}"
        )

    summaries = [
        summarize_result_dir(
            result_dir=result_dir,
            input_path=input_path,
            benchmark=args.benchmark,
            task_catalog=task_catalog,
        )
        for result_dir in result_dirs
    ]
    summaries = [
        summary for summary in summaries if summary.task_scores or summary.errors
    ]
    if not summaries:
        raise RuntimeError(
            "found candidate result directories, but no valid task scores were parsed"
        )

    summaries = sort_summaries(summaries, args.sort_runs_by)

    print(f"Benchmark: {args.benchmark}")
    print(f"Input path: {input_path.resolve()}")
    print(f"Discovered result dirs: {len(summaries)}")
    print(f"Selected views: {', '.join(views)}")
    print(f"Expected benchmark tasks: {len(task_catalog)}")

    if "run" in views:
        print_run_summary(summaries, task_catalog)
    for view in ("retrieval", "type", "task", "subset"):
        if view in views:
            print_grouped_view(summaries, view)

    if args.csv_dir:
        csv_dir = Path(args.csv_dir).expanduser()
        exported_files = []
        if "run" in views:
            exported_files.append(
                export_csv(csv_dir, "run", make_run_rows(summaries, task_catalog))
            )
        if "type" in views:
            exported_files.append(
                export_csv(csv_dir, "type", make_type_pivot_rows(summaries))
            )
        if "task" in views:
            exported_files.append(
                export_csv(csv_dir, "task", make_task_rows(summaries))
            )
        if "retrieval" in views:
            exported_files.append(
                export_csv(csv_dir, "retrieval", make_retrieval_rows(summaries))
            )
        if "subset" in views:
            exported_files.append(
                export_csv(csv_dir, "subset", make_subset_rows(summaries))
            )

        print("\n[CSV Export]")
        for output_path in exported_files:
            print(f"- {output_path.resolve()}")


if __name__ == "__main__":
    main()
