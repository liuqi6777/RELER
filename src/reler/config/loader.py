import argparse
import json
import os
import pathlib
import re
from copy import deepcopy

from transformers import HfArgumentParser

from .arguments import TrainingArguments

BASE_CONFIG_SLOTS = ("train", "dataset", "model", "grpo", "reward")
BASELINE_CONFIG_SLOTS = ("train", "dataset", "model", "baseline")
FIXED_GRPO_CONFIG_SLOTS = (*BASE_CONFIG_SLOTS, "corpus")


def load_raw_config_file(config_path: str) -> dict:
    config_path = os.path.abspath(config_path)
    suffix = pathlib.Path(config_path).suffix.lower()

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Reading YAML configs requires PyYAML. Install it with `pip install pyyaml`."
        ) from exc

    with open(config_path, "r", encoding="utf-8") as fp:
        if suffix == ".json":
            config = json.load(fp)
        elif suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(fp) or {}
        else:
            raise ValueError(
                f"Unsupported config file format: {config_path}. "
                "Expected one of: .json, .yaml, .yml."
            )

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_path}")

    return config


def merge_config_dicts(base_config: dict, override_config: dict) -> dict:
    merged = deepcopy(base_config)

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def normalize_base_entries(base_entries, config_path: str) -> list:
    """Coerce a `_base_` value (absent / string / list) into a list."""
    if base_entries is None:
        return []
    if isinstance(base_entries, str):
        return [base_entries]
    if isinstance(base_entries, list):
        return base_entries
    raise ValueError(f"`_base_` must be a string or a list of strings: {config_path}")


def resolve_config_inheritance(
    config_path: str, visited_paths: tuple[str, ...] = ()
) -> dict:
    return resolve_config_inheritance_from_dict(
        config=load_raw_config_file(config_path),
        config_path=config_path,
        visited_paths=visited_paths,
    )


def resolve_config_inheritance_from_dict(
    config: dict,
    config_path: str,
    visited_paths: tuple[str, ...] = (),
) -> dict:
    config_path = os.path.abspath(config_path)
    if config_path in visited_paths:
        cycle = " -> ".join((*visited_paths, config_path))
        raise ValueError(f"Detected config inheritance cycle: {cycle}")

    local_config = deepcopy(config)
    base_entries = normalize_base_entries(local_config.pop("_base_", []), config_path)
    merged_config: dict = {}
    config_dir = os.path.dirname(config_path)

    for base_entry in base_entries:
        if not isinstance(base_entry, str):
            raise ValueError(f"Each `_base_` entry must be a string: {config_path}")
        base_path = (
            base_entry
            if os.path.isabs(base_entry)
            else os.path.join(config_dir, base_entry)
        )
        resolved_base_config = resolve_config_inheritance(
            base_path,
            visited_paths=(*visited_paths, config_path),
        )
        merged_config = merge_config_dicts(merged_config, resolved_base_config)

    return merge_config_dicts(merged_config, local_config)


def apply_base_overrides(
    config_path: str, config: dict, base_overrides: dict[str, str]
) -> dict:
    if not base_overrides:
        return config

    updated_config = deepcopy(config)
    base_entries = normalize_base_entries(updated_config.get("_base_", []), config_path)

    replaced_slots: set[str] = set()
    resolved_entries: list[str] = []
    for base_entry in base_entries:
        if not isinstance(base_entry, str):
            raise ValueError(f"Each `_base_` entry must be a string: {config_path}")

        base_slot = pathlib.Path(base_entry).parent.name
        if base_slot in base_overrides:
            resolved_entries.append(os.path.abspath(base_overrides[base_slot]))
            replaced_slots.add(base_slot)
        else:
            resolved_entries.append(base_entry)

    missing_slots = sorted(set(base_overrides) - replaced_slots)
    if missing_slots:
        raise ValueError(
            f"Unknown `_base_` override slot(s) for {config_path}: {', '.join(missing_slots)}"
        )

    updated_config["_base_"] = resolved_entries
    return updated_config


def resolve_slot_config_paths(
    base_overrides: dict[str, str],
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> list[str]:
    invalid_slots = sorted(set(base_overrides) - set(base_slots))
    if invalid_slots:
        raise ValueError(f"Unsupported base config slot(s): {', '.join(invalid_slots)}")

    resolved_paths: list[str] = []
    for slot in base_slots:
        path = base_overrides.get(slot)
        if path:
            resolved_paths.append(os.path.abspath(path))
    return resolved_paths


def resolve_config_from_base_overrides(
    base_overrides: dict[str, str],
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> dict:
    merged_config: dict = {}
    for config_path in resolve_slot_config_paths(base_overrides, base_slots=base_slots):
        resolved_config = resolve_config_inheritance(config_path)
        merged_config = merge_config_dicts(merged_config, resolved_config)
    return merged_config


def slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-") or "run"


def build_auto_run_name(
    config_path: str | None,
    base_overrides: dict[str, str],
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> str:
    if base_overrides:
        parts: list[str] = []
        for slot in base_slots:
            slot_path = base_overrides.get(slot)
            if slot_path:
                stem = pathlib.Path(slot_path).stem
                if stem == "default":
                    continue
                parts.append(stem)
        if parts:
            return "__".join(slugify_name(part) for part in parts)

    if config_path:
        return slugify_name(pathlib.Path(config_path).stem)

    return "run"


def resolve_run_name_and_output_dir(
    config_path: str | None,
    base_overrides: dict[str, str],
    training_args: TrainingArguments,
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> None:
    output_dir = (training_args.output_dir or "").strip()
    run_name = (training_args.run_name or "").strip() if training_args.run_name else ""

    output_dir_is_default = output_dir in {"", "trainer_output"}
    run_name_is_default = run_name in {"", "trainer_output"}

    if run_name_is_default and not output_dir_is_default:
        run_name = pathlib.Path(output_dir).name

    if not run_name_is_default and output_dir_is_default:
        output_dir = str(pathlib.Path("checkpoints") / run_name)

    if run_name_is_default and output_dir_is_default:
        run_name = build_auto_run_name(
            config_path=config_path,
            base_overrides=base_overrides,
            base_slots=base_slots,
        )
        output_dir = str(pathlib.Path("checkpoints") / run_name)

    training_args.run_name = run_name
    training_args.output_dir = output_dir


def _parse_dict_with_cli_overrides(
    parser: HfArgumentParser,
    config: dict,
    cli_args: list[str] | None,
) -> tuple:
    if cli_args:
        override_config = parse_cli_overrides(
            dataclass_types=parser.dataclass_types,
            cli_args=cli_args,
        )
        config = merge_config_dicts(config, override_config)
    return parser.parse_dict(config)


def parse_config_file(
    parser: HfArgumentParser,
    config_path: str,
    cli_args: list[str] | None = None,
    base_overrides: dict[str, str] | None = None,
) -> tuple:
    root_config = apply_base_overrides(
        config_path=config_path,
        config=load_raw_config_file(config_path),
        base_overrides=base_overrides or {},
    )
    config = resolve_config_inheritance_from_dict(
        config=root_config,
        config_path=config_path,
    )
    return _parse_dict_with_cli_overrides(parser, config, cli_args)


def parse_config_from_base_overrides(
    parser: HfArgumentParser,
    base_overrides: dict[str, str],
    cli_args: list[str] | None = None,
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> tuple:
    config = resolve_config_from_base_overrides(base_overrides, base_slots=base_slots)
    return _parse_dict_with_cli_overrides(parser, config, cli_args)


def parse_cli_overrides(dataclass_types: list[type], cli_args: list[str]) -> dict:
    override_parser = HfArgumentParser(dataclass_types)
    for action in override_parser._actions:
        action.required = False
        if action.dest != "help":
            action.default = argparse.SUPPRESS

    namespace, remaining_args = override_parser.parse_known_args(cli_args)
    if remaining_args:
        joined_args = " ".join(remaining_args)
        raise ValueError(f"Unrecognized arguments: {joined_args}")

    return vars(namespace)


__all__ = [
    "BASE_CONFIG_SLOTS",
    "BASELINE_CONFIG_SLOTS",
    "FIXED_GRPO_CONFIG_SLOTS",
    "apply_base_overrides",
    "build_auto_run_name",
    "load_raw_config_file",
    "merge_config_dicts",
    "normalize_base_entries",
    "parse_cli_overrides",
    "parse_config_file",
    "parse_config_from_base_overrides",
    "resolve_config_from_base_overrides",
    "resolve_config_inheritance",
    "resolve_config_inheritance_from_dict",
    "resolve_run_name_and_output_dir",
    "resolve_slot_config_paths",
]
