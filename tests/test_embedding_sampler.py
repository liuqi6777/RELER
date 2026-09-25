"""Verify epoch regrouping through real datasets, workers and Accelerate sharding."""
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import prepare_data_loader
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reler.config import DataArguments
from reler.data.embedding import EmbeddingDataset, SingleSourceBatchSampler
from reler.training.trainer import build_single_source_sampler


def write_records(path, source_counts, bucket="short"):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        dict(id=f"{source}-{bucket}-{i}", schema="embedding_candidates_v2",
             source=source, bucket=bucket, query=f"query {i}", document=["a", "b"],
             relevance=[1, 0], ranking=[1, 2])
        for source, count in source_counts.items() for i in range(count)
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def make_dataset(path, **kwargs):
    random.seed(42)
    args = DataArguments(data_path=str(path), per_dataset_max_samples=None, **kwargs)
    return EmbeddingDataset(args, batch_size=4)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "train.ready.jsonl"
    write_records(path, {"biology": 19, "earth_science": 26, "economics": 3})
    data = make_dataset(path, index_cache_dir=str(tmp_path / "cache"))
    yield data
    data.close()


def batches(indices, size=4):
    return [indices[i:i + size] for i in range(0, len(indices), size)]


def companions(grouped):
    # Ignore both batch order and within-batch order: test who is paired with whom.
    return {frozenset(batch) for batch in grouped}


def collect_metadata(records):
    # Module-level function can be pickled by spawn-based DataLoader workers.
    return [(r["id"], r["source"], r["bucket"]) for r in records]


def test_epoch_changes_companions_without_changing_retained_rows(dataset):
    assert len(dataset) == 40  # Per-source tails of 3, 2 and 3 stay discarded.
    original_entries = dataset.entries.copy()
    sampler = SingleSourceBatchSampler(dataset, 4, seed=42)
    epoch_companions = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert len(sampler) == len(indices) == 40
        assert sorted(indices) == list(range(len(dataset)))
        grouped = batches(indices)
        for batch in grouped:
            assert len(batch) == 4
            assert len({dataset[i]["source"] for i in batch}) == 1
        epoch_companions.append(companions(grouped))
        assert dataset.entries == original_entries
    assert epoch_companions[0] != epoch_companions[1]
    assert epoch_companions[1] != epoch_companions[2]


def test_epoch_regrouping_preserves_length_buckets(tmp_path):
    for source, bucket, count in (("biology", "0-500", 19),
                                  ("biology", "500-1000", 14),
                                  ("earth_science", "0-500", 26)):
        write_records(tmp_path / source / f"train_len-{bucket}.jsonl", {source: count}, bucket)
    data = make_dataset(tmp_path, file_glob="*_len-*.jsonl", batch_per_length_bucket=True,
                        index_cache_dir=str(tmp_path / "cache"))
    try:
        assert len(data) == 52  # 16 + 12 + 24; tails are still dropped per bucket.
        sampler = SingleSourceBatchSampler(data, 4, seed=42)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            assert sorted(indices) == list(range(52))
            for batch in batches(indices):
                assert len({(data[i]["source"], data[i]["bucket"]) for i in batch}) == 1
    finally:
        data.close()


def test_seed_epoch_replay_is_independent_of_global_rng(dataset):
    first = SingleSourceBatchSampler(dataset, 4, seed=3407)
    first.set_epoch(2)
    torch_state = torch.random.get_rng_state()
    python_state = random.getstate()
    expected = list(first)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    torch.rand(37)
    random.random()
    second = SingleSourceBatchSampler(dataset, 4, seed=3407)
    for epoch in range(3):
        second.set_epoch(epoch)
        actual = list(second)
    assert actual == expected
    different_seed = SingleSourceBatchSampler(dataset, 4, seed=2026)
    different_seed.set_epoch(2)
    assert companions(batches(list(different_seed))) != companions(batches(expected))
    sequential = SingleSourceBatchSampler(dataset, 4, shuffle=False)
    for epoch in (0, 2):
        sequential.set_epoch(epoch)
        assert list(sequential) == list(range(len(dataset)))


def test_persistent_workers_receive_regrouped_indices(dataset):
    sampler = SingleSourceBatchSampler(dataset, 4, seed=42)
    loader = prepare_data_loader(
        DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2,
                   multiprocessing_context="spawn", persistent_workers=True,
                   collate_fn=collect_metadata),
        device=torch.device("cpu"), num_processes=1, process_index=0,
    )
    observed = []
    worker_pids = None
    original_entries = dataset.entries.copy()
    try:
        for epoch in range(2):
            loader.set_epoch(epoch)
            assert sampler.epoch == epoch
            actual = list(loader)
            expected = [collect_metadata([dataset[i] for i in batch])
                        for batch in batches(list(sampler))]
            assert actual == expected
            pids = [worker.pid for worker in loader.base_dataloader._iterator._workers]
            if worker_pids is not None:
                assert pids == worker_pids
            worker_pids = pids
            observed.append(companions([[r[0] for r in batch] for batch in actual]))
        assert observed[0] != observed[1]
        assert dataset.entries == original_entries
    finally:
        if loader.base_dataloader._iterator is not None:
            loader.base_dataloader._iterator._shutdown_workers()


@pytest.mark.parametrize("drop_last", [False, True])
def test_eight_rank_sharding_preserves_global_epoch_order(dataset, drop_last):
    loaders, samplers = [], []
    for rank in range(8):
        sampler = SingleSourceBatchSampler(dataset, 4, seed=42)
        samplers.append(sampler)
        loaders.append(prepare_data_loader(
            DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=drop_last,
                       collate_fn=collect_metadata),
            device=torch.device("cpu"), num_processes=8, process_index=rank,
            split_batches=False, even_batches=True,
        ))
    global_sampler = SingleSourceBatchSampler(dataset, 4, seed=42)
    for epoch in range(3):
        global_sampler.set_epoch(epoch)
        global_batches = [collect_metadata([dataset[i] for i in batch])
                          for batch in batches(list(global_sampler))]
        rank_batches = []
        for sampler, loader in zip(samplers, loaders):
            loader.set_epoch(epoch)
            assert sampler.epoch == epoch
            rank_batches.append(list(loader))
        assert {len(group) for group in rank_batches} == ({1} if drop_last else {2})
        interleaved = [batch for step in zip(*rank_batches) for batch in step]
        # Accelerate drops incomplete rank groups or pads with initial full batches.
        expected = global_batches[:8] if drop_last else global_batches + global_batches[:6]
        assert interleaved == expected
        assert all(len(batch) == 4 and len({r[1] for r in batch}) == 1 for batch in interleaved)


@pytest.mark.parametrize("data_seed", [None, 0, 3407])
def test_trainer_uses_data_seed_and_rejects_inconsistent_batch_size(dataset, data_seed):
    trainer = SimpleNamespace(_train_batch_size=4,
                              args=SimpleNamespace(train_batch_size=4, seed=42, data_seed=data_seed))
    sampler = build_single_source_sampler(trainer, dataset)
    expected = SingleSourceBatchSampler(dataset, 4, seed=42 if data_seed is None else data_seed)
    sampler.set_epoch(1)
    expected.set_epoch(1)
    assert list(sampler) == list(expected)
    trainer._train_batch_size = 8
    with pytest.raises(ValueError, match="must match"):
        build_single_source_sampler(trainer, dataset)
    with pytest.raises(ValueError, match="must match"):
        SingleSourceBatchSampler(dataset, 8)


@pytest.mark.parametrize("count", [0, 3, 4])
def test_empty_and_single_batch_groups(tmp_path, count):
    path = tmp_path / "train.ready.jsonl"
    write_records(path, {"biology": count})
    data = make_dataset(path, index_cache_dir=str(tmp_path / "cache"))
    try:
        sampler = SingleSourceBatchSampler(data, 4)
        for epoch in (0, 1):
            sampler.set_epoch(epoch)
            assert sorted(sampler) == list(range((count // 4) * 4))
    finally:
        data.close()
