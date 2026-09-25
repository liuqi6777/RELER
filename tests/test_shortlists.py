"""Coverage, independent sampling, and joint gradients for multiple shortlists."""
import json
from datetime import timedelta

import pytest
import torch
import torch.nn.functional as F
from test_conditional_projection import explicit_projection

import reler.objectives.shortlists as shortlist_module
from reler.config import RLArguments
from reler.objectives.policy_math import mean_alignment
from reler.objectives.rewards import compute_reward_from_scores
from reler.objectives.rollout_rng import RolloutRNG
from reler.objectives.shortlists import sample_shortlists, shortlist_contract
from reler.training.grpo_model import GRPO, GRPOModel, _ActionComponent
from reler.training.trainer import GRPOTrainer, restore_exploration_state


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('pool_size,size,hard,band,count', [
    (100, 15, 8, 64, 4), (100, 15, 8, 16, 9), (70, 15, 8, 64, 6),
    (9, 15, 8, 64, 4), (1, 15, 8, 64, 3), (0, 15, 8, 64, 4),
    (100, 15, 0, 64, 8), (100, 15, 15, 15, 4),
])
def test_coverage_is_maximal_and_no_list_contains_duplicates(pool_size, size, hard, band, count):
    scores = torch.arange(pool_size + 5, dtype=torch.float32)[None].repeat(2, 1)
    valid = torch.zeros_like(scores, dtype=torch.bool)
    valid[0, :pool_size] = True
    valid[1, :pool_size:2] = True
    torch.manual_seed(12)
    indices, mask, high = sample_shortlists(scores, valid, count=count, size=size,
                                           hard_count=hard, hard_pool_size=band)
    for b in range(2):
        n = int(valid[b].sum())
        selected = indices[b][mask[b]]
        assert valid[b, selected].all()
        assert selected.unique().numel() == min(n, count * size)
        for t in range(count):
            row = indices[b, t, mask[b, t]]
            assert row.numel() == min(n, size) == row.unique().numel()
        if n:
            # Across repeated traversals, no candidate can be overexposed by >1.
            frequencies = torch.bincount(selected, minlength=scores.size(1))[valid[b]]
            assert int(frequencies.max() - frequencies.min()) <= 1
        if hard and n >= band + (size-hard) * count and band >= hard * count:
            assert (high[b].sum(-1) == hard).all()
    assert not high[~mask].any()


def test_shortlist_rng_does_not_change_actions_and_resumes_at_step_boundaries():
    scores = torch.arange(100.).reshape(1, -1)
    valid = torch.ones_like(scores, dtype=torch.bool)
    def draw(rng, count):
        with rng.draw('cpu', 7, stream='shortlist'):
            return sample_shortlists(scores, valid, count=count, size=15, hard_count=8, hard_pool_size=64)
    rng = RolloutRNG(42)
    torch.manual_seed(5)
    ambient = torch.get_rng_state().clone()
    one = draw(rng, 1)
    with rng.draw('cpu', 7):
        actions = torch.rand(40)
    fresh = RolloutRNG(42)
    with fresh.draw('cpu', 7):
        expected_actions = torch.rand(40)
    four = draw(fresh, 4)
    torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
    torch.testing.assert_close(one[0][:, 0], four[0][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), ambient, rtol=0, atol=0)
    restored = RolloutRNG(42)
    for current in (rng, restored):
        with current.draw('cpu', 8, stream='shortlist'):
            value = torch.rand(10)
        if current is rng:
            continued = value
        else:
            torch.testing.assert_close(value, continued, rtol=0, atol=0)


def options(**changes):
    base = dict(action_components='query;positive,negative', group_size=4, kappa=9.,
                reward_type='ndcg_in_batch', reward_ndcg_k=2, rollout_seed=42,
                ndcg_in_batch_include_negatives=True, reward_cross_device_negatives=True,
                gradient_estimator='conditional_projection', reward_shortlist_count=4,
                reward_shortlist_size=2, reward_shortlist_hard_count=1, reward_shortlist_hard_pool_size=2)
    return {**base, **changes}


def metadata():
    rows = [dict(keys=[f'd{3*i+j}' for j in range(3)], ids=[None]*3,
                 source='test', known_ids=[], known_positive_keys=[]) for i in range(3)]
    rows[2]['keys'][2] = None
    rows[0]['known_positive_keys'] = ['d7']
    # One entire rank will have no eligible cross documents in the DDP test.
    rows[2]['known_positive_keys'] = [f'd{i}' for i in range(6)]
    return rows


def reference_loss(means, q, docs, labels, valid, pool, indices, masks, estimator, scale):
    own = torch.einsum('bid,bjmd->bijm', q, docs)
    total = means.new_zeros(())
    rewards = []
    for t in range(indices.size(1)):
        fixed = pool[indices[:, t]]
        cross = torch.einsum('bid,bkd->bik', q, fixed) * scale
        cross = cross.masked_fill(~masks[:, t, None], -torch.inf)
        reward = compute_reward_from_scores(
            own, labels, reward_type='ndcg_in_batch', k=2, candidate_mask=valid,
            ndcg_in_batch_include_negatives=True,
            in_batch_candidate_scores=cross[:, :, None].expand(-1, -1, docs.size(1), -1))
        rewards.append(reward)
        if estimator == 'conditional_projection':
            coefficient = explicit_projection(means.detach(), q, docs, reward, valid,
                                              fixed, masks[:, t], 9.)
            total = total + (means * coefficient.to(means)).sum()
        else:
            # Per-cell score estimator, with baselines explicitly omitting the draw.
            for i in range(q.size(1)):
                for j in range(docs.size(1)):
                    aq = reward[:, i, j] - torch.cat((reward[:, :i, j], reward[:, i+1:, j]), 1).mean(1)
                    ad = reward[:, i, j] - torch.cat((reward[:, i, :j], reward[:, i, j+1:]), 1).mean(1)
                    qlog = 9 * (means[:, 0] * q[:, i]).sum(-1)
                    dlog = 9 * (means[:, 1:] * docs[:, j]).masked_fill(~valid[..., None], 0).sum((1, 2))
                    total = total - (aq * qlog + ad * dlog).mean() / (q.size(1)*docs.size(1))
    return total / indices.size(1), torch.stack(rewards, 1)


def components(means, q, docs, valid):
    return (
        _ActionComponent(role='query', name='query', rollout_embeddings=means[:, 0].detach(),
                         policy_embeddings=means[:, 0], sampled_embeddings=q, kappa=torch.tensor(9.)),
        _ActionComponent(role='document', name='documents', rollout_embeddings=means[:, 1:].detach(),
                         policy_embeddings=means[:, 1:], sampled_embeddings=docs, kappa=torch.tensor(9.),
                         document_mask=valid),
    )


@pytest.mark.parametrize('estimator', ['score_function', 'conditional_projection'])
@pytest.mark.parametrize('rescale', [True, False])
def test_shared_encoder_gradient_matches_per_cell_oracle(monkeypatch, estimator, rescale):
    torch.manual_seed(8)
    source = torch.randn(3, 4, 17)
    weight = torch.eye(17, requires_grad=True)
    means = F.normalize(source @ weight.T, dim=-1)
    q = F.normalize(torch.randn(3, 4, 17), dim=-1)
    valid = torch.tensor([[True]*3, [True]*3, [True, True, False]])
    docs = F.normalize(torch.randn(3, 4, 3, 17), dim=-1).masked_fill(~valid[:, None, :, None], 0)
    labels = torch.tensor([[3., 1., 0.], [3., 0., 0.], [3., 0., 0.]])
    captured = []
    def sample(*args, **kwargs):
        result = sample_shortlists(*args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(shortlist_module, 'sample_shortlists', sample)
    head = GRPO(**options(gradient_estimator=estimator, frozen_doc_rescale=rescale))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, stats, _, _ = head._compute_component_loss(
            labels, None, components(means, q, docs, valid), candidate_mask=valid,
            cross_batch_metadata=metadata())
    actual = torch.autograd.grad(loss, weight)[0]
    reference_weight = torch.eye(17, requires_grad=True)
    ref = F.normalize(source @ reference_weight.T, dim=-1)
    idx, mask, _ = captured[0]
    expected, reward = reference_loss(ref, q, docs, labels, valid, ref[:, 1:].detach().reshape(-1, 17),
                                    idx, mask, estimator, mean_alignment(17, 9.) if rescale else 1.)
    reference = torch.autograd.grad(expected, reference_weight)[0]
    torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-5)
    assert actual.norm() > 0
    torch.testing.assert_close(stats['reward_mean'], reward.mean())
    torch.testing.assert_close(stats['reward_std'], reward.std(unbiased=False), atol=1e-6, rtol=1e-5)
    assert stats['shortlist/unique_candidates_mean'] == 3  # 4, 5, 0 after identity filtering.
    if estimator == 'conditional_projection':
        assert stats['projection/query_span_rank_max'] <= 6  # 1 query + 3 own + 2 selected.


def _distributed_worker(rank, rendezvous, estimator, pool_source, binary_weight=0., pairwise_coef=0.):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(91)
        source = torch.randn(3, 4, 12)
        q = F.normalize(torch.randn(3, 4, 12), dim=-1)
        valid = torch.tensor([[True]*3, [True]*3, [True, True, False]])
        docs = F.normalize(torch.randn(3, 4, 3, 12), dim=-1).masked_fill(~valid[:, None, :, None], 0)
        labels = torch.tensor([[3., 1., 0.], [3., 0., 0.], [3., 0., 0.]])
        positives = torch.tensor([[True, False, True], [True, False, False], [True, False, False]])
        start, stop, width = (0, 2, 3) if rank == 0 else (2, 3, 2)
        weight = torch.eye(12, requires_grad=True)
        local = F.normalize(source[start:stop, :width+1] @ weight.T, dim=-1)
        rows = [dict(row, keys=row['keys'][:width], ids=row['ids'][:width]) for row in metadata()[start:stop]]
        captured = []
        original = shortlist_module.sample_shortlists
        def sample(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result)
            return result
        shortlist_module.sample_shortlists = sample
        head = GRPO(**options(gradient_estimator=estimator, reward_shortlist_pool_source=pool_source,
                              reward_cross_device_negatives=pool_source != 'local_all',
                              reward_shortlist_binary_weight=binary_weight,
                              reward_shortlist_pairwise_coef=pairwise_coef))
        loss, _, _, _ = head._compute_component_loss(labels[start:stop, :width], None,
            components(local, q[start:stop], docs[start:stop, :, :width], valid[start:stop, :width]),
            candidate_mask=valid[start:stop, :width], cross_batch_metadata=rows,
            positive_mask=positives[start:stop, :width])
        loss.backward()
        actual = weight.grad.clone()
        dist.all_reduce(actual)
        actual /= 2
        selections = [None, None]
        indices, masks = captured[0][:2]
        if pool_source == 'local_all':
            # Map rank-local, possibly narrow slates into the global oracle pool.
            indices = start * 3 + indices // width * 3 + indices % width
            assert ((indices[masks] >= start * 3) & (indices[masks] < stop * 3)).all()
        elif pool_source == 'cross_device_representatives':
            indices = indices * 3
        # Independently check pool eligibility, including exhausted/empty pools.
        expected_counts = {
            'cross_device_all': [4, 5, 0],
            'local_all': [3, 3, 0],
            'cross_device_representatives': [2, 2, 0],
        }[pool_source][start:stop]
        for b, n in enumerate(expected_counts):
            selected = indices[b][masks[b]]
            assert selected.unique().numel() == n
            assert (masks[b].sum(-1) == min(n, 2)).all()
        dist.all_gather_object(selections, (indices, masks))
        idx = torch.cat([row[0] for row in selections])
        mask = torch.cat([row[1] for row in selections])
        reference_weight = torch.eye(12, requires_grad=True)
        ref = F.normalize(source @ reference_weight.T, dim=-1)
        expected, _ = reference_loss(ref, q, docs, labels, valid, ref[:, 1:].detach().reshape(-1, 12),
                                    idx, mask, estimator, mean_alignment(12, 9.))
        if binary_weight:
            binary, _ = reference_loss(ref, q, docs, positives.float(), valid,
                ref[:, 1:].detach().reshape(-1, 12), idx, mask, estimator, mean_alignment(12, 9.))
            expected = (1-binary_weight)*expected + binary_weight*binary
        if pairwise_coef:
            from test_pairwise_projection import explicit_pair_loss
            pool = ref[:,1:].detach().reshape(-1,12)
            pair_loss = sum(explicit_pair_loss(ref,q,docs,positives,valid,pool[idx[:,t]],mask[:,t],
                                               9.,mean_alignment(12,9.)) for t in range(idx.size(1))) / idx.size(1)
            expected = expected + pairwise_coef * pair_loss
        expected.backward()
        torch.testing.assert_close(actual, reference_weight.grad, atol=4e-6, rtol=4e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('estimator', ['score_function', 'conditional_projection'])
@pytest.mark.parametrize('source', ['cross_device_all', 'local_all', 'cross_device_representatives'])
def test_two_ranks_with_uneven_tails_and_no_cross_candidates_on_one_rank(tmp_path, estimator, source):
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path / 'rendezvous'), estimator, source),
                               nprocs=2, join=True)


@pytest.mark.parametrize('changes', [dict(reward_shortlist_count=-1), dict(reward_shortlist_size=0),
    dict(reward_shortlist_hard_count=3), dict(reward_shortlist_hard_pool_size=0),
    dict(reward_cross_device_negatives=False), dict(cross_query_document_gradients=True),
    dict(reward_shortlist_pool_source='unknown'),
    dict(reward_shortlist_pool_source='local_all'),
    dict(reward_shortlist_pool_source='cross_device_representatives', reward_shortlist_count=0),
    dict(reward_shortlist_pool_source='local_all', reward_cross_device_negatives=False,
         ndcg_in_batch_include_negatives=False),
    dict(rollout_seed=None), dict(advantage_norm='shared'), dict(advantage_baseline='group'),
    dict(sampling_law='gaussian'), dict(sigma_learnable=True)])
def test_invalid_policy_and_sampler_rejected_by_both_entrypoints(changes):
    for constructor in (RLArguments, GRPO):
        with pytest.raises(ValueError):
            constructor(**options(**changes))


@pytest.mark.parametrize('source', ['cross_device_all', 'local_all', 'cross_device_representatives'])
@pytest.mark.parametrize('binary_weight,pairwise_coef', [(0.,0.), (.25,0.), (0.,.25)])
def test_trainer_updates_saves_and_reuses_encoder_and_actions(tmp_path, monkeypatch, source, binary_weight, pairwise_coef):
    from transformers import BertConfig, BertModel, TrainingArguments
    torch.manual_seed(7)
    backbone = BertModel(BertConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, hidden_dropout_prob=0, attention_probs_dropout_prob=0))
    wrapper = GRPOModel(backbone, RLArguments(**options(reward_shortlist_pool_source=source,
        reward_cross_device_negatives=source != 'local_all', reward_shortlist_binary_weight=binary_weight,
        reward_shortlist_pairwise_coef=pairwise_coef)))
    calls = dict(encoder=0, actions=0, pool=0)
    originals = (backbone.forward, wrapper.grpo._draw_actions, shortlist_module.cross_query_document_pool)
    def count(name, fn):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)
        return wrapped
    monkeypatch.setattr(backbone, 'forward', count('encoder', originals[0]))
    monkeypatch.setattr(wrapper.grpo, '_draw_actions', count('actions', originals[1]))
    monkeypatch.setattr(shortlist_module, 'cross_query_document_pool', count('pool', originals[2]))
    def tokens(ids):
        return dict(input_ids=torch.tensor(ids)[:, None], attention_mask=torch.ones(len(ids), 1, dtype=torch.long))
    batch = dict(query=tokens([0, 1]), positive_document=tokens([2, 5]),
        negative_document=tokens([3, 4, 6, 7]), relevance_labels=torch.tensor([[3., 1., 0.], [3., 0., 0.]]),
        positive_mask=torch.tensor([[True, False, True], [True, False, False]]),
        candidate_mask=torch.ones(2, 3, dtype=torch.bool), cross_batch_metadata=metadata()[:2])
    before = backbone.embeddings.word_embeddings.weight.detach().clone()
    trainer = GRPOTrainer(model=wrapper, args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True,
        max_steps=1, per_device_train_batch_size=2, learning_rate=1e-3, logging_steps=1,
        save_steps=1, report_to=[], disable_tqdm=True, remove_unused_columns=False),
        train_dataset=[0, 1], data_collator=lambda _: batch)
    trainer.train()
    assert calls == dict(encoder=2, actions=2, pool=1)
    assert (backbone.embeddings.word_embeddings.weight.detach() - before).norm() > 0
    checkpoint = tmp_path / 'checkpoint-1'
    payload = json.loads((checkpoint / 'exploration_state.json').read_text())
    assert payload['reward_shortlists'] == shortlist_contract(wrapper.grpo)
    restore_exploration_state(wrapper, checkpoint)
    wrapper.grpo.reward_shortlist_binary_weight = .5
    with pytest.raises(ValueError):
        restore_exploration_state(wrapper, checkpoint)
    wrapper.grpo.reward_shortlist_binary_weight = binary_weight
    wrapper.grpo.reward_shortlist_pairwise_coef = .5
    with pytest.raises(ValueError):
        restore_exploration_state(wrapper, checkpoint)
    wrapper.grpo.reward_shortlist_pairwise_coef = pairwise_coef
    for name in ('count', 'size', 'hard_count', 'hard_pool_size'):
        attribute = f'reward_shortlist_{name}'
        old = getattr(wrapper.grpo, attribute)
        setattr(wrapper.grpo, attribute, old + 1)
        with pytest.raises(ValueError):
            restore_exploration_state(wrapper, checkpoint)
        setattr(wrapper.grpo, attribute, old)
    wrapper.grpo.reward_shortlist_pool_source = ('local_all' if source != 'local_all' else 'cross_device_all')
    with pytest.raises(ValueError):
        restore_exploration_state(wrapper, checkpoint)
    wrapper.grpo.reward_shortlist_pool_source = source
    payload.pop('reward_shortlists')
    payload.pop('config')  # Simulate a legacy checkpoint without the full spec.
    (checkpoint / 'exploration_state.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        restore_exploration_state(wrapper, checkpoint)
    wrapper.grpo.reward_shortlist_count = 0
    restore_exploration_state(wrapper, checkpoint)
