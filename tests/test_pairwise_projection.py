"""Pair-local SVD oracles, masked endpoints, and shared-encoder integration."""
import pytest
import torch
import torch.nn.functional as F
from test_shortlists import (
    _distributed_worker,
    components,
    metadata,
    options,
    reference_loss,
)

import reler.objectives.shortlists as shortlist_module
from reler.config import RLArguments
from reler.objectives.policy_math import mean_alignment
from reler.objectives.projection import pairwise_shortlist_loss
from reler.objectives.shortlists import sample_shortlists
from reler.training.grpo_model import GRPO


def explicit_pair_loss(means, q, docs, positives, valid, fixed, fixed_mask, kappa, scale):
    """Slow full SVD per cell/endpoint, independent of the vectorized implementation."""
    batch, gq, dim = q.shape
    gd, count = docs.shape[1:3]
    coefficient = torch.zeros_like(means, dtype=torch.float64)
    def project(action, *columns):
        matrix = torch.stack(columns, -1).double()
        u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
        u = u[:, s > 1e-10]
        return u @ (u.T @ action.double())
    with torch.no_grad():
        h = F.normalize(means.detach().double(), dim=-1)
        for b in range(batch):
            ps = (positives[b] & valid[b]).nonzero().flatten().tolist()
            ns = (valid[b] & ~positives[b]).nonzero().flatten().tolist()
            ns += [count+k for k in fixed_mask[b].nonzero().flatten().tolist()]
            if not ns:
                continue
            for p in ps:
                for n in ns:
                    da = docs[b, :, p]
                    dn = docs[b, :, n] if n < count else scale * fixed[b, n-count].expand(gd, dim)
                    sa, sn = q[b] @ da.T, q[b] @ dn.T
                    reward = (sa > sn).double() + .5 * (sa == sn).double()
                    for i in range(gq):
                        for j in range(gd):
                            aq = reward[i,j] - torch.cat((reward[:i,j], reward[i+1:,j])).mean()
                            ad = reward[i,j] - torch.cat((reward[i,:j], reward[i,j+1:])).mean()
                            divisor = len(ps)*len(ns)*gq*gd*batch
                            coefficient[b,0] += -kappa * aq * project(q[b,i], h[b,0], da[j]-dn[j]) / divisor
                            for m in (p, n) if n < count else (p,):
                                coefficient[b,m+1] += -kappa * ad * project(docs[b,j,m], h[b,m+1], q[b,i]) / divisor
    return (means * coefficient.to(means)).sum()


@pytest.mark.parametrize('scale', [1., .35])
@pytest.mark.parametrize('chunk', [1, 7, 32])
@pytest.mark.parametrize('cached', [False, True])
def test_pair_cp_matches_full_cellwise_svd_and_only_credits_endpoints(scale, chunk, cached):
    torch.manual_seed(38)
    source = torch.randn(2, 5, 11)
    weight = torch.eye(11, requires_grad=True)
    means = F.normalize(source @ weight.T, dim=-1)
    q = F.normalize(torch.randn(2, 3, 11), dim=-1).requires_grad_()
    docs = F.normalize(torch.randn(2, 4, 4, 11), dim=-1).requires_grad_()
    fixed = F.normalize(torch.randn(2, 2, 11), dim=-1).requires_grad_()
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    positives = torch.tensor([[True, True, False, False], [True, True, False, False]])
    mask = torch.tensor([[True, True], [False, False]])
    kwargs = (dict(own_scores=torch.einsum('bid,bjmd->bijm', q, docs),
                   cross_scores=torch.einsum('bid,bkd->bik', q, fixed) * scale) if cached else {})
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, stats = pairwise_shortlist_loss(means[:,0], means[:,1:], q, docs, positives,
            valid, fixed, mask, 9., frozen_scale=scale, chunk_size=chunk, **kwargs)
    actual, fixed_grad, q_grad, doc_grad = torch.autograd.grad(loss, (weight, fixed, q, docs), allow_unused=True)
    assert fixed_grad is None and q_grad is None and doc_grad is None
    ref_weight = torch.eye(11, requires_grad=True)
    ref = F.normalize(source @ ref_weight.T, dim=-1)
    expected = explicit_pair_loss(ref, q, docs, positives, valid, fixed, mask, 9., scale)
    torch.testing.assert_close(actual, torch.autograd.grad(expected, ref_weight)[0], atol=2e-6, rtol=3e-5)
    assert actual.norm() > 0
    assert stats['reward/pairwise/pairs_mean'] == 3  # Six pairs then zero, averaged over queries.
    assert stats['reward/pairwise/no_pairs_frac'] == .5
    assert stats['projection/pairwise_query_span_rank_max'] == 2
    # Per-input gradients for the zero-pair query and padded documents are zero.
    live = means.detach().requires_grad_()
    direct, _ = pairwise_shortlist_loss(live[:,0], live[:,1:], q, docs, positives, valid, fixed, mask, 9.)
    grad = torch.autograd.grad(direct, live)[0]
    assert grad[1].norm() == 0 and grad[0,-1].norm() == 0


@pytest.mark.parametrize('no_pairs', [False, True])
def test_ties_zero_pairs_and_collinear_directions_are_finite(no_pairs):
    x = torch.tensor([[[1.,0.,0.], [1.,0.,0.], [1.,0.,0.]]], requires_grad=True)
    h = F.normalize(x, dim=-1)
    q = h[:,0,None].expand(1,3,3).detach()
    docs = h[:,None,1:].expand(1,4,2,3).detach()
    loss, stats = pairwise_shortlist_loss(h[:,0],h[:,1:],q,docs,
        torch.tensor([[True,no_pairs]]),torch.ones(1,2,dtype=torch.bool),
        torch.empty(1,0,3),torch.empty(1,0,dtype=torch.bool),9.)
    assert torch.isfinite(loss) and loss == 0
    assert torch.autograd.grad(loss,x)[0].norm() == 0
    assert stats['reward/pairwise/mean'] == (0. if no_pairs else .5)
    assert stats['reward/pairwise/active_pair_fraction'] == 0


@pytest.mark.parametrize('coefficient,binary_weight', [(.25,0.), (.5,0.), (.25,.25), (.5,.25)])
def test_full_shortlist_objective_matches_separate_list_and_pair_oracles(monkeypatch, coefficient, binary_weight):
    torch.manual_seed(43)
    source = torch.randn(3, 4, 13)
    weight = torch.eye(13, requires_grad=True)
    means = F.normalize(source @ weight.T, dim=-1)
    q = F.normalize(torch.randn(3,3,13), dim=-1)
    valid = torch.tensor([[True]*3, [True]*3, [True,True,False]])
    docs = F.normalize(torch.randn(3,4,3,13),dim=-1).masked_fill(~valid[:,None,:,None],0)
    labels = torch.tensor([[0.,3.,1.], [3.,0.,0.], [0.,3.,0.]])
    positives = torch.tensor([[True,False,True], [True,False,False], [True,False,False]])
    captured = []
    def sample(*args, **kwargs):
        result = sample_shortlists(*args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(shortlist_module,'sample_shortlists',sample)
    head = GRPO(**options(reward_shortlist_pairwise_coef=coefficient,
                          reward_shortlist_binary_weight=binary_weight, reward_shortlist_count=2))
    loss, stats, _, _ = head._compute_component_loss(labels,None,components(means,q,docs,valid),
        candidate_mask=valid,cross_batch_metadata=metadata(),positive_mask=positives)
    actual = torch.autograd.grad(loss,weight)[0]
    ref_weight = torch.eye(13, requires_grad=True)
    ref = F.normalize(source @ ref_weight.T,dim=-1)
    idx,mask,_ = captured[0]
    pool = ref[:,1:].detach().reshape(-1,13)
    scale = mean_alignment(13,9.)
    main,_ = reference_loss(ref,q,docs,labels,valid,pool,idx,mask,'conditional_projection',scale)
    if binary_weight:
        binary,_ = reference_loss(ref,q,docs,positives.float(),valid,pool,idx,mask,'conditional_projection',scale)
        main = (1-binary_weight)*main + binary_weight*binary
    pairs = sum(explicit_pair_loss(ref,q,docs,positives,valid,pool[idx[:,t]],mask[:,t],9.,scale)
                for t in range(idx.size(1))) / idx.size(1)
    expected = main + coefficient * pairs
    torch.testing.assert_close(actual,torch.autograd.grad(expected,ref_weight)[0],atol=3e-6,rtol=4e-5)
    torch.testing.assert_close(stats['train/loss_pairwise_weighted'],coefficient*stats['train/loss_pairwise'])
    torch.testing.assert_close(stats['reward/combined_mean'],stats['reward_mean']+coefficient*stats['reward/pairwise/mean'])


@pytest.mark.parametrize('changes', [dict(reward_shortlist_pairwise_coef=-1),
    dict(reward_shortlist_pairwise_coef=float('nan')), dict(reward_shortlist_pairwise_coef=float('inf')),
    dict(reward_shortlist_pairwise_coef=.25,reward_shortlist_count=0),
    dict(reward_shortlist_pairwise_coef=.25,gradient_estimator='score_function')])
def test_invalid_pair_configuration_rejected(changes):
    for constructor in (RLArguments,GRPO):
        with pytest.raises(ValueError):
            constructor(**options(**changes))


def test_two_rank_pair_gradient_with_uneven_tails_and_empty_cross_pool(tmp_path):
    torch.multiprocessing.spawn(_distributed_worker,
        args=(str(tmp_path / 'rendezvous'), 'conditional_projection', 'cross_device_all', 0., .25),
        nprocs=2, join=True)
