from __future__ import annotations

import torch

from trainers import (
    masked_greedy_policy_from_logits,
    masked_softmax_from_logits,
)


def test_masked_softmax_assigns_zero_mass_to_blocked_actions():
    logits = torch.tensor([[10.0, 2.0, 8.0], [1.0, 5.0, 3.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])

    policy = masked_softmax_from_logits(logits, mask)

    assert torch.allclose(policy.sum(dim=1), torch.ones(2), atol=1e-7)
    assert policy[0, 0].item() == 0.0
    assert policy[1, 1].item() == 0.0
    assert torch.all(policy[mask < 0.5] == 0.0)


def test_masked_greedy_never_selects_blocked_argmax():
    logits = torch.tensor([[10.0, 2.0, 8.0], [1.0, 5.0, 3.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])

    policy = masked_greedy_policy_from_logits(logits, mask)
    selected = torch.argmax(policy, dim=1)

    assert selected.tolist() == [2, 2]
    assert mask[torch.arange(mask.shape[0]), selected].tolist() == [1.0, 1.0]
