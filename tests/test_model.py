import torch

from nanogo.go.features import NUM_GLOBAL, NUM_SPATIAL
from nanogo.net.losses import LossWeights, compute_losses
from nanogo.net.model import Model, ModelConfig


def _synthetic_batch(B, P, device):
    g = torch.Generator().manual_seed(0)
    spatial = torch.zeros(B, NUM_SPATIAL, P, P)
    spatial[:, 0] = 1.0  # on-board
    spatial[:, 1] = (torch.rand(B, P, P, generator=g) < 0.2).float()
    glob = torch.randn(B, NUM_GLOBAL, generator=g)
    policy = torch.zeros(B, 1, P * P + 1)
    idx = torch.randint(0, P * P + 1, (B,), generator=g)
    for i in range(B):
        policy[i, 0, idx[i]] = 1.0
    gt = torch.zeros(B, 41)
    gt[:, 0] = 1.0  # value target = win
    gt[:, 3] = 5.0  # score
    gt[:, 27] = 1.0  # ownership weight
    own = (torch.rand(B, 1, P, P, generator=g) * 2 - 1)
    batch = {
        "spatial": spatial, "glob": glob, "policyTargetsNCMove": policy,
        "globalTargetsNC": gt, "valueTargetsNCHW": own,
    }
    return {k: v.to(device) for k, v in batch.items()}


def test_forward_shapes():
    P = 19
    model = Model(ModelConfig.plain(16, 2, pos_len=P))
    batch = _synthetic_batch(4, P, "cpu")
    policy, value, score, ownership = model(batch["spatial"], batch["glob"])
    assert policy.shape == (4, P * P + 1)
    assert value.shape == (4, 3)
    assert score.shape == (4,)
    assert ownership.shape == (4, 1, P, P)


def test_policy_loss_ignores_offboard():
    # On a 9x9 board placed in a 19x19 tensor, off-board policy logits must not affect the loss.
    P, B = 19, 2
    spatial = torch.zeros(B, NUM_SPATIAL, P, P)
    spatial[:, 0, :9, :9] = 1.0  # 9x9 on-board mask
    glob = torch.zeros(B, NUM_GLOBAL)
    policy_t = torch.zeros(B, 1, P * P + 1)
    policy_t[:, 0, 0] = 1.0  # target = an on-board move
    gt = torch.zeros(B, 41)
    gt[:, 27] = 1.0
    own = torch.zeros(B, 1, P, P)
    batch = {"spatial": spatial, "glob": glob, "policyTargetsNCMove": policy_t,
             "globalTargetsNC": gt, "valueTargetsNCHW": own}
    weights = LossWeights()
    zeros = (torch.zeros(B, P * P + 1), torch.zeros(B, 3), torch.zeros(B), torch.zeros(B, 1, P, P))
    legal = torch.cat([spatial[:, 0].reshape(B, -1), torch.ones(B, 1)], dim=1)
    big_offboard = torch.where(legal == 0, torch.full((B, P * P + 1), 50.0),
                               torch.zeros(B, P * P + 1))
    out2 = (big_offboard, torch.zeros(B, 3), torch.zeros(B), torch.zeros(B, 1, P, P))
    _, p1 = compute_losses(zeros, batch, spatial, weights)
    _, p2 = compute_losses(out2, batch, spatial, weights)
    assert abs(p1["policy"] - p2["policy"]) < 1e-4  # off-board logits masked out


def test_overfit_one_batch():
    P = 9
    model = Model(ModelConfig.plain(16, 2, pos_len=P))
    batch = _synthetic_batch(8, P, "cpu")
    weights = LossWeights()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.train()
    first = None
    for _ in range(60):
        out = model(batch["spatial"], batch["glob"])
        loss, _ = compute_losses(out, batch, batch["spatial"], weights)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5  # loss should drop substantially
