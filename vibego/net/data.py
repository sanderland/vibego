"""Stream training batches from katagoarchive .npz files.

Each .npz holds many positions. We unpack the packed binary features, select our
channel subset (vibego.features), pull the targets we train on, and optionally apply
one of the 8 board symmetries per batch. Modeled on KataGo's
data_processing_pytorch.read_npz_training_data but trimmed to our subset.
"""
from __future__ import annotations

import glob as _glob
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ..go import features as F


def list_npz(data_dir: str) -> list[str]:
    return sorted(_glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))


def _apply_symmetry(t: torch.Tensor, symm: int) -> torch.Tensor:
    """t: (..., P, P). symm 0-3 rotations, 4-7 add a transpose. Matches KataGo."""
    if symm == 0:
        return t
    if symm == 1:
        return t.transpose(-2, -1).flip(-2)
    if symm == 2:
        return t.flip(-1).flip(-2)
    if symm == 3:
        return t.transpose(-2, -1).flip(-1)
    if symm == 4:
        return t.transpose(-2, -1)
    if symm == 5:
        return t.flip(-1)
    if symm == 6:
        return t.transpose(-2, -1).flip(-1).flip(-2)
    return t.flip(-2)


def _apply_symmetry_policy(t: torch.Tensor, symm: int, pos_len: int) -> torch.Tensor:
    """t: (B, C, P*P+1) with a trailing pass entry."""
    b, c = t.shape[0], t.shape[1]
    board = t[:, :, :-1].reshape(b, c, pos_len, pos_len)
    board = _apply_symmetry(board, symm)
    return torch.cat([board.reshape(b, c, pos_len * pos_len), t[:, :, -1:]], dim=2)


def _load(path: str, pos_len: int):
    with np.load(path) as npz:
        spatial, glob = F.decode_npz(
            npz["binaryInputNCHWPacked"], npz["globalInputNC"], pos_len
        )
        policy = npz["policyTargetsNCMove"][:, 0:1, :].astype(np.float32)
        global_targets = npz["globalTargetsNC"].astype(np.float32)
        ownership = npz["valueTargetsNCHW"][:, 0:1, :, :].astype(np.float32)
    return spatial, glob, policy, global_targets, ownership


def _prefetch(files, pos_len, workers=6, ahead=128):
    """Load npz files with bounded parallel prefetch, preserving order."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = iter(files)
        pending = deque()
        for _ in range(ahead):
            try:
                pending.append(ex.submit(_load, next(it), pos_len))
            except StopIteration:
                break
        while pending:
            res = pending.popleft().result()
            try:
                pending.append(ex.submit(_load, next(it), pos_len))
            except StopIteration:
                pass
            yield res


def read_batches(
    npz_files: list[str],
    batch_size: int,
    pos_len: int,
    device,
    randomize_symmetries: bool = True,
    seed: int = 0,
    shuffle_buffer: int = 20000,
    drop_last: bool = True,
    prefetch_ahead: int = 128,
    only_full_board: bool = False,
):
    """Stream batches with a shuffle buffer (archive npz files hold only ~25 rows each, and
    rows within a file are correlated, so we accumulate and shuffle before batching).

    Yields dicts: spatial, glob, policyTargetsNCMove, globalTargetsNC, valueTargetsNCHW.
    """
    if not npz_files:
        return
    rng = np.random.default_rng(seed)
    cols = ["spatial", "glob", "policy", "gt", "own"]
    pending = {c: [] for c in cols}
    count = 0

    def emit(arrays, full_only):
        nonlocal pending, count
        n = arrays["spatial"].shape[0]
        perm = rng.permutation(n)
        limit = (n // batch_size) * batch_size if full_only else n
        for start in range(0, limit, batch_size):
            end = min(start + batch_size, n)
            if full_only and end - start < batch_size:
                break
            idx = perm[start:end]
            yield _make_batch({c: arrays[c][idx] for c in arrays}, pos_len, device,
                              randomize_symmetries, rng)
        # carry over leftover (unbatched) rows
        leftover = perm[limit:]
        for c in cols:
            pending[c] = [arrays[c][leftover]] if len(leftover) else []
        count = len(leftover)

    for spatial, glob, policy, gt, own in _prefetch(npz_files, pos_len, ahead=prefetch_ahead):
        if only_full_board:  # keep only full pos_len×pos_len boards (our 19×19 play setup)
            keep = spatial[:, 0].sum(axis=(1, 2)) == pos_len * pos_len
            spatial, glob, policy, gt, own = (a[keep] for a in (spatial, glob, policy, gt, own))
            if spatial.shape[0] == 0:
                continue
        for c, a in zip(cols, (spatial, glob, policy, gt, own)):
            pending[c].append(a)
        count += spatial.shape[0]
        if count >= shuffle_buffer:
            arrays = {c: np.concatenate(pending[c], axis=0) for c in cols}
            yield from emit(arrays, full_only=True)

    if count >= batch_size or (count > 0 and not drop_last):
        arrays = {c: np.concatenate(pending[c], axis=0) for c in cols}
        yield from emit(arrays, full_only=drop_last)


def _make_batch(arrays, pos_len, device, randomize_symmetries, rng):
    b_spatial = torch.from_numpy(np.ascontiguousarray(arrays["spatial"])).to(device)
    b_glob = torch.from_numpy(np.ascontiguousarray(arrays["glob"])).to(device)
    b_policy = torch.from_numpy(np.ascontiguousarray(arrays["policy"])).to(device)
    b_gt = torch.from_numpy(np.ascontiguousarray(arrays["gt"])).to(device)
    b_own = torch.from_numpy(np.ascontiguousarray(arrays["own"])).to(device)
    if randomize_symmetries:
        symm = int(rng.integers(0, 8))
        b_spatial = _apply_symmetry(b_spatial, symm)
        b_own = _apply_symmetry(b_own, symm)
        b_policy = _apply_symmetry_policy(b_policy, symm, pos_len)
    return {
        "spatial": b_spatial.contiguous(),
        "glob": b_glob,
        "policyTargetsNCMove": b_policy.contiguous(),
        "globalTargetsNC": b_gt,
        "valueTargetsNCHW": b_own.contiguous(),
    }
