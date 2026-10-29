"""Export a trained state (``net_ft.pt``) to the flat float64 weight file read by the numpy/numba member.

    python -m networks.export_weights "$OUT_DIR/broken_or_not/full/net_ft.pt" "$OUT_DIR/broken_or_not/member.npz"

``member.npz`` holds ``W`` (all parameters, float64, concatenated in ``KEYS``
order, C order), ``OFF`` (int64 offset of each parameter in ``W``) and ``H``
(GRU hidden size).  The same export serves broken_or_not, which_one_broke and
blind_to_size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KEYS = ['gru.weight_ih_l0', 'gru.weight_hh_l0', 'gru.bias_ih_l0', 'gru.bias_hh_l0',
        'gru.weight_ih_l1', 'gru.weight_hh_l1', 'gru.bias_ih_l1', 'gru.bias_hh_l1',
        'emb.weight', 'emb.bias', 'head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias',
        'dnorm.weight', 'dnorm.bias', 'dhead.0.weight', 'dhead.0.bias', 'dhead.2.weight', 'dhead.2.bias']


def export_flat(state_dict, out_npz):
    """Torch tensors in, float64 flat vector out."""
    parts, off, pos = [], [], 0
    for k in KEYS:
        a = state_dict[k].detach().cpu().double().numpy().ravel(order='C')
        parts.append(a)
        off.append(pos)
        pos += a.size
    H = int(state_dict['gru.weight_hh_l0'].shape[1])
    np.savez(out_npz, W=np.concatenate(parts), OFF=np.array(off, np.int64), H=np.array([H], np.int64))
    return out_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import torch

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_flat(state, args.output)


if __name__ == "__main__":
    main()
