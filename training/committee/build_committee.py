"""Assemble the ten trained members into one committee folder and write ``committee.json``.

    python -m committee.build_committee --out-dir "$OUT_DIR" --committee-dir "$OUT_DIR/committee"

Each member contributes one probability per online point, p_S = the mean of
its per-step CDF maps evaluated on its raw score(s) (map q fed by model
``model_index``).  Fusion in log-odds space:

    l_S   = log(p / (1 - p)),  p = clip(p_S, 1e-6, 1 - 1e-6)            (float64)
    votes = [l_S] * multiplicity_S, members in the order below
    acc   = votes[0]; acc += votes[i] for i = 1..26  (left fold);  m = acc / 27
    score = clip(sigmoid(m), 1e-9, 1 - 1e-9), cast to float32 then back to float64

Votes (N = 27): surprises 1, pattern_removed 1, order_not_size 1, slow_and_fast 1,
reads_two_rebuilds 1, float_grid 16, reads_three_others 2, broken_or_not 1,
which_one_broke 2, blind_to_size 1.

The script copies the artifacts produced by the training steps from
``--out-dir`` (layout of the training README), writes ``committee.json`` and a
``SHA256SUMS`` file over every copied file.  ``--check-only`` verifies the
sources without copying.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

SEEDS = (0, 1, 2)
NAMES = ['surprises', 'pattern_removed', 'order_not_size', 'slow_and_fast', 'reads_two_rebuilds', 'float_grid',
         'reads_three_others', 'broken_or_not', 'which_one_broke', 'blind_to_size']
NETWORKS = ('broken_or_not', 'which_one_broke', 'blind_to_size')


def _bag(name: str, bag_mode: str, mult: int, model_seeds, map_index) -> dict:
    return dict(name=name, kind=name, bag_mode=bag_mode, multiplicity=mult,
                models=[dict(seed=k, dir=f'model_full_{name}_s{k}') for k in model_seeds],
                maps=[dict(seed=k, path=f'maps_{name}_train10k_s{k}.npz', model_index=map_index(k)) for k in SEEDS])


MEMBERS = [
    _bag('surprises', 'shared_mean_of_mapped', 1, SEEDS, lambda k: k),
    _bag('pattern_removed', 'shared_mean_of_mapped', 1, SEEDS, lambda k: k),
    _bag('order_not_size', 'one_scorer_k_maps', 1, (0,), lambda k: 0),
    _bag('slow_and_fast', 'mean_of_mapped', 1, SEEDS, lambda k: k),
    dict(name='reads_two_rebuilds', kind='reads_two_rebuilds', bag_mode='one_scorer_k_maps', multiplicity=1,
         models=[dict(seed=0, dir='model_full_reads_two_rebuilds_s0')],
         maps=[dict(seed=0, path='maps_reads_two_rebuilds_train10k_s0.npz', model_index=0)],
         feature_inputs=['surprises', 'pattern_removed']),
    dict(name='float_grid', kind='float_grid', bag_mode='one_scorer_k_maps', multiplicity=16,
         models=[dict(seed=0, dir='model_full_float_grid_s0')],
         maps=[dict(seed=0, path='maps_float_grid_train10k_s0.npz', model_index=0)]),
    dict(name='reads_three_others', kind='reads_three_others', bag_mode='one_scorer_k_maps', multiplicity=2,
         models=[dict(seed=0, dir='model_reads_three_others')],
         maps=[dict(seed=0, path='map_reads_three_others.npz', model_index=0)],
         feature_inputs=['surprises', 'pattern_removed']),
] + [dict(name=n, kind=n, bag_mode='one_scorer_k_maps', multiplicity=m, models=[dict(seed=0, dir=n)],
          maps=[dict(seed=0, path=f'{n}_map.npz', model_index=0)])
      for n, m in zip(NETWORKS, (1, 2, 1), strict=True)]

KINDS = {
    'surprises': dict(shared_features='surprises_stream_step'),
    'pattern_removed': dict(shared_features='pattern_removed_feature_state'),
    'order_not_size': {}, 'slow_and_fast': {}, 'reads_two_rebuilds': dict(feature_head=True), 'float_grid': {},
    'reads_three_others': dict(feature_head=True), 'broken_or_not': {}, 'which_one_broke': {}, 'blind_to_size': {},
}

# Where each committee entry comes from, relative to --out-dir (training layout).
SOURCES = {
    **{f'model_full_{n}_s{k}': f'{n}/model_full_{n}_s{k}' for n in ('surprises', 'pattern_removed', 'slow_and_fast')
       for k in SEEDS},
    **{f'maps_{n}_train10k_s{k}.npz': f'{n}/maps_{n}_train10k_s{k}.npz'
       for n in ('surprises', 'pattern_removed', 'order_not_size', 'slow_and_fast') for k in SEEDS},
    'model_full_order_not_size_s0': 'order_not_size/model_full_order_not_size_s0',
    'model_full_reads_two_rebuilds_s0': 'reads_two_rebuilds/model_full_reads_two_rebuilds_s0',
    'maps_reads_two_rebuilds_train10k_s0.npz': 'reads_two_rebuilds/maps_reads_two_rebuilds_train10k_s0.npz',
    'model_full_float_grid_s0': 'float_grid/model_full_float_grid_s0',
    'maps_float_grid_train10k_s0.npz': 'float_grid/maps_float_grid_train10k_s0.npz',
    'model_reads_three_others': 'reads_three_others/model_reads_three_others',
    'map_reads_three_others.npz': 'reads_three_others/map_reads_three_others.npz',
    **{f'{n}_map.npz': f'{n}/{n}_map.npz' for n in NETWORKS},
    **{f'{n}/{f}': f'{n}/{f}' for n in NETWORKS for f in ('member.npz', 'fallback_raw.npy', 'member_config.json')},
}


def committee_json() -> dict:
    n_votes = sum(m['multiplicity'] for m in MEMBERS)
    assert [m['name'] for m in MEMBERS] == NAMES
    assert [m['multiplicity'] for m in MEMBERS] == [1, 1, 1, 1, 1, 16, 2, 1, 2, 1] and n_votes == 27
    return dict(
        schema='committee/v2', final_reduction='left_fold',
        fusion=dict(space='log_odds', member_clip=[1e-6, 1 - 1e-6], final_clip=[1e-9, 1 - 1e-9], n_votes=n_votes,
                    output='float32 then float64'),
        semantics=('member value p = sum(map_q.evaluate(t, raw[model_index_q]) for q in maps order) / len(maps) '
                   '(builtin sum); member logit = log(p / (1 - p)) after clipping p to [1e-6, 1 - 1e-6]; final = '
                   'left fold over [logit_S] * multiplicity_S in members order, divided by the number of votes, '
                   'sigmoid, clip [1e-9, 1 - 1e-9]; reads_two_rebuilds raw = mean of its 3 bag logits; t 0-based, '
                   'maps clamp t at 999'),
        kinds=KINDS, members=MEMBERS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', type=Path, default=os.environ.get('OUT_DIR'), help='training outputs')
    ap.add_argument('--committee-dir', type=Path, required=True)
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if a.out_dir is None:
        ap.error('set --out-dir or OUT_DIR')
    missing = [src for src in SOURCES.values() if not (Path(a.out_dir) / src).exists()]
    if missing:
        raise SystemExit('missing training outputs:\n  ' + '\n  '.join(missing))
    if a.check_only:
        print(f'all {len(SOURCES)} sources present')
        return
    dst_root = a.committee_dir
    dst_root.mkdir(parents=True, exist_ok=True)
    for dst, src in SOURCES.items():
        s, d = Path(a.out_dir) / src, dst_root / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copyfile(s, d)
    (dst_root / 'committee.json').write_text(json.dumps(committee_json(), indent=2) + '\n', encoding='utf-8')
    files = sorted(p for p in dst_root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (dst_root / 'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(dst_root).as_posix()}\n'
                                                 for p in files), encoding='utf-8')
    print(f'committee with {len(MEMBERS)} members ({sum(m["multiplicity"] for m in MEMBERS)} votes) in {dst_root}')


if __name__ == '__main__':
    main()
