#!/usr/bin/env python3
"""Review a bounded Nsight training replay using its distinctive RGB transfers."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import tomllib


def union_seconds(rows, start, end):
    intervals = []
    for a, b in sorted(rows):
        a, b = max(a, start), min(b, end)
        if a >= b:
            continue
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], b)
        else:
            intervals.append([a, b])
    return sum(b-a for a, b in intervals) / 1e9


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = tomllib.loads(args.config.read_text())
    output = Path(c['output'])
    assert output.resolve().is_relative_to(Path('.data').resolve()) and not output.exists()
    rows = lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()]
    replay = rows(c['replay_metrics'])
    original = {r['step']: r for r in rows(c['original_metrics'])}
    assert len(replay) == c['updates']
    for r in replay:
        old = original[r['step']]
        assert r['samples'] == old['samples'] and r['stage'] == old['stage'] == 2
        assert r['learning_rate'] == old['learning_rate']
    db = sqlite3.connect('file:' + str(Path(c['sqlite']).resolve()) + '?mode=ro', uri=True)
    size = c['batch_size'] * c['image_size']**2 * 3 * 4
    rgb = list(db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE bytes=? AND copyKind=1 ORDER BY start', (size,)))
    transfers = c['references'] + 1
    assert len(rgb) == len(replay) * transfers, 'RGB-transfer count does not match replay'
    first, last = c['first_update_index'], c['end_update_index']
    assert 0 <= first < last < len(replay)
    start, end = rgb[first * transfers][0], rgb[last * transfers][0]
    duration = (end-start)/1e9
    kernels = list(db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start<? AND end>?', (end, start)))
    copies = list(db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>?', (end, start)))
    rgb_seconds = sum(b-a for a,b in rgb if start <= a < end)/1e9
    kernel_seconds = union_seconds(kernels, start, end)
    event_seconds = union_seconds(kernels + copies, start, end)
    api = list(db.execute('SELECT s.value,SUM(r.end-r.start)/1e9,COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE r.start>=? AND r.start<? GROUP BY s.value ORDER BY SUM(r.end-r.start) DESC LIMIT 12', (start, end)))
    result = dict(scope='One profiled process. Kernel union is active kernel time, not SM occupancy. Gaps may include host work, dispatch, synchronization or untraced GPU activity. Cold start and final evaluation are excluded from the selected window.',
                  sqlite_sha256=digest(c['sqlite']), config_sha256=digest(args.config),
                  replay_metrics_sha256=digest(c['replay_metrics']), original_metrics_sha256=digest(c['original_metrics']),
                  replay_max_total_loss_delta=max(abs(r['total']-original[r['step']]['total']) for r in replay),
                  replay_max_gradient_norm_delta=max(abs(r['gradient_norm']-original[r['step']]['gradient_norm']) for r in replay),
                  selected_updates=last-first, first_step=replay[first]['step'], end_step_exclusive=replay[last]['step'],
                  window_start_seconds=start/1e9, window_end_seconds=end/1e9, window_seconds=duration,
                  kernel_count=len(kernels), kernel_union_seconds=kernel_seconds,
                  kernel_time_fraction=kernel_seconds/duration, device_event_time_fraction=event_seconds/duration,
                  unattributed_gap_seconds=duration-event_seconds, rgb_transfer_seconds=rgb_seconds,
                  rgb_transfer_seconds_per_update=rgb_seconds/(last-first),
                  profiled_median_update_seconds=statistics.median(r['seconds'] for r in replay[first:last]),
                  original_median_update_seconds=statistics.median(original[r['step']]['seconds'] for r in replay[first:last]),
                  top_cuda_apis=[dict(name=n,total_seconds=t,calls=count) for n,t,count in api])
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
