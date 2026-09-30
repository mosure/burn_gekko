#!/usr/bin/env python3
"""Account a completed, separately reserved study leg; overlap is counted in full."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from run_study import atomic_json


def account(budget_path, leg_path):
    budget_path, leg_path = Path(budget_path).resolve(), Path(leg_path).resolve()
    root = Path('.data').resolve()
    if not budget_path.is_relative_to(root) or not leg_path.is_relative_to(root):
        raise ValueError('both ledgers must be under .data')
    with budget_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = json.loads(budget_path.read_text())
        raw = leg_path.read_bytes(); leg = json.loads(raw)
        if budget.get('running') or not leg.get('finished_utc'):
            raise ValueError('both study legs must have finished')
        if any(c['status'] == 'running' for c in leg['commands']):
            raise ValueError('external command is still marked running')
        if str(leg_path) in budget['legs']:
            raise ValueError('leg is already accounted')
        elapsed = sum(c['elapsed_seconds'] for c in leg['commands'])
        if abs(elapsed-leg['command_seconds']) > 1e-6 or elapsed < 0:
            raise ValueError('external ledger arithmetic mismatch')
        total = budget['command_seconds'] + elapsed
        if not total <= budget['ceiling_seconds'] <= 43200:
            raise ValueError('study ceiling would be exceeded')
        budget['command_seconds'] = total
        budget['legs'].append(str(leg_path))
        budget.setdefault('external_leg_receipts', []).append(dict(
            path=str(leg_path), sha256=hashlib.sha256(raw).hexdigest(), command_seconds=elapsed,
            overlap_policy='entire command wall duration counted even if capture overlapped'))
        atomic_json(budget_path, budget)
        return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-ledger', type=Path, required=True)
    parser.add_argument('--external-ledger', type=Path, required=True)
    args = parser.parse_args()
    print(f'Accounted cumulative command seconds: {account(args.budget_ledger, args.external_ledger):.3f}')


if __name__ == '__main__':
    main()
