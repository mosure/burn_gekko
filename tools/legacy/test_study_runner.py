"""Integration checks for the persistent experiment ceiling and failure handling."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_study
from account_study_leg import account


class StudyRunnerTests(unittest.TestCase):
    def setUp(self):
        root = Path('.data/tests'); root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_plan(self, code, name, ledger):
        plan = self.root / (name + '.toml')
        plan.write_text('max_command_seconds = 30\n[[commands]]\nname = "probe"\nmax_seconds = 2\n'
                        + 'argv = ' + json.dumps([sys.executable, '-c', code]) + '\n')
        argv = ['run_study.py', '--plan', str(plan), '--output', str(self.root / name), '--budget-ledger', str(ledger)]
        with patch.object(sys, 'argv', argv), patch.object(run_study, 'telemetry', return_value={}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_study.main()

    def test_legs_accumulate_and_failed_command_does_not_lose_accounting(self):
        ledger = self.root / 'budget.json'
        self.run_plan('pass', 'first', ledger)
        first = json.loads(ledger.read_text())['command_seconds']
        with self.assertRaises(SystemExit):
            self.run_plan('raise SystemExit(3)', 'failed', ledger)
        result = json.loads(ledger.read_text())
        self.assertGreater(result['command_seconds'], first)
        self.assertEqual(len(result['legs']), 2)
        self.assertNotIn('running', result)
        self.assertEqual(json.loads((self.root / 'failed/ledger.json').read_text())['commands'][0]['status'], 'failed')

    def test_incomplete_or_exhausted_or_expanded_budget_is_rejected(self):
        ledger = self.root / 'budget.json'
        for i, budget in enumerate([
            dict(ceiling_seconds=43200, command_seconds=0, running={'pid': 123}),
            dict(ceiling_seconds=43200, command_seconds=43200),
            dict(ceiling_seconds=43201, command_seconds=0),
        ]):
            ledger.write_text(json.dumps(budget))
            with self.assertRaises(SystemExit):
                self.run_plan('pass', f'rejected-{i}', ledger)
            self.assertFalse((self.root / f'rejected-{i}').exists())

    def test_external_leg_is_counted_once_and_reservation_blocks_new_work(self):
        budget = self.root/'budget.json'
        external_budget = self.root/'external-budget.json'
        self.run_plan('pass', 'external', external_budget)
        external = self.root/'external/ledger.json'
        budget.write_text(json.dumps(dict(ceiling_seconds=43200, command_seconds=5, legs=[])))
        budget.with_suffix('.reservations.toml').write_text('ledgers = ' + json.dumps([str(external)]) + '\n')
        with self.assertRaises(SystemExit):
            self.run_plan('pass', 'not-accounted', budget)
        elapsed = json.loads(external.read_text())['command_seconds']
        self.assertAlmostEqual(account(budget, external), 5+elapsed)
        with self.assertRaises(ValueError):
            account(budget, external)
        self.run_plan('pass', 'after-accounting', budget)


if __name__ == '__main__':
    unittest.main()
