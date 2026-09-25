import argparse
import json
import tempfile
import unittest
from pathlib import Path

import runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task_dir = self.root / "task"
        starter = self.task_dir / "starter"
        starter.mkdir(parents=True)
        (starter / "kernel.hip").write_text("// initial\n", encoding="utf-8")
        (self.task_dir / "prompt.md").write_text("Implement a HIP kernel.\n", encoding="utf-8")
        self.task = self.task_dir / "task.json"
        self.task.write_text(json.dumps({
            "task_id": "fixture", "task_revision": "v1", "aiter_sha": "a" * 40,
            "gpu_sku": "MI350X", "target_arch": "gfx950",
            "operator_entry": "example", "prompt_file": "prompt.md",
            "starter_dir": "starter", "harness_revision": "b" * 40,
            "scored_eligible": True,
        }), encoding="utf-8")
        self.freeze = self.root / "freeze.json"
        self.freeze.write_text(json.dumps(runner.freeze_payload(self.task)), encoding="utf-8")
        self.fake_cli = self.root / "fake-codex"
        self.fake_cli.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "if '--version' in sys.argv:\n"
            "    print('fake-codex 1.0')\n"
            "else:\n"
            "    sys.stdin.read()\n"
            "    pathlib.Path('kernel.hip').write_text('// candidate\\n')\n"
            "    print(json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution'}}), flush=True)\n",
            encoding="utf-8",
        )
        self.fake_cli.chmod(0o755)

    def tearDown(self):
        self.temp.cleanup()

    def args(self, dry_run=False):
        return argparse.Namespace(task=self.task, freeze=self.freeze,
            results=self.root / "results", replicate_id="r001", model="fake-model",
            cli=str(self.fake_cli), wall_seconds=5, reasoning_effort="medium", dry_run=dry_run)

    def test_freeze_rejects_changed_starter(self):
        (self.task_dir / "starter" / "kernel.hip").write_text("// changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed since freeze"):
            runner.run_agent(self.args())

    def test_dry_run_does_not_create_artifacts(self):
        self.assertEqual(runner.run_agent(self.args(dry_run=True)), 0)
        self.assertFalse((self.root / "results").exists())

    def test_fake_agent_captures_source_and_event(self):
        self.assertEqual(runner.run_agent(self.args()), 0)
        result = self.root / "results" / "fixture--v1--r001"
        snapshots = [json.loads(line) for line in (result / "snapshots.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(snapshots), 2)
        self.assertNotEqual(snapshots[0]["tree_sha256"], snapshots[-1]["tree_sha256"])
        self.assertEqual((result / "workspace" / "kernel.hip").read_text(), "// candidate\n")
        self.assertEqual(len((result / "events.jsonl").read_text().splitlines()), 1)
        self.assertEqual(json.loads((result / "result.json").read_text())["scored"], False)
        self.assertTrue((result / snapshots[-1]["diff"]).exists())

    def test_task_preview_cannot_launch_agent(self):
        task = json.loads(self.task.read_text())
        task["scored_eligible"] = False
        self.task.write_text(json.dumps(task), encoding="utf-8")
        self.freeze.write_text(json.dumps(runner.freeze_payload(self.task)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "scored_eligible"):
            runner.run_agent(self.args())

    def test_timeout_keeps_failure_record(self):
        self.fake_cli.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys, time\n"
            "if '--version' in sys.argv: print('fake-codex 1.0')\n"
            "else:\n"
            "    sys.stdin.read()\n"
            "    pathlib.Path('kernel.hip').write_text('// before timeout\\n')\n"
            "    time.sleep(10)\n",
            encoding="utf-8",
        )
        args = self.args()
        args.wall_seconds = 1
        self.assertEqual(runner.run_agent(args), 1)
        result = self.root / "results" / "fixture--v1--r001"
        status = json.loads((result / "result.json").read_text())
        self.assertEqual(status["status"], "timeout")
        self.assertGreaterEqual(status["snapshot_count"], 2)


if __name__ == "__main__":
    unittest.main()
