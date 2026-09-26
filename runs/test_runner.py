import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            "harness_spec_sha256": "c" * 64, "task_mode": "no_feedback",
            "build_sources": ["kernel.hip"], "build_flags": ["-O3", "-shared", "-fPIC", "--offload-arch=gfx950"],
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
            cli=str(self.fake_cli), wall_seconds=5, reasoning_effort="medium", dry_run=dry_run,
            unscored_preview=False)

    def test_freeze_rejects_changed_starter(self):
        (self.task_dir / "starter" / "kernel.hip").write_text("// changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed since freeze"):
            runner.run_agent(self.args())

    def test_documented_script_entrypoint_freezes_task(self):
        output = self.root / "script-freeze.json"
        subprocess.run(
            [sys.executable, str(Path(runner.__file__)), "freeze", str(self.task), str(output)],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(output.read_text()), runner.freeze_payload(self.task))

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

    def test_unscored_preview_runs_but_cannot_score(self):
        task = json.loads(self.task.read_text())
        task["scored_eligible"] = False
        task["harness_revision"] = "UNSET-DRY-RUN"
        task.pop("harness_spec_sha256")
        self.task.write_text(json.dumps(task), encoding="utf-8")
        self.freeze.write_text(json.dumps(runner.freeze_payload(self.task)), encoding="utf-8")
        args = self.args()
        args.unscored_preview = True
        self.assertEqual(runner.run_agent(args), 0)
        run_dir = self.root / "results" / "fixture--v1--r001"
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["run_purpose"], "unscored_preview")
        self.assertFalse(manifest["incidence_eligible"])
        self.assertGreaterEqual(len((run_dir / "snapshots.jsonl").read_text().splitlines()), 2)
        with self.assertRaisesRegex(ValueError, "cannot enter the scored harness"):
            runner.score_snapshots(argparse.Namespace(task=self.task, freeze=self.freeze, run_dir=run_dir))

    def test_agent_namespace_hides_private_paths_and_host_credentials(self):
        private = self.root / "private" / "withheld.json"
        private.parent.mkdir()
        private.write_text('{"secret":"test-only"}', encoding="utf-8")
        (self.task_dir / "starter" / "kernel.hip").write_text(
            '#include <hip/hip_runtime.h>\nextern "C" __global__ void noop() {}\n', encoding="utf-8"
        )
        self.freeze.write_text(json.dumps(runner.freeze_payload(self.task)), encoding="utf-8")
        self.fake_cli.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, subprocess, sys\n"
            "if '--version' in sys.argv: print('fake-codex 1.0')\n"
            "else:\n"
            "    sys.stdin.read()\n"
            "    data={'private_visible':pathlib.Path(" + repr(str(private)) + ").exists(),"
            "'ssh_visible':pathlib.Path('/home/harsh/.ssh').exists(),"
            "'repo_visible':pathlib.Path('/home/harsh/aiter-rs').exists(),"
            "'home_entries':sorted(p.name for p in pathlib.Path('/home').iterdir()),"
            "'ssh_agent_env':'SSH_AUTH_SOCK' in os.environ,"
            "'cloud_secret_env':'AWS_SECRET_ACCESS_KEY' in os.environ,"
            "'github_token_env':'GITHUB_TOKEN' in os.environ,"
            "'auth_present':pathlib.Path('/home/agent/.codex/auth.json').is_file(),"
            "'auth_readable':bool(pathlib.Path('/home/agent/.codex/auth.json').open('rb').read(1)),"
            "'hipcc_works':subprocess.run(['/usr/bin/hipcc','-O2','-shared','-fPIC','--offload-arch=gfx950','kernel.hip','-o','libcandidate.so'],capture_output=True,timeout=60).returncode==0}\n"
            "    pathlib.Path('/home/agent/.codex/ephemeral_marker').write_text('inside only')\n"
            "    data['codex_home_writable']=pathlib.Path('/home/agent/.codex/ephemeral_marker').is_file()\n"
            "    pathlib.Path('isolation.json').write_text(json.dumps(data))\n"
            "    pathlib.Path('kernel.hip').write_text('// isolated candidate\\n')\n"
            "    print(json.dumps({'type':'item.completed'}),flush=True)\n",
            encoding="utf-8",
        )
        args = self.args()
        args.wall_seconds = 60  # HIP startup and compilation vary under shared-host load.
        with patch.dict(os.environ, {"SSH_AUTH_SOCK": "/tmp/host-ssh-agent.sock", "AWS_SECRET_ACCESS_KEY": "test-only", "GITHUB_TOKEN": "test-only"}):
            self.assertEqual(runner.run_agent(args), 0)
        run_dir = self.root / "results" / "fixture--v1--r001"
        observed = json.loads((run_dir / "workspace" / "isolation.json").read_text())
        self.assertEqual(observed, {
            "private_visible": False, "ssh_visible": False, "repo_visible": False,
            "home_entries": ["agent"], "ssh_agent_env": False,
            "cloud_secret_env": False, "github_token_env": False,
            "auth_present": True, "auth_readable": True,
            "hipcc_works": True, "codex_home_writable": True,
        })
        self.assertTrue((run_dir / "workspace" / "libcandidate.so").is_file())
        self.assertNotIn("host-ssh-agent.sock", (run_dir / "manifest.json").read_text())
        self.assertFalse((Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "ephemeral_marker").exists())

    def test_bwrap_is_required_without_unsandboxed_fallback(self):
        with patch("runner.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "bwrap is required"):
                runner.agent_sandbox_command(str(self.fake_cli), self.root / "workspace", "fake-model", "medium")

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

    def make_git_repo(self, path):
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "add", "."], check=True)
        subprocess.run(["git", "-C", str(path), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"], check=True)
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

    def setup_scorer(self):
        aiter_root = self.root / "aiter"
        aiter_root.mkdir()
        (aiter_root / "README.md").write_text("fixture\n", encoding="utf-8")
        aiter_sha = self.make_git_repo(aiter_root)
        private = self.root / "private"
        private.mkdir()
        self.withheld = private / "withheld.json"
        self.withheld.write_text(json.dumps({
            "schema_version": 1, "task_id": "fixture",
            "cases": [{"id": "secret_case", "visibility": "withheld"}],
        }), encoding="utf-8")
        self.host_report = private / "host-gpu-report.json"
        self.host_report.write_text(json.dumps({
            "schema_version": 1, "gpu_name": "MI350X", "arch": "gfx950", "card_model": "0x75a0",
        }), encoding="utf-8")
        harness_root = self.root / "trusted-harness"
        (harness_root / "references").mkdir(parents=True)
        (harness_root / "harness").mkdir()
        spec = harness_root / "references" / "spec.json"
        spec.write_text(json.dumps({
            "task_id": "fixture", "aiter_sha": aiter_sha, "gpu_sku": "MI350X",
            "gpu_pci_device_id": "0x75a0", "target_arch": "gfx950",
            "withheld_cases_sha256": hashlib.sha256(self.withheld.read_bytes()).hexdigest(),
        }), encoding="utf-8")
        (harness_root / "harness" / "__init__.py").write_text("", encoding="utf-8")
        (harness_root / "harness" / "run.py").write_text(
            "import argparse, hashlib, json, pathlib, subprocess, sys\n"
            "p=argparse.ArgumentParser()\n"
            "for flag in ('spec','candidate','aiter-source','output','withheld-spec','host-gpu-report','task-freeze-sha256','final-tree-sha256'): p.add_argument('--'+flag)\n"
            "p.add_argument('--correctness-only',action='store_true')\n"
            "a=p.parse_args()\n"
            "out=pathlib.Path(a.output); out.mkdir()\n"
            "spec=json.loads(pathlib.Path(a.spec).read_text())\n"
            "def path_hash(path):\n"
            "    p=pathlib.Path(path); h=hashlib.sha256(); h.update(p.name.encode()+b'\\0'); h.update(p.read_bytes()); h.update(b'\\0'); return h.hexdigest()\n"
            "result={'task_id':'fixture','aiter_sha':subprocess.check_output(['git','-C',a.aiter_source,'rev-parse','HEAD'],text=True).strip(),"
            "'spec_sha256':path_hash(a.spec),"
            "'candidate_path_sha256':path_hash(a.candidate),"
            "'task_freeze_sha256':a.task_freeze_sha256,'final_tree_sha256':a.final_tree_sha256,"
            "'withheld_cases_sha256':spec['withheld_cases_sha256'],'withheld_cases_evaluated':True,"
            "'environment':{'gpu_sku':'MI350X','host_gpu_report_sha256':hashlib.sha256(pathlib.Path(a.host_gpu_report).read_bytes()).hexdigest()},'joint_pass':False,"
            "'candidate_exists':pathlib.Path(a.candidate).is_file()}\n"
            "(out/'result.json').write_text(json.dumps(result))\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        harness_sha = self.make_git_repo(harness_root)
        task = json.loads(self.task.read_text())
        task["aiter_sha"] = aiter_sha
        task["harness_revision"] = harness_sha
        task["harness_spec_sha256"] = hashlib.sha256(spec.read_bytes()).hexdigest()
        self.task.write_text(json.dumps(task), encoding="utf-8")
        self.freeze.write_text(json.dumps(runner.freeze_payload(self.task)), encoding="utf-8")
        compiler = self.root / "fake-hipcc"
        compiler.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "if '--version' in sys.argv: print('fake-hipcc 1.0')\n"
            "else: pathlib.Path(sys.argv[-1]).write_bytes(b'fake ELF')\n",
            encoding="utf-8",
        )
        compiler.chmod(0o755)
        return harness_root, spec, aiter_root, compiler

    def score_args(self, harness_root, spec, aiter_root, compiler):
        return argparse.Namespace(task=self.task, freeze=self.freeze,
            run_dir=self.root / "results" / "fixture--v1--r001",
            harness_root=harness_root, spec=spec, aiter_source=aiter_root,
            withheld_spec=self.withheld, host_gpu_report=self.host_report,
            output=self.root / "scores", snapshot="all", score_wall_seconds=5,
            build_wall_seconds=5, correctness_only=True, dry_run=False,
            compiler=str(compiler))

    def test_scores_exported_snapshots_with_trusted_harness(self):
        harness_root, spec, aiter_root, compiler = self.setup_scorer()
        self.assertEqual(runner.run_agent(self.args()), 0)
        self.assertEqual(runner.score_snapshots(self.score_args(harness_root, spec, aiter_root, compiler)), 0)
        scores = json.loads((self.root / "scores" / "score_manifest.json").read_text())
        self.assertEqual([item["status"] for item in scores["results"]], ["completed", "completed"])
        final = self.root / "scores" / "snapshot-000002"
        self.assertEqual((final / "candidate" / "kernel.hip").read_text(), "// candidate\n")
        self.assertTrue(json.loads((final / "harness" / "result.json").read_text())["candidate_exists"])
        self.assertEqual(scores["results"][-1]["binary_sha256"], hashlib.sha256(b"fake ELF").hexdigest())
        self.assertEqual(scores["results"][-1]["scorer"]["exit_code"], 1)
        self.assertNotIn(str(self.withheld), (final / "score_record.json").read_text())
        self.assertNotIn("secret_case", (self.root / "scores" / "score_manifest.json").read_text())

    def test_refuses_tampered_snapshot_blob(self):
        harness_root, spec, aiter_root, compiler = self.setup_scorer()
        self.assertEqual(runner.run_agent(self.args()), 0)
        run_dir = self.root / "results" / "fixture--v1--r001"
        final = json.loads((run_dir / "snapshots.jsonl").read_text().splitlines()[-1])
        (run_dir / "blobs" / final["files"]["kernel.hip"]).write_text("tampered", encoding="utf-8")
        args = self.score_args(harness_root, spec, aiter_root, compiler)
        args.snapshot = "final"
        with self.assertRaisesRegex(ValueError, "blob hash mismatch"):
            runner.score_snapshots(args)

    def test_compile_timeout_is_recorded_without_invoking_scorer(self):
        harness_root, spec, aiter_root, compiler = self.setup_scorer()
        self.assertEqual(runner.run_agent(self.args()), 0)
        compiler.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, time\n"
            "if '--version' in sys.argv: print('fake-hipcc 1.0')\n"
            "else: time.sleep(10)\n",
            encoding="utf-8",
        )
        args = self.score_args(harness_root, spec, aiter_root, compiler)
        args.snapshot = "final"
        args.build_wall_seconds = 1
        self.assertEqual(runner.score_snapshots(args), 1)
        score = json.loads((self.root / "scores" / "snapshot-000002" / "score_record.json").read_text())
        self.assertEqual(score["status"], "compile_timeout")
        self.assertFalse((self.root / "scores" / "snapshot-000002" / "harness").exists())

    def test_rejects_mismatched_scorer_result(self):
        result = self.root / "result.json"
        result.write_text(json.dumps({"task_id": "wrong", "joint_pass": False, "environment": {}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "task_id mismatch"):
            runner.validate_harness_result(result, json.loads(self.task.read_text()), "c" * 64, "d" * 64, "e" * 64, "f" * 64, "1" * 64, "2" * 64)

    def test_rejects_withheld_manifest_commitment_mismatch(self):
        harness_root, spec, aiter_root, compiler = self.setup_scorer()
        self.assertEqual(runner.run_agent(self.args()), 0)
        self.withheld.write_text('{"changed":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "withheld-case SHA256"):
            runner.score_snapshots(self.score_args(harness_root, spec, aiter_root, compiler))
        self.assertFalse((self.root / "scores").exists())

    def test_rejects_wrong_host_gpu_report(self):
        harness_root, spec, aiter_root, compiler = self.setup_scorer()
        self.assertEqual(runner.run_agent(self.args()), 0)
        self.host_report.write_text(json.dumps({
            "schema_version": 1, "gpu_name": "MI300X", "arch": "gfx942", "card_model": "0x74a1",
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "host GPU report"):
            runner.score_snapshots(self.score_args(harness_root, spec, aiter_root, compiler))
        self.assertFalse((self.root / "scores").exists())


if __name__ == "__main__":
    unittest.main()
