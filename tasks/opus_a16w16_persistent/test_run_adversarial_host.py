import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_adversarial_host.sh")


class HostWatchdogTests(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_missing_environment_fails_before_docker(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "aligned-nooob"],
            env={"PATH": "/usr/bin:/bin"}, text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AITERRS_STUDY_ROOT", result.stderr)


if __name__ == "__main__":
    unittest.main()
