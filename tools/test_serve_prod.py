#!/usr/bin/env python3
"""Exercise the actual Bash launcher in no-launch mode, including a real PTY."""
import errno
import os
from pathlib import Path
import pty
import select
import subprocess
import tempfile
import time
import unittest


LAUNCHER = Path(__file__).resolve().with_name("serve_prod.sh")


def environment(overrides=None):
    values = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("TQ_", "KSL_"))
    }
    values.update(overrides or {})
    return values


def resolved_environment(output):
    return dict(
        line.strip().split("=", 1)
        for line in output.splitlines()
        if line.startswith("  TQ_")
    )


class ProductionLauncherTests(unittest.TestCase):
    def invoke(self, *args, overrides=None, cwd=None):
        return subprocess.run(
            ["bash", str(LAUNCHER), *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=environment(overrides),
            cwd=cwd,
            timeout=10,
        )

    def terminal(self, reply, overrides=None):
        master, slave = pty.openpty()
        process = None
        captured = bytearray()
        deadline = time.monotonic() + 10
        try:
            process = subprocess.Popen(
                ["bash", str(LAUNCHER), "--dry-run"],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                env=environment(overrides),
            )
            os.close(slave)
            slave = None
            sent = False
            while True:
                if time.monotonic() >= deadline:
                    self.fail(f"Terminal launcher timed out: {captured.decode(errors='replace')}")
                if not select.select([master], [], [], 0.1)[0]:
                    continue
                try:
                    data = os.read(master, 65536)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                if not data:
                    break
                captured.extend(data)
                if not sent and b"Choose 1/nvfp4" in captured:
                    os.write(master, reply)
                    sent = True
            self.assertTrue(sent, "Launcher did not prompt on a terminal")
            return process.wait(timeout=5), captured.decode()
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            os.close(master)
            if slave is not None:
                os.close(slave)

    def test_explicit_variants_select_the_corresponding_weight_tier(self):
        for profile, weight in (("nvfp4", "all"), ("mixed", "mlp"), ("fp6", "0")):
            for args in (("--variant", profile), (f"--variant={profile}",)):
                with self.subTest(args=args):
                    result = self.invoke(*args, "--dry-run")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(resolved_environment(result.stdout)["TQ_W_NVFP4"], weight)
                    self.assertNotIn("Choose 1/nvfp4", result.stdout)

    def test_inherited_weight_tier_is_respected_without_a_terminal(self):
        for weight in ("all", "mlp", "0"):
            with self.subTest(weight=weight):
                result = self.invoke("--dry-run", overrides={"TQ_W_NVFP4": weight})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(resolved_environment(result.stdout)["TQ_W_NVFP4"], weight)
                self.assertNotIn("Choose 1/nvfp4", result.stdout)

    def test_explicit_variant_overrides_inherited_weights(self):
        for inherited in ("0", "all", "unsupported"):
            with self.subTest(inherited=inherited):
                result = self.invoke(
                    "--variant", "mixed", "--dry-run",
                    overrides={"TQ_W_NVFP4": inherited},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(resolved_environment(result.stdout)["TQ_W_NVFP4"], "mlp")

    def test_tuning_overrides_survive_profile_selection(self):
        overrides = {
            "TQ_NVFP4_TMA_CTA": "0",
            "TQ_LOAD_MTP": "1",
            "TQ_PAGED_N1_ROWS": "128",
            "TQ_NVF4_DECODE_AUTOTUNE_COLS": "2",
            "TQ_PAGED_SPEC": "1",
            "TQ_PG_SPEC_NODES": "8",
            "TQ_CKPT_HOST_GB": "2",
        }
        result = self.invoke("--variant", "nvfp4", "--dry-run", overrides=overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = resolved_environment(result.stdout)
        for name, value in overrides.items():
            with self.subTest(setting=name):
                self.assertEqual(actual[name], value)

    def test_noninteractive_unconfigured_launch_does_not_wait_for_input(self):
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(resolved_environment(result.stdout)["TQ_W_NVFP4"], ("all", "mlp", "0"))
        self.assertNotIn("Choose 1/nvfp4", result.stdout)

    def test_preview_can_run_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke("--variant", "fp6", "--dry-run", cwd=directory)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resolved_environment(result.stdout)["TQ_W_NVFP4"], "0")
        self.assertIn("[wrapper] Command:", result.stdout)
        self.assertIn("Dry run:", result.stdout)

    def test_help_and_profile_listing_exit_before_launch_configuration(self):
        for flag in ("--help", "-h", "--list-variants"):
            with self.subTest(flag=flag):
                result = self.invoke(flag, overrides={"TQ_W_NVFP4": "unsupported"})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("[wrapper] Command:", result.stdout)
                self.assertNotIn("Choose 1/nvfp4", result.stdout)

    def test_invalid_arguments_fail_without_falling_through_to_launch(self):
        for args in (
            ("--unknown",),
            ("--variant",),
            ("--variant", ""),
            ("--variant=",),
            ("--variant", "fp16"),
            ("--variant", "fp6", "--variant", "mixed"),
            ("--num-blocks", "3328"),
        ):
            with self.subTest(args=args):
                result = self.invoke(*args, "--dry-run")
                self.assertEqual(result.returncode, 64, result.stdout)
                self.assertNotIn("[wrapper] Command:", result.stdout)
                self.assertTrue(result.stderr)

    def test_missing_variant_value_fails_at_end_of_arguments(self):
        result = self.invoke("--variant")
        self.assertEqual(result.returncode, 64)
        self.assertNotIn("[wrapper] Command:", result.stdout)

    def test_unsupported_inherited_tier_requires_an_explicit_selection(self):
        result = self.invoke("--dry-run", overrides={"TQ_W_NVFP4": "unsupported"})
        self.assertEqual(result.returncode, 64)
        self.assertNotIn("[wrapper] Command:", result.stdout)

    def test_terminal_accepts_numbers_and_names(self):
        for reply, weight in ((b"1\n", "all"), (b"2\n", "mlp"), (b"3\n", "0"), (b"mixed\n", "mlp")):
            with self.subTest(reply=reply):
                code, output = self.terminal(reply)
                self.assertEqual(code, 0, output)
                self.assertEqual(resolved_environment(output)["TQ_W_NVFP4"], weight)

    def test_terminal_enter_keeps_the_inherited_selection(self):
        code, output = self.terminal(b"\n", overrides={"TQ_W_NVFP4": "0"})
        self.assertEqual(code, 0, output)
        self.assertEqual(resolved_environment(output)["TQ_W_NVFP4"], "0")

    def test_terminal_invalid_choice_reprompts_before_accepting(self):
        code, output = self.terminal(b"wrong\n2\n")
        self.assertEqual(code, 0, output)
        self.assertGreaterEqual(output.count("Choose 1/nvfp4"), 2)
        self.assertEqual(resolved_environment(output)["TQ_W_NVFP4"], "mlp")

    def test_terminal_eof_aborts_instead_of_starting_a_default_server(self):
        code, output = self.terminal(b"\x04")
        self.assertEqual(code, 64, output)
        self.assertNotIn("[wrapper] Command:", output)


if __name__ == "__main__":
    unittest.main()
