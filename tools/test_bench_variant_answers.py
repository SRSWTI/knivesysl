#!/usr/bin/env python3
"""Real JSON grading and isolated-code execution checks; no model or GPU needed."""
from pathlib import Path
import tempfile
import unittest

from tools import bench_variant_answers as bench


class AnswerGradeTests(unittest.TestCase):
    def test_json_fences_and_key_order_do_not_change_answers(self):
        grade = bench.grade_json('```json\n{"b":null,"a":[1,2]}\n```', {"a": [1, 2], "b": None})
        self.assertTrue(grade["passed"])

    def test_wrong_values_fail(self):
        self.assertFalse(bench.grade_json("[-3,2,-3,-1]", [-3, 2, -3, -2])["passed"])

    def test_correct_values_in_wrong_requested_shape_still_fail(self):
        grade = bench.grade_json('{"shape":[3,4,5]}', [3, 4, 5])
        self.assertFalse(grade["passed"])
        self.assertEqual(grade["actual"], {"shape": [3, 4, 5]})

    def test_invalid_json_has_an_explicit_failure(self):
        grade = bench.grade_json("not JSON", None)
        self.assertFalse(grade["passed"])
        self.assertIn("error", grade)

    def test_correct_code_must_pass_every_case(self):
        tests = [{"args": [2, 3], "expected": 5}, {"args": [-1, 1], "expected": 0}]
        grade = bench.grade_code("```python\ndef solve(a, b):\n    return a + b\n```", tests)
        self.assertTrue(grade["passed"], grade)
        self.assertEqual(grade["passed_tests"], len(tests))
        self.assertEqual(grade["total_tests"], len(tests))

    def test_one_wrong_code_result_fails_the_task(self):
        tests = [{"args": [2, 3], "expected": 5}, {"args": [0, 0], "expected": 0}]
        grade = bench.grade_code("def solve(a, b):\n    return a - b\n", tests)
        self.assertFalse(grade["passed"])
        self.assertEqual(grade["passed_tests"], 1)
        self.assertEqual(grade["total_tests"], 2)

    def test_mutating_inputs_fails_even_when_the_return_value_matches(self):
        grade = bench.grade_code("def solve(a):\n    a.sort()\n    return a\n",
                                 [{"args": [[2, 1]], "expected": [1, 2]}])
        self.assertFalse(grade["passed"])
        self.assertIn("mutated inputs", grade["failures"][0]["error"])

    def test_missing_function_and_invalid_syntax_fail(self):
        for code in ("value = 1", "def solve(:"):
            with self.subTest(code=code):
                grade = bench.grade_code(code, [{"args": [], "expected": 1}])
                self.assertFalse(grade["passed"])
                self.assertTrue(grade["failures"])

    def test_sandbox_cannot_read_a_host_temporary_file(self):
        with tempfile.TemporaryDirectory(prefix="answer-host-only-") as directory:
            marker = Path(directory) / "marker.txt"
            marker.write_text("host-only test marker")
            code = f"def solve():\n    return open({str(marker)!r}).read()\n"
            grade = bench.grade_code(code, [{"args": [], "expected": "host-only test marker"}])
        self.assertFalse(grade["passed"])
        self.assertIn("FileNotFoundError", grade["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
