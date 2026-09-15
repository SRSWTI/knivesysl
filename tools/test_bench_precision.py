#!/usr/bin/env python3
"""CPU-only mathematical and reference-file regressions for bench_precision."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

# The benchmarks use sibling imports when invoked as scripts. Support unittest's
# package-style discovery without replacing any imported module or model API.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import bench_precision as bp
finally:
    sys.path.pop(0)


class RowMetricsTests(unittest.TestCase):
    def test_identical_distributions_have_zero_divergence_and_error(self):
        probabilities = np.array([0.125, 0.25, 0.625], dtype=np.float64)
        logits = np.log(probabilities)
        for target in range(logits.size):
            with self.subTest(target=target):
                row = bp.row_metrics(logits, target, logits.copy())
                self.assertAlmostEqual(row["candidate_nll"], -math.log(probabilities[target]))
                self.assertAlmostEqual(row["reference_nll"], row["candidate_nll"])
                for key in ("nll_delta", "kl_ref_to_candidate", "absolute_error_sum",
                            "max_abs_logit_error", "squared_error_sum"):
                    self.assertEqual(row[key], 0.0)
                self.assertTrue(row["argmax_agrees"])
                self.assertEqual(row["candidate_argmax"], 2)

    def test_kl_uses_reference_to_candidate_direction(self):
        reference = np.array([0.25, 0.75], dtype=np.float64)
        candidate = np.array([0.5, 0.5], dtype=np.float64)
        row = bp.row_metrics(np.log(candidate), 1, np.log(reference))
        expected = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
        self.assertAlmostEqual(row["kl_ref_to_candidate"], expected)
        self.assertAlmostEqual(row["nll_delta"], math.log(1.5))
        self.assertFalse(row["argmax_agrees"])

    def test_constant_shift_preserves_distribution_but_not_raw_error(self):
        reference = np.array([-3, 0, 2, 5], dtype=np.float32)
        for shift in (-1024.0, 128.0):
            with self.subTest(shift=shift):
                row = bp.row_metrics(reference + shift, 1, reference)
                summary = bp.summarize([row], compared=True)
                self.assertAlmostEqual(row["candidate_nll"], row["reference_nll"])
                self.assertAlmostEqual(row["nll_delta"], 0.0)
                self.assertAlmostEqual(row["kl_ref_to_candidate"], 0.0)
                self.assertAlmostEqual(row["candidate_logsumexp"] - row["reference_logsumexp"], shift)
                self.assertEqual(summary["argmax_agreement"], 1.0)
                self.assertEqual(summary["mean_abs_logit_error"], abs(shift))
                self.assertEqual(summary["max_abs_logit_error"], abs(shift))
                self.assertEqual(summary["rms_logit_error"], abs(shift))
                expected_l2 = abs(shift) * math.sqrt(reference.size / float(np.dot(reference, reference)))
                self.assertAlmostEqual(summary["relative_l2_logit_error"], expected_l2)

    def test_extreme_finite_float32_logits_remain_stable(self):
        limit = float(np.finfo(np.float32).max)
        for offset in (-limit, limit):
            with self.subTest(offset=offset):
                logits = np.full(4, offset, dtype=np.float32)
                row = bp.row_metrics(logits, 2, logits.copy())
                self.assertAlmostEqual(row["candidate_nll"], math.log(4))
                self.assertEqual(row["kl_ref_to_candidate"], 0.0)
                self.assertTrue(all(math.isfinite(v) for v in row.values()))
                self.assertTrue(all(math.isfinite(v) for v in bp.summarize([row], True).values()))
        logits = np.array([-limit, 0, limit], dtype=np.float32)
        for target, expected in enumerate((2 * limit, limit, 0.0)):
            with self.subTest(target=target):
                row = bp.row_metrics(logits, target, logits.copy())
                self.assertEqual(row["candidate_nll"], expected)
                self.assertEqual(row["kl_ref_to_candidate"], 0.0)
                self.assertTrue(all(math.isfinite(v) for v in row.values()))

    def test_nonfinite_logits_are_rejected_on_either_side(self):
        finite = np.array([0.0, 1.0], dtype=np.float32)
        for value in (np.nan, np.inf, -np.inf):
            invalid = np.array([0.0, value], dtype=np.float32)
            with self.subTest(value=value, side="candidate"):
                with self.assertRaisesRegex(ValueError, "candidate logits"):
                    bp.row_metrics(invalid, 0, finite)
            with self.subTest(value=value, side="reference"):
                with self.assertRaisesRegex(ValueError, "reference logits"):
                    bp.row_metrics(finite, 0, invalid)


class SummarizeTests(unittest.TestCase):
    def test_unequal_batches_are_weighted_by_rows_and_errors_by_elements(self):
        # summarize accepts flattened request rows, not per-batch summaries.
        # Unequal vector sizes additionally expose accidental per-row averaging
        # of absolute/RMS errors, although a model run uses a fixed vocabulary.
        references = [np.array(v, dtype=np.float64) for v in
                      ([0, 2], [1, -1, 3], [2, 0, -2, 1], [-2, 4])]
        candidates = [np.array(v, dtype=np.float64) for v in
                      ([3, -1], [2, -1, 1], [2, 1, -1, 1], [-1, 5])]
        targets = [1, 2, 0, 1]
        rows = [bp.row_metrics(c, t, r) for c, t, r in zip(candidates, targets, references)]
        batches = [rows[:1], rows[1:]]
        summary = bp.summarize([row for batch in batches for row in batch], compared=True)
        c_nll = [float(np.logaddexp.reduce(c) - c[t]) for c, t in zip(candidates, targets)]
        r_nll = [float(np.logaddexp.reduce(r) - r[t]) for r, t in zip(references, targets)]
        kls = []
        for c, r in zip(candidates, references):
            clp, rlp = c - np.logaddexp.reduce(c), r - np.logaddexp.reduce(r)
            kls.append(float(np.sum(np.exp(rlp) * (rlp - clp))))
        delta = np.concatenate(candidates) - np.concatenate(references)
        reference = np.concatenate(references)
        expected = {
            "requests": 4,
            "candidate_mean_nll": sum(c_nll) / 4,
            "candidate_max_nll": max(c_nll),
            "reference_mean_nll": sum(r_nll) / 4,
            "mean_nll_delta": (sum(c_nll) - sum(r_nll)) / 4,
            "mean_kl_ref_to_candidate": sum(kls) / 4,
            "max_kl_ref_to_candidate": max(kls),
            "mean_abs_logit_error": float(np.mean(np.abs(delta))),
            "max_abs_logit_error": float(np.max(np.abs(delta))),
            "rms_logit_error": float(np.sqrt(np.mean(delta ** 2))),
            "reference_rms_logits": float(np.sqrt(np.mean(reference ** 2))),
            "relative_l2_logit_error": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
            "argmax_agreement": 0.5,
        }
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(summary[key], value)

    def test_candidate_only_summary(self):
        rows = [bp.row_metrics(np.array([0.0, 0.0]), 0),
                bp.row_metrics(np.log(np.array([0.25, 0.75])), 0)]
        summary = bp.summarize(rows, compared=False)
        self.assertEqual(summary["requests"], 2)
        self.assertAlmostEqual(summary["candidate_mean_nll"], (math.log(2) + math.log(4)) / 2)
        self.assertAlmostEqual(summary["candidate_max_nll"], math.log(4))

    def test_zero_reference_norm_is_defined_only_for_zero_error(self):
        zeros = np.zeros(3, dtype=np.float32)
        summary = bp.summarize([bp.row_metrics(zeros, 0, zeros)], True)
        self.assertEqual(summary["relative_l2_logit_error"], 0.0)
        with self.assertRaisesRegex(ValueError, "reference logit norm is zero"):
            bp.summarize([bp.row_metrics(np.ones(3, dtype=np.float32), 0, zeros)], True)


class ReferenceFileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = str(Path(directory.name) / "reference.npy")
        corpus_path = Path(directory.name) / "corpus.txt"
        corpus_path.write_bytes(b"abcdabcd")
        corpus = corpus_path.read_bytes()
        # A deterministic CPU token stream suffices to exercise the file contract;
        # no tokenizer or model is constructed by these tests.
        ids = [byte - ord("a") for byte in corpus]
        context, steps, slots, stride = 1, 2, 2, 3
        inputs = [[ids[(slot * stride + context + step) % (len(ids) - 1)]
                   for slot in range(slots)] for step in range(steps)]
        targets = [[ids[(slot * stride + context + step + 1) % (len(ids) - 1)]
                    for slot in range(slots)] for step in range(steps)]
        self.contract = {
            "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
            "token_ids_sha256": bp.hash_json(ids), "corpus_token_count": len(ids),
            "corpus_cycle": len(ids) - 1, "slot_stride": stride, "context": context,
            "positions": [context, context + 1], "shape_prefix": [steps, slots],
            "decode_input_mode": "strict_teacher_force",
            "input_token_ids_sha256": bp.hash_json(inputs),
            "target_token_ids_sha256": bp.hash_json(targets),
            "target_rule": "corpus at absolute input position + 1", "slot_order": [0, 1],
        }
        self.logits = np.arange(16, dtype="<f4").reshape(2, 2, 4) / 4
        self.metadata = {
            "schema_version": 1, "status": "complete", "input_contract": self.contract,
            "shape": list(self.logits.shape), "dtype": "<f4", "baseline_environment": {},
            "baseline_arguments": {"corpus": str(corpus_path), "context": context,
                                   "steps": steps, "concurrency": slots},
            "logits_payload_sha256": hashlib.sha256(self.logits.tobytes(order="C")).hexdigest(),
        }
        np.save(self.path, self.logits, allow_pickle=False)
        self.save_metadata(self.metadata)

    def save_metadata(self, metadata):
        bp.json_write(self.path + ".json", metadata)

    def test_matching_reference_loads_as_readonly_mmap(self):
        reference, metadata = bp.load_reference(self.path, copy.deepcopy(self.contract))
        self.assertIsInstance(reference, np.memmap)
        self.assertFalse(reference.flags.writeable)
        np.testing.assert_array_equal(reference, self.logits)
        self.assertEqual(metadata, self.metadata)
        with open(self.path + ".json", encoding="utf-8") as stream:
            self.assertEqual(json.load(stream), self.metadata)
        self.assertFalse(Path(self.path + ".json.tmp").exists())

    def test_corpus_and_input_alignment_mismatches_are_rejected(self):
        differences = {
            "corpus_sha256": hashlib.sha256(b"another corpus").hexdigest(),
            "token_ids_sha256": bp.hash_json([3, 2, 1, 0]),
            "input_token_ids_sha256": bp.hash_json([[3, 2], [1, 0]]),
            "target_token_ids_sha256": bp.hash_json([[0, 1], [2, 3]]),
            "positions": [2, 3], "context": 2, "slot_order": [1, 0],
            "shape_prefix": [1, 4],
        }
        for key, value in differences.items():
            with self.subTest(field=key):
                contract = copy.deepcopy(self.contract)
                contract[key] = value
                with self.assertRaisesRegex(ValueError, "corpus/token hashes, positions, or input layout differ"):
                    bp.load_reference(self.path, contract)

    def test_payload_sidecar_and_contract_shape_must_agree(self):
        for shape in ((2, 2, 5), (1, 4, 4), (4, 4)):
            for accurate_sidecar in (False, True):
                with self.subTest(shape=shape, accurate_sidecar=accurate_sidecar):
                    np.save(self.path, np.zeros(shape, dtype="<f4"), allow_pickle=False)
                    metadata = copy.deepcopy(self.metadata)
                    if accurate_sidecar:
                        metadata["shape"] = list(shape)
                    self.save_metadata(metadata)
                    # A vocabulary change is rejected here only if the sidecar
                    # disagrees. Candidate vocabulary matching happens in main.
                    if shape == (2, 2, 5) and accurate_sidecar:
                        with self.assertRaisesRegex(ValueError, "checksum"):
                            bp.load_reference(self.path, self.contract)
                    else:
                        with self.assertRaisesRegex(ValueError, "shapes differ"):
                            bp.load_reference(self.path, self.contract)

    def test_incomplete_or_unbound_sidecars_are_rejected(self):
        for field, value in (("schema_version", -1), ("status", "incomplete"),
                             ("baseline_environment", None), ("baseline_arguments", {})):
            with self.subTest(field=field):
                metadata = copy.deepcopy(self.metadata)
                metadata[field] = value
                self.save_metadata(metadata)
                with self.assertRaises(ValueError):
                    bp.load_reference(self.path, self.contract)

    def test_payload_changes_cannot_reuse_sidecar_checksum(self):
        altered = self.logits.copy()
        altered[1, 1, 3] += 1
        np.save(self.path, altered, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "checksum"):
            bp.load_reference(self.path, self.contract)

    def test_nonfinite_payload_is_rejected_even_with_matching_checksum(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                invalid = self.logits.copy()
                invalid[1, 0, 0] = value
                np.save(self.path, invalid, allow_pickle=False)
                metadata = copy.deepcopy(self.metadata)
                metadata["logits_payload_sha256"] = hashlib.sha256(invalid.tobytes()).hexdigest()
                self.save_metadata(metadata)
                with self.assertRaisesRegex(ValueError, "payload contains NaN or infinity"):
                    bp.load_reference(self.path, self.contract)

    def test_payload_dtype_and_layout_are_validated(self):
        for array in (self.logits.astype(np.float64), np.asfortranarray(self.logits)):
            with self.subTest(dtype=str(array.dtype), c_contiguous=array.flags.c_contiguous):
                np.save(self.path, array, allow_pickle=False)
                with self.assertRaisesRegex(ValueError, "C-contiguous little-endian float32"):
                    bp.load_reference(self.path, self.contract)


if __name__ == "__main__":
    unittest.main()
