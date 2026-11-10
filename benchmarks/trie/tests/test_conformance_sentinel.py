"""A normal GLM punctuation token is not evidence of forced-state leakage."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

target = Path(os.environ.get("CONFORMANCE_TEST_TARGET", Path(__file__).resolve().parents[1] / "scripts/sglang_strict_replay_conformance.py"))
spec = importlib.util.spec_from_file_location("conformance", target)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class SentinelTests(unittest.TestCase):
    recorded_control_ids = [576, 9101, 374, 39483, 323, 8204, 624, 785, 9101, 374, 39483, 323, 8204, 13, 154827, 198, 3838, 374, 279, 9101, 1075, 3351, 5267, 40, 2776, 14579, 11, 358, 4157, 3410, 1931, 7246, 9101, 1995, 13, 154827, 198, 6713, 498, 3291, 752, 264, 21594, 911, 279, 9101, 5267, 10229, 1521, 279, 9101, 1895, 1184, 311, 36688, 5267, 17922, 432, 572, 2480, 315, 8599, 0, 154827]

    def probe(self, *, leak=False, broken_control=False):
        def native(client, base, prompt, length, *, label, forced=None, forbidden_tokens=None):
            if forced is not None:
                self.assertIsNone(forbidden_tokens)
                return {"output_ids": list(forced)}
            ids = list(self.recorded_control_ids) if length == 64 else [2] * length
            # A benign batch variant contains ! unless explicitly excluded.
            if "normal" in label and "reference" not in label and forbidden_tokens is None:
                ids[27] = 0
            if leak and label == "mixed-batch normal":
                ids[2] = 154878
            if broken_control and label == "mixed-batch control-0 normal":
                ids[2] = 154878
            return {"output_ids": ids}
        with patch.object(checker, "_FORCED_ALPHABET", (154878,154879)), patch.object(checker, "_native_generate", side_effect=native), patch.object(
                checker, "_forced_result", return_value={"output_tokens": 1}):
            return checker._mixed_batch_probe(None, "http://local", {"prompt_token_ids": [2,3]}, 6, False, "overlap")

    def test_benign_normal_punctuation_does_not_look_like_leakage(self):
        result = self.probe()
        self.assertEqual(result["status"], "pass")

    def test_real_marker_leak_still_fails(self):
        with self.assertRaisesRegex(RuntimeError, "forced request state leaked"):
            self.probe(leak=True)

    def test_unusable_control_fails_before_claiming_leakage(self):
        with self.assertRaisesRegex(RuntimeError, "control-0 naturally emitted a sentinel"):
            self.probe(broken_control=True)

    def test_bias_only_applies_to_ordinary_probes(self):
        calls = []
        class Response:
            def raise_for_status(self): pass
            def json(self): return {"output_ids":[2]}
        class Client:
            def post(self, url, json):
                calls.append(json)
                return Response()
        checker._native_generate(Client(), "http://local", [3], 1, label="normal", forbidden_tokens={0,1})
        checker._native_generate(Client(), "http://local", [3], 1, label="forced", forced=[0])
        self.assertEqual(calls[0]["sampling_params"]["logit_bias"], {"0":-1e9,"1":-1e9})
        self.assertNotIn("logit_bias", calls[1]["sampling_params"])
        self.assertEqual(calls[1]["sampling_params"]["forced_output_token_ids"],[0])

    def test_marker_argument_rejects_ambiguous_configuration(self):
        import argparse
        self.assertEqual(checker._sentinel_ids_arg("154878,154879"), (154878,154879))
        for value in ("0,0", "-1,2", "1", "a,b", "1,2,3"):
            with self.assertRaises(argparse.ArgumentTypeError):
                checker._sentinel_ids_arg(value)
        self.assertEqual(checker._forced_pattern(5, 1, (154878,154879)), [154879,154878,154879,154878,154879])

    def test_native_responses_are_preserved_for_diagnosis(self):
        class Response:
            def raise_for_status(self): pass
            def json(self): return {"output_ids":[0], "meta_info":{"completion_tokens":1}}
        class Client:
            def post(self, url, json): return Response()
        with tempfile.TemporaryDirectory() as tmp, patch.object(checker, "_NATIVE_CAPTURE_PATH", Path(tmp)/"native.jsonl"):
            body = checker._native_generate(Client(), "http://local", [3], 1, label="mixed normal", forbidden_tokens={0,1})
            saved=json.loads((Path(tmp)/"native.jsonl").read_text())
            self.assertEqual(saved["response"],body)
            self.assertEqual(saved["label"],"mixed normal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
