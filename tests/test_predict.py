import importlib.util
import json
import pathlib
import sys
import types
import unittest


class _Input:
    def __init__(self, **_kwargs):
        pass


stub = types.ModuleType("cog")
stub.BaseRunner = object
stub.Input = _Input
stub.Path = pathlib.Path
sys.modules.setdefault("cog", stub)
spec = importlib.util.spec_from_file_location("predict", pathlib.Path(__file__).parents[1] / "predict.py")
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)


class PredictContractTest(unittest.TestCase):
    def test_scale_is_bounded(self):
        self.assertEqual(predict.validated_scale(2), 2)
        with self.assertRaisesRegex(ValueError, "2 or 4"):
            predict.validated_scale(8)

    def test_alpha_output_stays_lossless(self):
        self.assertEqual(predict.destination_for("portrait.webp", True).suffix, ".png")
        self.assertEqual(predict.destination_for("portrait.webp", False).suffix, ".jpg")

    def test_schema_matches_models(self):
        schema = json.loads((pathlib.Path(__file__).parents[1] / "appnz.schema.json").read_text())
        model = next(field for field in schema["inputs"] if field["name"] == "model")
        self.assertEqual(model["choices"], list(predict.MODEL_SPECS))


if __name__ == "__main__":
    unittest.main()
