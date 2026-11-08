import csv
import json
import tempfile
import unittest
from pathlib import Path

from inspect_results import FILES, inspect, run, sha256


def recipes(n_test=3, n_ood=2, negative=()):
    out = []
    for split, n, prefix in [("test", n_test, "r"), ("ood", n_ood, "ood")]:
        for i in range(n):
            rid = f"{prefix}{i:04d}"
            out.append(
                dict(
                    rid=rid,
                    split=split,
                    energy=-7.5 if rid in negative else 100.0 + i,
                    steps=[0.1 + 0.01 * i, 0.2, 0.3][: 2 + i % 2],
                )
            )
    return out


def write_folder(folder, items, mutate=None):
    """Write a self-consistent set of the four CSVs, then apply mutate(tables)."""
    tables = {name: [] for name in FILES}
    for it in items:
        tables["recipe_conditions.csv"].append(
            dict(
                rid=it["rid"],
                split=it["split"],
                step_min=1.0,
                ion_flux=10.0,
                etchant_flux=2000.0,
                oxygen_flux=500.0,
                ion_mean_energy=it["energy"],
                ion_exponent=200.0,
                mask_width=8.0,
                mask_height=9.0,
            )
        )
        for s, v in enumerate(it["steps"]):
            tables["per_step.csv"].append(
                dict(
                    split=it["split"],
                    rid=it["rid"],
                    step=s,
                    model_um=v,
                    baseline_um=2 * v,
                    depth_mae_um=v / 2,
                )
            )
        m = sum(it["steps"]) / len(it["steps"])
        tables["per_recipe.csv"].append(
            dict(
                split=it["split"], rid=it["rid"], model_um=m, baseline_um=2 * m, depth_mae_um=m / 2
            )
        )
    for split in ["ood", "test"]:
        rows = [r for r in tables["per_recipe.csv"] if r["split"] == split]

        def mean(c):
            return sum(r[c] for r in rows) / len(rows)

        tables["summary.csv"].append(
            dict(
                split=split,
                recipes=len(rows),
                model_um=mean("model_um"),
                baseline_um=mean("baseline_um"),
                depth_mae_um=mean("depth_mae_um"),
                gain_pct=50.0,
            )
        )
    if mutate:
        mutate(tables)
    for name, rows in tables.items():
        with (Path(folder) / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FILES[name])
            w.writeheader()
            w.writerows(rows)


class InspectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "in"
        self.dir.mkdir()
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def kinds(self, result):
        return {p["kind"] for p in result["problems"]}

    def test_clean_input_is_hold_not_pass_and_count_is_not_five(self):
        write_folder(self.dir, recipes())
        out, ins, dec = run(self.dir, self.out, "m")
        self.assertEqual(ins["problems"], [])
        self.assertEqual(ins["recipe_conditions"]["negative_ion_mean_energy_count"], 0)
        self.assertEqual(ins["data_check"]["invalid_recipe_count"], 0)
        self.assertNotEqual(ins["data_check"]["invalid_recipe_count"], 5)
        self.assertEqual(dec["action"], "HOLD")
        self.assertFalse(dec["training_started"])
        self.assertFalse(dec["active_model_changed"])

    def test_counts_computed_from_data(self):
        write_folder(self.dir, recipes(n_test=4, n_ood=3))
        rc = inspect(self.dir)["recipe_conditions"]
        self.assertEqual(rc["recipe_count"], 7)
        self.assertEqual(rc["split_counts"], dict(test=4, ood=3))

    def test_negative_energy_ids_and_values_reported(self):
        write_folder(self.dir, recipes(negative=("ood0001", "r0002")))
        out, ins, dec = run(self.dir, self.out, "m")
        rc = ins["recipe_conditions"]
        self.assertEqual(rc["negative_ion_mean_energy_count"], 2)
        self.assertEqual({x["rid"] for x in rc["negative_ion_mean_energy"]}, {"ood0001", "r0002"})
        self.assertTrue(all(x["ion_mean_energy"] == -7.5 for x in rc["negative_ion_mean_energy"]))
        self.assertEqual(ins["data_check"]["invalid_recipe_count"], 2)
        self.assertEqual(dec["action"], "DATA_REVIEW")

    def test_duplicate_recipe_detected(self):
        write_folder(
            self.dir, recipes(), lambda t: t["per_recipe.csv"].append(dict(t["per_recipe.csv"][0]))
        )
        self.assertIn("duplicate", self.kinds(inspect(self.dir)))

    def test_missing_and_unknown_ids_detected(self):
        def mutate(t):
            t["per_step.csv"] = [r for r in t["per_step.csv"] if r["rid"] != "r0001"]
            t["per_recipe.csv"][0]["rid"] = "zzz"

        write_folder(self.dir, recipes(), mutate)
        k = self.kinds(inspect(self.dir))
        self.assertIn("id_missing", k)
        self.assertIn("id_unknown", k)

    def test_abnormal_numbers_detected(self):
        for bad in ["nan", "inf", "", "abc", -0.5]:
            with self.subTest(bad=bad):
                write_folder(
                    self.dir, recipes(), lambda t: t["per_recipe.csv"][1].update(model_um=bad)
                )
                result = inspect(self.dir)
                self.assertFalse(result["data_check"]["passed"])
                self.assertTrue(
                    self.kinds(result) & {"non_finite_or_missing_number", "negative_value"}
                )

    def test_step_gap_and_aggregate_mismatch_detected(self):
        def mutate(t):
            t["per_step.csv"][0]["step"] = 5
            t["summary.csv"][0]["model_um"] = 99.0

        write_folder(self.dir, recipes(), mutate)
        k = self.kinds(inspect(self.dir))
        self.assertIn("step_gap", k)
        self.assertIn("aggregate_mismatch", k)

    def test_structural_problem_without_rid_still_routes_to_data_review(self):
        write_folder(self.dir, recipes(), lambda t: t["summary.csv"][0].update(recipes=999))
        out, ins, dec = run(self.dir, self.out, "m")
        self.assertIn("count_mismatch", self.kinds(ins))
        self.assertEqual(dec["action"], "DATA_REVIEW")

    def test_missing_file_detected(self):
        write_folder(self.dir, recipes())
        (self.dir / "per_step.csv").unlink()
        self.assertIn("missing_file", self.kinds(inspect(self.dir)))

    def test_rollout_and_self_intersection_unmeasured(self):
        write_folder(self.dir, recipes())
        ins = inspect(self.dir)
        self.assertEqual(set(ins["unmeasured"]), {"self_intersection", "rollout"})

    def test_sources_unchanged_and_outputs_written_as_utf8_json(self):
        write_folder(self.dir, recipes(negative=("ood0000",)))
        before = {p.name: sha256(p) for p in self.dir.iterdir()}
        out, ins, dec = run(self.dir, self.out, "m")
        self.assertEqual(before, {p.name: sha256(p) for p in self.dir.iterdir()})
        names = {p.name for p in out.iterdir()}
        self.assertEqual(
            names,
            {"inspection.json", "source_sha256.json", "controller_input.json", "decision.json"},
        )
        hashes = json.loads((out / "source_sha256.json").read_text(encoding="utf-8"))
        self.assertEqual({n: v["sha256"] for n, v in hashes.items()}, before)
        saved = json.loads((out / "decision.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["action"], dec["action"])
        ctrl = json.loads((out / "controller_input.json").read_text(encoding="utf-8"))
        self.assertEqual(ctrl["policy"]["status"], "draft")
        self.assertEqual(ctrl["current"]["data_check"], ins["data_check"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
