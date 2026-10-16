from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.quantization_study import scan_groups
from tools.quantize_ncnn import filter_table, layer_inventory, set_activation_ranges


class QuantizationStudyTests(unittest.TestCase):
    def test_known_input_range_changes_only_its_activation_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table = Path(temporary) / "source.table"
            table.write_text("conv_3_param_0 1 2\nconv_3 283.483368\nconv_30 6\n")
            set_activation_ranges(table, (("conv_3", 1.0),))
            self.assertEqual(table.read_text(), "conv_3_param_0 1 2\nconv_3 127\nconv_30 6\n")

    def test_activation_range_rejects_missing_layers_and_nonphysical_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table = Path(temporary) / "source.table"
            original = "conv_3_param_0 1 2\nconv_3 283.483368\n"
            for name, maximum in (
                ("conv_4", 1.0),
                ("conv_3_param_0", 1.0),
                ("conv_3", 0.0),
                ("conv_3", float("nan")),
                ("conv_3", float("inf")),
            ):
                with self.subTest(name=name, maximum=maximum):
                    table.write_text(original)
                    with self.assertRaises(ValueError):
                        set_activation_ranges(table, ((name, maximum),))
                    self.assertEqual(table.read_text(), original)

    def test_float_filter_removes_activation_and_weight_entries_without_prefix_collisions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.table"
            destination = Path(temporary) / "mixed.table"
            source.write_text("conv_3_param_0 1 2\nconv_30_param_0 3 4\nconv_3 5\nconv_30 6\n")
            filter_table(source, destination, ("conv_3",))
            self.assertEqual(destination.read_text(), "conv_30_param_0 3 4\nconv_30 6\n")

    def test_filter_rejects_unknown_layers_and_empty_quantisation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.table"
            source.write_text("conv_3_param_0 1 2\nconv_3 5\n")
            for names, message in ((("conv_4",), "absent"), (("conv_3",), "at least one")):
                with self.subTest(names=names):
                    with self.assertRaisesRegex(ValueError, message):
                        filter_table(source, Path(temporary) / "out.table", names)

    def test_inventory_reads_quantisation_flags_and_graph_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parameters = Path(temporary) / "model.param"
            parameters.write_text(
                "7767517\n4 4\nInput input 0 1 in0\n"
                "Convolution conv_3 1 1 in0 x 0=16 8=102\n"
                "ConvolutionDepthWise convdw_212 1 1 x y 0=64 7=64\n"
                "InnerProduct classifier 1 1 y out0 0=1 8=2\n"
            )
            inventory = layer_inventory(parameters)
            self.assertEqual([layer["quantised"] for layer in inventory], [True, False, True])
            self.assertEqual(inventory[0]["inputs"], ["in0"])
            self.assertEqual(inventory[0]["outputs"], ["x"])
            self.assertEqual(inventory[0]["output_channels"], 16)
            self.assertEqual(inventory[0]["int8_scale_term"], 102)

    def test_sensitivity_groups_partition_the_graph_and_separate_head_branches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            names = (
                "conv_3",
                "conv_4",
                "conv_78",
                "conv_79",
                "conv_87",
                "conv_88",
                "conv_96",
                "convdw_211",
                "convdw_212",
                "convdw_217",
            )
            (model / "model.ncnn.param").write_text(
                "7767517\n10 11\n"
                + "\n".join(f"Convolution {name} 1 1 x y 0=1" for name in names)
                + "\n"
            )
            groups = scan_groups(model)
            scanned = [name for group in groups.values() for name in group]
            self.assertCountEqual(scanned, names)
            self.assertEqual(len(scanned), len(set(scanned)))
            self.assertEqual(groups["box"], ("conv_79", "conv_87"))
            self.assertEqual(groups["class"], ("conv_88", "conv_96", "convdw_212", "convdw_217"))


if __name__ == "__main__":
    unittest.main()
