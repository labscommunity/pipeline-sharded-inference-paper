"""Post-hoc beam_idx Gather injection on already-exported stateful OV shards.

Reads an OV IR, checks it is stateful (has ReadValue ops), adds a beam_idx
Parameter and a Gather(ReadValue, beam_idx, axis=0) on every KV ReadValue,
and writes a new IR. Lets us apply the v5_beam fusion unlock to shards
that were exported before we had the technique—e.g. Gemma 4 FP32 shards
from April.

Usage:
  python inject_beam_idx_gemma.py <input_dir> <output_dir>
    input_dir must contain openvino_model.xml + openvino_model.bin
    output_dir written with the same layout
"""
import os, sys, shutil
import openvino as ov
import openvino.opset13 as opset
from openvino import PartialShape, Type


def inject(input_dir, output_dir):
    xml_in = os.path.join(input_dir, "openvino_model.xml")
    if not os.path.isfile(xml_in):
        print(f"ERROR: no openvino_model.xml in {input_dir}")
        sys.exit(1)

    core = ov.Core()
    model = core.read_model(xml_in)

    # Quick introspection
    read_values = [n for n in model.get_ops() if n.get_type_name() == "ReadValue"]
    params = [p.get_friendly_name() for p in model.get_parameters()]
    print(f"Input IR: {xml_in}")
    print(f"  parameters: {params}")
    print(f"  ReadValue op count: {len(read_values)}")

    if len(read_values) == 0:
        print("  NOT STATEFUL — no KV ReadValue ops. beam_idx injection not applicable.")
        sys.exit(2)

    if "beam_idx" in params:
        print("  Model ALREADY has a beam_idx parameter. Skipping injection.")
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(xml_in, os.path.join(output_dir, "openvino_model.xml"))
        shutil.copy(xml_in.replace(".xml", ".bin"), os.path.join(output_dir, "openvino_model.bin"))
        return

    # Inject beam_idx
    beam_idx_param = opset.parameter(PartialShape([-1]), Type.i32, name="beam_idx")
    beam_idx_param.set_friendly_name("beam_idx")
    beam_idx_param.output(0).set_names({"beam_idx"})
    axis_const = opset.constant(0, Type.i32)

    for rv in read_values:
        rv_out = rv.output(0)
        gather_node = opset.gather(rv_out, beam_idx_param, axis_const)
        gather_out = gather_node.output(0)
        for target_input in list(rv_out.get_target_inputs()):
            if target_input.get_node() is gather_node:
                continue
            target_input.replace_source_output(gather_out)

    model.add_parameters([beam_idx_param])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    xml_out = os.path.join(output_dir, "openvino_model.xml")
    ov.save_model(model, xml_out, compress_to_fp16=False)
    print(f"OK: wrote {xml_out}")
    print(f"  new parameters: {[p.get_friendly_name() for p in model.get_parameters()]}")
    print(f"  Gather nodes injected: {len(read_values)}")

    # Copy stage_config if present
    src_cfg = os.path.join(input_dir, "stage_config.json")
    if os.path.isfile(src_cfg):
        shutil.copy(src_cfg, os.path.join(output_dir, "stage_config.json"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    inject(sys.argv[1], sys.argv[2])
