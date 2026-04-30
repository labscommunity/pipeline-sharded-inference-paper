"""Test KV_CACHE_PRECISION hint on target/draft. Does INT8 KV help on Arc iGPU?"""
import os, time, statistics, numpy as np, openvino as ov
from transformers import AutoTokenizer

TARGET = r"C:\cascadia\models\llama-3.1-8b-int4"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 128


def feed(req, has_beam, input_ids, pos_start):
    n = input_ids.shape[1]
    att = np.ones((1, pos_start + n), dtype=np.int64)
    pos = np.arange(pos_start, pos_start + n, dtype=np.int64).reshape(1, -1)
    fd = {"input_ids": input_ids, "attention_mask": att, "position_ids": pos}
    if has_beam:
        fd["beam_idx"] = np.zeros(1, dtype=np.int32)
    req.infer(fd)
    return req.get_output_tensor(0).data


def simple_decode(req, has_beam, prompt_ids, max_tokens):
    req.reset_state(); pos = 0; gens = []
    l = feed(req, has_beam, prompt_ids, pos); pos += prompt_ids.shape[1]
    nt = int(np.argmax(l[0, -1, :])); gens.append(nt)
    for _ in range(1, max_tokens):
        l = feed(req, has_beam, np.array([[nt]], dtype=np.int64), pos); pos += 1
        nt = int(np.argmax(l[0, -1, :])); gens.append(nt)
    return gens


tok = AutoTokenizer.from_pretrained(TARGET)
input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)
core = ov.Core()

print(f"OV {ov.__version__}  device=GPU", flush=True)
print(f"GPU props:", flush=True)
try:
    print(f"  SUPPORTED_PROPERTIES: {core.get_property('GPU', 'SUPPORTED_PROPERTIES')}", flush=True)
except Exception as e:
    print(f"  couldn't introspect: {e}", flush=True)

configs = [
    {"name": "default (no KV hint)", "config": {}},
    {"name": "KV=f16", "config": {"KV_CACHE_PRECISION": "f16"}},
    {"name": "KV=u8", "config": {"KV_CACHE_PRECISION": "u8"}},
    {"name": "INFERENCE_PRECISION=f16", "config": {"INFERENCE_PRECISION_HINT": "f16"}},
]

model_xml = os.path.join(TARGET, "openvino_model.xml")
results = []
for c in configs:
    name = c["name"]
    try:
        compiled = core.compile_model(core.read_model(model_xml), "GPU", c["config"])
    except Exception as e:
        print(f"\n--- {name} ---  COMPILE FAIL: {type(e).__name__}: {str(e)[:120]}", flush=True)
        continue
    req = compiled.create_infer_request()
    beam = any("beam_idx" in i.get_names() for i in compiled.inputs)
    # warmup
    for _ in range(2):
        simple_decode(req, beam, input_ids, MAX_TOKENS)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        gens = simple_decode(req, beam, input_ids, MAX_TOKENS)
        ts.append(MAX_TOKENS / (time.perf_counter() - t0))
    m = statistics.mean(ts); s = statistics.stdev(ts)
    print(f"--- {name}: {m:.2f} tok/s  sd={s:.2f}  first10={gens[:10]}", flush=True)
    results.append({"name": name, "tok_s": m, "sd": s, "first10": gens[:10]})

print("\n=== SUMMARY ===", flush=True)
base = results[0]["tok_s"] if results else 1.0
for r in results:
    print(f"  {r['name']:35s}  {r['tok_s']:6.2f} tok/s  ({r['tok_s']/base:.2f}x)  match0? {r['first10'] == results[0]['first10']}", flush=True)
