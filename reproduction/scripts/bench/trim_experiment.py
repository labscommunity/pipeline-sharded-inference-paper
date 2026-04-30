"""Compare trim_kv strategies:
1. copy-via-numpy (current baseline)
2. set_shape() in-place
"""
import os, time, numpy as np, openvino as ov

TARGET = r"C:\cascadia\models\llama-3.1-8b-int4"

core = ov.Core()
c = core.compile_model(os.path.join(TARGET, "openvino_model.xml"), "GPU")
r = c.create_infer_request()

# Prefill with 64 tokens to build up state
prompt = np.arange(64, dtype=np.int64).reshape(1, -1)
r.infer({
    'input_ids': prompt,
    'attention_mask': np.ones((1, 64), dtype=np.int64),
    'position_ids': np.arange(64, dtype=np.int64).reshape(1, -1),
    'beam_idx': np.zeros(1, dtype=np.int32),
})
states = r.query_state()
print(f'{len(states)} states, shape[0]: {states[0].state.shape}', flush=True)

# Method 1: numpy copy (current)
def trim_numpy_copy(req, keep):
    for sv in req.query_state():
        cur = np.asarray(sv.state.data)
        if cur.ndim == 4 and cur.shape[2] > keep:
            sv.state = ov.Tensor(cur[:, :, :keep, :].copy())


def trim_set_shape(req, keep):
    for sv in req.query_state():
        s = sv.state
        sh = s.shape
        if len(sh) == 4 and sh[2] > keep:
            # Try in-place shape shrink. Shape is [batch, heads, seq, head_dim].
            s.set_shape([sh[0], sh[1], keep, sh[3]])
            # Must re-assign? or is set_shape enough?
            sv.state = s


def trim_via_ov_slice(req, keep):
    # Use ov.Tensor directly via buffer slicing without numpy
    for sv in req.query_state():
        s = sv.state
        sh = s.shape
        if len(sh) == 4 and sh[2] > keep:
            # Create a new tensor with new shape, copy first keep positions
            new_shape = [sh[0], sh[1], keep, sh[3]]
            # np view into state, then slice as view (not copy)
            cur = np.asarray(s.data)
            new_tensor = ov.Tensor(s.element_type, new_shape)
            np.asarray(new_tensor.data)[:] = cur[:, :, :keep, :]
            sv.state = new_tensor


# Benchmark — re-prefill each iteration so state has data
def prefill():
    r.reset_state()
    r.infer({
        'input_ids': prompt,
        'attention_mask': np.ones((1, 64), dtype=np.int64),
        'position_ids': np.arange(64, dtype=np.int64).reshape(1, -1),
        'beam_idx': np.zeros(1, dtype=np.int32),
    })


N = 20
for name, trim_fn in [("numpy_copy", trim_numpy_copy),
                      ("set_shape_only", trim_set_shape),
                      ("ov_tensor_slice", trim_via_ov_slice)]:
    # warmup
    for _ in range(3):
        prefill()
        trim_fn(r, 32)
    # measure
    ts = []
    for _ in range(N):
        prefill()
        t0 = time.perf_counter()
        trim_fn(r, 32)
        dt = time.perf_counter() - t0
        ts.append(dt * 1000)
    print(f'{name}: {np.mean(ts):.2f}ms  sd={np.std(ts):.2f}', flush=True)

# Verify set_shape actually trims the state (check by doing infer and checking output)
prefill()
print(f'Pre-trim state shape: {r.query_state()[0].state.shape}')
trim_set_shape(r, 32)
print(f'Post-trim state shape: {r.query_state()[0].state.shape}')
# Verify we can do a forward on the trimmed state
try:
    out = r.infer({
        'input_ids': np.array([[100]], dtype=np.int64),
        'attention_mask': np.ones((1, 33), dtype=np.int64),
        'position_ids': np.array([[32]], dtype=np.int64),
        'beam_idx': np.zeros(1, dtype=np.int32),
    })
    print(f'Forward after set_shape trim: OK, output shape {r.get_output_tensor(0).shape}')
    print(f'State shape after forward: {r.query_state()[0].state.shape}')
except Exception as e:
    print(f'Forward after set_shape trim: FAIL  {type(e).__name__}: {str(e)[:200]}')
