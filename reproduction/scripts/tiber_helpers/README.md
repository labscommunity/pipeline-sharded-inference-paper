# Tiber Cloud helpers

These are the actual launcher scripts used during the §6.8 / §6.10 reproduction
runs on Tiber Cloud, kept for reference and re-runs.

## Files

- `start_70b_worker.bat` — Runs `mini_worker_stage1.py` for one of the 70B 4-stage
  shards. Takes `<stage_idx>` (1, 2, or 3) as the first arg and optional
  `<num_streams>` (default 1) as the second. Stage_3 auto-sets `SEND_TOPK=1`.
- `run_70b_4stage.ps1` — Coord-side launcher for the 4-stage 70B K-sweep,
  runs `mini_coord_nstage_spec_mbatch.py` with the right env vars wired up.
  Run from `cascadia-matias-01` (LL coord) or `cascadia-tate-04` (PL coord).
- `start_8b_worker.bat` — Runs `mini_worker_stage1.py` for the 8B 2-stage stage_1
  shard. Takes `<send_topk>` (0 or 1) as the first arg.
- `run_8b_derp.ps1` — Coord for the 2-node 8B DERP test (§6.8 tab:tiber).

## Why bat instead of inline Start-Process?

We learned the hard way that `Start-Process -WindowStyle Hidden` over SSH does
not detach correctly when invoked through bastion-jumped sessions — the cmd
launcher dies and Python's stdout never makes it to the redirect. Wrapping each
launch in a `.bat` file that lives on the remote box avoids the SSH double-quote
hell entirely; we just `ssh ... 'cmd /c "C:\cascadia\start_*.bat"'` and the
remote cmd holds Python's lifetime properly.

## Python paths

- rainier alpha/charlie/beta: `C:\Program Files\Python311\python.exe` (system).
- Tiber matias-01/02, pawan-01/02, tate-04: `C:\cascadia\inference\venv\Scripts\python.exe`
  (Python 3.14 system install on Tiber doesn't have numpy/openvino — must use the venv).

## Stage_0 path quirk

On Tiber matias-01 the 8B 2-stage stage_0 shard lives **directly** under
`C:\cascadia\shards_2stage_v5_beam` (single dir with `openvino_model.{xml,bin}`),
not inside a `stage_0` subdir. The 70B shards do follow the `stage_<N>` pattern
under `shards_70b_v5_beam_4stage`. The launchers default to the right paths.
