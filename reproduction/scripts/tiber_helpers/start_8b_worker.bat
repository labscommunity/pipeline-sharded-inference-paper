@echo off
set STAGE1_SHARD=C:\cascadia\shards_2stage_v5_beam_stage_1
set LISTEN_PORT=19100
set NUM_STREAMS=2
set DEVICE=GPU
set LATENCY_MS=0
set SEND_TOPK=%1
if "%SEND_TOPK%"=="" set SEND_TOPK=0
"C:\cascadia\inference\venv\Scripts\python.exe" -u C:\cascadia\scripts\mini_worker_stage1.py 1>C:\cascadia\worker_8b_derp.log 2>C:\cascadia\worker_8b_derp.err
