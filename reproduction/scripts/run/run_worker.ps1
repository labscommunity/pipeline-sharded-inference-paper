param(
    [Parameter(Mandatory=$true)][string]$Shard,
    [int]$ListenPort = 19100,
    [int]$NumStreams = 2,
    [int]$LatencyMs = 0,
    [int]$SendTopK = 0,
    [string]$Python = "C:\Program Files\Python311\python.exe",
    [string]$Script = "C:\cascadia\scripts\mini_worker_stage1.py"
)
# Wrapper that the bench/orchestration scripts (e.g. bench_3stage_sweep.sh, run_70b_tiber_bench.sh)
# invoke over SSH to start a worker. Reads shard path + listen port + per-stream count from CLI;
# all other knobs come through env vars that the worker python script reads.
$env:STAGE1_SHARD = $Shard
$env:LISTEN_PORT = "$ListenPort"
$env:NUM_STREAMS = "$NumStreams"
$env:LATENCY_MS = "$LatencyMs"
$env:SEND_TOPK = "$SendTopK"
& $Python $Script
