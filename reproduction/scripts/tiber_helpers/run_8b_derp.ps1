param(
    [int]$NumStreams = 2,
    [int]$K = 3,
    [int]$MaxTokens = 128,
    [int]$SendTopK = 0,
    [string]$Stage1Host = "100.77.178.45",
    [int]$Port = 19100,
    [string]$Stage0Shard = "C:\cascadia\shards_2stage_v5_beam",
    [string]$DraftModel = "C:\cascadia\models\llama-3.2-1b-int4",
    [string]$Python = "C:\cascadia\inference\venv\Scripts\python.exe",
    [string]$Script = "C:\cascadia\scripts\mini_coord_spec_mbatch.py"
)
$env:STAGE0_SHARD = $Stage0Shard
$env:DRAFT_MODEL = $DraftModel
$env:STAGE1_HOST = $Stage1Host
$env:STAGE1_PORT = "$Port"
$env:NUM_STREAMS = "$NumStreams"
$env:K = "$K"
$env:MAX_TOKENS = "$MaxTokens"
$env:SEND_TOPK = "$SendTopK"
$env:WARMUP_RUNS = "1"
$env:TIMED_RUNS = "1"
Write-Host "STAGE0_SHARD=$env:STAGE0_SHARD STAGE1=${env:STAGE1_HOST}:${env:STAGE1_PORT} NUM_STREAMS=$env:NUM_STREAMS K=$env:K SEND_TOPK=$env:SEND_TOPK"
& $Python -u $Script
