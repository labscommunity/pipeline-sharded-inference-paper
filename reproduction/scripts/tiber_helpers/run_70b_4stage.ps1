param(
    [int]$NumStreams = 1,
    [int]$K = 10,
    [int]$MaxTokens = 128,
    [int]$WarmupRuns = 1,
    [int]$TimedRuns = 1,
    [string]$Stage1Host = "100.77.178.45",
    [string]$Stage2Host = "100.127.88.82",
    [string]$Stage3Host = "100.75.226.6",
    [int]$Port = 19100,
    [string]$Stage0Shard = "C:\cascadia\shards_70b_v5_beam_4stage\stage_0",
    [string]$DraftModel = "C:\cascadia\models\llama-3.2-1b-int4",
    [string]$Python = "C:\cascadia\inference\venv\Scripts\python.exe",
    [string]$Script = "C:\cascadia\scripts\mini_coord_nstage_spec_mbatch.py"
)
$env:STAGE0_SHARD = $Stage0Shard
$env:DRAFT_MODEL = $DraftModel
$env:STAGE_HOSTS = "${Stage1Host}:${Port},${Stage2Host}:${Port},${Stage3Host}:${Port}"
$env:NUM_STREAMS = "$NumStreams"
$env:K = "$K"
$env:MAX_TOKENS = "$MaxTokens"
$env:WARMUP_RUNS = "$WarmupRuns"
$env:TIMED_RUNS = "$TimedRuns"
Write-Host "STAGE_HOSTS=$env:STAGE_HOSTS  NUM_STREAMS=$env:NUM_STREAMS  K=$env:K  MAX_TOKENS=$env:MAX_TOKENS"
& $Python -u $Script
