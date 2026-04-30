@echo off
rem Usage: start_tiber_70b_worker.bat <stage_idx> [num_streams]
set STAGE_IDX=%1
set NUM_STREAMS=%2
if "%STAGE_IDX%"=="" goto :err
if "%NUM_STREAMS%"=="" set NUM_STREAMS=1
set STAGE1_SHARD=C:\cascadia\shards_70b_v5_beam_4stage\stage_%STAGE_IDX%
set LISTEN_PORT=19100
set DEVICE=GPU
set LATENCY_MS=0
if "%STAGE_IDX%"=="3" set SEND_TOPK=1
if not "%STAGE_IDX%"=="3" set SEND_TOPK=0
"C:\cascadia\inference\venv\Scripts\python.exe" -u C:\cascadia\scripts\mini_worker_stage1.py 1>C:\cascadia\worker_70b.log 2>C:\cascadia\worker_70b.err
goto :eof
:err
echo USAGE: %~n0 STAGE_IDX [NUM_STREAMS]
exit /b 1
