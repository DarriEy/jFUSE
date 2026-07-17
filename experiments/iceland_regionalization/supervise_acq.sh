#!/bin/zsh
# Self-healing supervisor for the 0.01 CARRA 2004-2018 re-acquisition.
# DEATH-ONLY (v4): only relaunch when the acquisition PROCESS EXITS. Does NOT
# kill on stall. Rationale (learned the hard way 2026-07-07): killing a client
# whose CDS request is "accepted"/"running" does NOT cancel the server-side
# request -> orphaned requests pile up and clog the account's CDS concurrency
# (21 orphans stalled everything for 24h). CDS queue waits are normal and can be
# hours; a genuinely hung client self-resolves via the ecmwf client's own retry
# timeout (maximum_tries=500 x retry_after=120s ~= 16h) then exits, and we
# relaunch. The download is resumable (skips months with a valid _processed
# file). Runs until 2018-12 present. Multi-day; nohup detached.
set -u
DOM=/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Iceland_multivar
CFG=$DOM/config_jfuse_prod6_18yr.yaml
SYMF=/Users/darri.eythorsson/compHydro/SYMFLUENCE/venv/bin/symfluence
RAW=$DOM/data/forcing/raw_data
LOG=$DOM/_workLog_Iceland_multivar/acq18yr_resume.log
DONE_MARK=$RAW/Iceland_multivar_CARRA_processed_201812_temp.nc
SUPLOG=$DOM/_workLog_Iceland_multivar/acq_supervisor.log

CHECK_SECS=600  # 10 min

echo "$(date) supervisor(v4 death-only) started (target 2018-12)" >> "$SUPLOG"
while true; do
  if [ -f "$DONE_MARK" ] || ls "$RAW"/Iceland_multivar_CARRA_201812.nc >/dev/null 2>&1; then
    echo "$(date) DONE — 2018-12 present. supervisor exiting." >> "$SUPLOG"; break
  fi
  if ! pgrep -f "workflow steps.*acquire_forcings" >/dev/null 2>&1; then
    NLAST=$(ls "$RAW"/*processed_*_temp.nc 2>/dev/null | grep -oE 'processed_20[0-9]{4}' | sort | tail -1)
    echo "$(date) acquisition not running (last=$NLAST) — relaunching" >> "$SUPLOG"
    nohup $SYMF workflow steps --config "$CFG" acquire_forcings >> "$LOG" 2>&1 &
    sleep 90
  fi
  sleep $CHECK_SECS
done
