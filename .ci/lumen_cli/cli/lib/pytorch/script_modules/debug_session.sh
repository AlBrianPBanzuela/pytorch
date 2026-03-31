#!/bin/bash
set -eu
: "${IDLE_TIMEOUT:?IDLE_TIMEOUT must be set}"

env > /tmp/.lumen_env
echo "cd $PWD && source /tmp/.lumen_env" >> /root/.bashrc
echo 'trap "touch /tmp/.lumen_done" EXIT' >> /root/.bashrc

echo ""
echo "=== Job finished. Container idle for $((IDLE_TIMEOUT / 60))m. ==="
echo "Attach: kubectl exec -it $HOSTNAME -n remote-execution-job-space-beta -- bash"
echo ""

SECONDS=0
while [ ! -f /tmp/.lumen_done ] && [ "$SECONDS" -lt "$IDLE_TIMEOUT" ]; do
    sleep 10
done
echo "Session ended."
