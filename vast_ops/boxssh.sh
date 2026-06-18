#!/bin/bash
# SSH/SCP helpers for the active vast box. Source me: `source vast_ops/boxssh.sh`
export VAST_HOST=ssh7.vast.ai
export VAST_PORT=20086
export VAST_KEY=/home/debian/.ssh/id_ed25519
SSHK="-p $VAST_PORT -i $VAST_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/home/debian/.ssh/known_hosts -o ConnectTimeout=20 -o ServerAliveInterval=30"
SCPK="-P $VAST_PORT -i $VAST_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/home/debian/.ssh/known_hosts"
box()  { ssh $SSHK root@$VAST_HOST "$@"; }
boxcp() { scp $SCPK "$@"; }
