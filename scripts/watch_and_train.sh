#!/bin/bash
# Watch for 200k checkpoint, verify it, then start training
# Checks every 30 seconds for pacman_latents.pt with >= 200k latents

REPO=/teamspace/studios/this_studio/personal/Sana-fork
LATENT_FILE="$REPO/datasets/pacman_latents/pacman_latents.pt"
LOG_FILE="$REPO/output/precompute_stdout.log"
TRAIN_LOG="$REPO/output/pacman_latent/train_stdout.log"
TRIGGER_FILE="$REPO/output/TRIGGER_200K"
CHECK_INTERVAL=30

echo "[watcher] Started at $(date)"
echo "[watcher] Waiting for checkpoint at $LATENT_FILE with >= 200k frames..."

while true; do
    if [ -f "$LATENT_FILE" ]; then
        # Check latent count
        COUNT=$(cd "$REPO" && python -c "
import torch
try:
    d = torch.load('$LATENT_FILE', map_location='cpu', weights_only=False)
    print(d['latents'].shape[0])
except:
    print(0)
" 2>/dev/null)
        
        if [ "$COUNT" -ge 200000 ] 2>/dev/null; then
            echo "[watcher] $(date) - Checkpoint found with $COUNT frames!"
            
            # Write trigger file so the main agent gets notified on next interaction
            echo "$COUNT" > "$TRIGGER_FILE"
            echo "[watcher] Trigger file written to $TRIGGER_FILE"
            echo "[watcher] Waiting 120s for main agent to verify before starting training..."
            
            # Wait for main agent to verify and remove the trigger file,
            # or timeout after 10 minutes and start training anyway
            WAIT_START=$(date +%s)
            while [ -f "$TRIGGER_FILE" ]; do
                sleep 10
                ELAPSED=$(( $(date +%s) - WAIT_START ))
                if [ $ELAPSED -ge 600 ]; then
                    echo "[watcher] $(date) - Timeout waiting for verification. Starting training."
                    rm -f "$TRIGGER_FILE"
                    break
                fi
            done
            
            # Delete stale pycache
            find "$REPO" -name '*.pyc' -delete 2>/dev/null
            find "$REPO" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
            
            # Start training
            cd "$REPO" && CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps PYTHONPATH=. nohup bash train_scripts/train_pacman.sh > "$TRAIN_LOG" 2>&1 &
            TRAIN_PID=$!
            echo "[watcher] Training started with PID $TRAIN_PID"
            echo "[watcher] Training log: $TRAIN_LOG"
            
            # Wait 60s and verify training started
            sleep 60
            if ps -p $TRAIN_PID > /dev/null 2>&1; then
                echo "[watcher] $(date) - Training is running (PID $TRAIN_PID alive after 60s)"
                tail -3 "$REPO/output/pacman_latent/train_log.log" 2>/dev/null
            else
                echo "[watcher] $(date) - WARNING: Training process died within 60s!"
                tail -20 "$TRAIN_LOG" 2>/dev/null
            fi
            exit 0
        else
            echo "[watcher] $(date) - Checkpoint exists but only $COUNT frames (need 200k)"
        fi
    else
        # Check precompute is still running
        if ! pgrep -f precompute_vae_latents > /dev/null 2>&1; then
            echo "[watcher] $(date) - WARNING: Precompute process not found!"
            tail -10 "$LOG_FILE" 2>/dev/null
            echo "[watcher] $(date) - Attempting to restart precomputation..."
            cd "$REPO" && CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps PYTHONPATH=. nohup python scripts/precompute_vae_latents.py --output local --save-dir ./datasets/pacman_latents --batch-size 512 --dtype fp16 > "$LOG_FILE" 2>&1 &
            echo "[watcher] Restarted precompute with PID $!"
        fi
    fi
    sleep $CHECK_INTERVAL
done
