#!/bin/bash
# Watch for precompute completion (1.2M frames), then restart training with Map-style dataset
REPO=/teamspace/studios/this_studio/personal/Sana-fork
LATENT_FILE="$REPO/datasets/pacman_latents/pacman_latents.pt"
PRECOMPUTE_LOG="$REPO/output/precompute_stdout.log"
TRAIN_LOG="$REPO/output/pacman_latent/train_stdout.log"
TRIGGER_FILE="$REPO/output/TRIGGER_PRECOMP_DONE"
CHECK_INTERVAL=60

echo "[watcher] Started at $(date)"
echo "[watcher] Waiting for precompute to finish (1.2M frames)..."

while true; do
    # Check if precompute finished (look for "Done!" in log)
    if grep -q "^Done!" "$PRECOMPUTE_LOG" 2>/dev/null; then
        echo "[watcher] $(date) - Precompute finished!"
        
        # Verify checkpoint
        COUNT=$(cd "$REPO" && python -c "
import torch
d = torch.load('$LATENT_FILE', map_location='cpu', weights_only=False)
print(d['latents'].shape[0])
" 2>/dev/null)
        echo "[watcher] Latent count: $COUNT"
        
        if [ "$COUNT" -ge 1199000 ] 2>/dev/null; then
            echo "[watcher] Checkpoint verified — $COUNT frames"
            echo "$COUNT" > "$TRIGGER_FILE"
            echo "[watcher] Trigger file written. Waiting for main agent verification..."
            
            # Wait up to 10 min for verification
            WAIT_START=$(date +%s)
            while [ -f "$TRIGGER_FILE" ]; do
                sleep 10
                ELAPSED=$(( $(date +%s) - WAIT_START ))
                if [ $ELAPSED -ge 600 ]; then
                    echo "[watcher] Timeout. Proceeding to restart training."
                    rm -f "$TRIGGER_FILE"
                    break
                fi
            done
            
            # Stop current training
            echo "[watcher] Stopping current training..."
            pkill -f train_pacman 2>/dev/null
            sleep 5
            
            # Delete pycache
            find "$REPO" -name '*.pyc' -delete 2>/dev/null
            find "$REPO" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
            
            # Start training with new config
            echo "[watcher] Starting training with Map-style dataset..."
            cd "$REPO" && CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps PYTHONPATH=. nohup bash train_scripts/train_pacman.sh > "$TRAIN_LOG" 2>&1 &
            TRAIN_PID=$!
            echo "[watcher] Training PID: $TRAIN_PID"
            
            sleep 120
            if ps -p $TRAIN_PID > /dev/null 2>&1; then
                echo "[watcher] Training running after 120s"
                tail -5 "$REPO/output/pacman_latent/train_log.log" 2>/dev/null
            else
                echo "[watcher] WARNING: Training died!"
                tail -20 "$TRAIN_LOG" 2>/dev/null
            fi
            exit 0
        else
            echo "[watcher] WARNING: Only $COUNT frames (need ~1.2M). Waiting..."
        fi
    fi
    
    # Check precompute is still running
    if ! pgrep -f precompute_vae > /dev/null 2>&1; then
        if ! grep -q "^Done!" "$PRECOMPUTE_LOG" 2>/dev/null; then
            echo "[watcher] $(date) - WARNING: Precompute died! Restarting..."
            tail -10 "$PRECOMPUTE_LOG" 2>/dev/null
            cd "$REPO" && CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps PYTHONPATH=. nohup python scripts/precompute_vae_latents.py --save-dir ./datasets/pacman_latents --batch-size 32 --dtype fp16 > "$PRECOMPUTE_LOG" 2>&1 &
            echo "[watcher] Restarted precompute PID: $!"
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
