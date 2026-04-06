#!/bin/bash

# A script to run different models and nlayers cases in independent tmux sessions/windows
# Models: dpa, les, sog
# Nlayers: 1, 3, 6

SESSION_NAME="dp_nlayers_2"
TMUX_CMD="tmux -2"
RUN_ID=$(date +%Y%m%d_%H%M%S)
MARKER_DIR="/tmp/${SESSION_NAME}_${RUN_ID}_done"
TASK_MARKERS=()
TASK_COUNT=0

mkdir -p "$MARKER_DIR"

# Ensure tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux is not installed."
    exit 1
fi

# Check if session exists
$TMUX_CMD has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
    # Create new detached session
    $TMUX_CMD new-session -d -s $SESSION_NAME
    echo "Created new tmux session: $SESSION_NAME"
else
    echo "Using existing tmux session: $SESSION_NAME"
fi

MODELS=("dpa" "les" "sog")
#NLAYERS=("1" "3" "6")
NLAYERS=("4" "5")
NUMB_STEPS=500000

for model in "${MODELS[@]}"; do
    for nl in "${NLAYERS[@]}"; do
        WINDOW_NAME="${model}-${nl}"
        
        # Check if window already exists in the session
        if $TMUX_CMD list-windows -t $SESSION_NAME | grep -q "${WINDOW_NAME}"; then
            echo "Window $WINDOW_NAME already exists, skipping."
            continue
        fi

        # Assign GPU based on model
        if [ "$model" == "dpa" ]; then
            GPU_ID=5
        elif [ "$model" == "les" ]; then
            GPU_ID=6
        elif [ "$model" == "sog" ]; then
            GPU_ID=7
        else
            GPU_ID=0 # fallback
        fi

        # Create window
        $TMUX_CMD new-window -t $SESSION_NAME -n $WINDOW_NAME
        MARKER_FILE="${MARKER_DIR}/${WINDOW_NAME}.done"
        TASK_MARKERS+=("$MARKER_FILE")
        TASK_COUNT=$((TASK_COUNT + 1))
        
        # We cd to the correct directory first to avoid relative path issues seen in data paths
        # Auto-create output dir and write a completion marker so session can be auto-killed when all tasks finish.
        CMD="set -o pipefail && mkdir -p /data/zyjin/dp_pt/dp_example/nacl/output/nlayers_${nl} && cd /data/zyjin/dp_pt/dp_example/nacl/output/nlayers_${nl} && source /home/zyjin/anaconda3/etc/profile.d/conda.sh && conda activate dp_devel && export CUDA_VISIBLE_DEVICES=${GPU_ID} && PYTHONPATH=/data/zyjin/dp_pt/dp_devel/deepmd-kit-devel PYTHON_BIN=/home/zyjin/anaconda3/envs/dp_devel/bin/python DP_CMD=\"/home/zyjin/anaconda3/envs/dp_devel/bin/dp --pt train\" /data/zyjin/dp_pt/dp_example/nacl/deepmd_input/run_nlayers_cases.sh nlayers --model ${model} --nlayers-list ${nl} --run --numb-steps ${NUMB_STEPS} 2>&1 | tee train_log_${model}_nl${nl}.txt; status=\$?; touch ${MARKER_FILE}; exit \$status"
        
        $TMUX_CMD send-keys -t ${SESSION_NAME}:${WINDOW_NAME} "$CMD" C-m
        echo "Started ${model} with ${nl} layers on GPU ${GPU_ID} in window ${WINDOW_NAME}"
    done
done

if [ "$TASK_COUNT" -gt 0 ]; then
    MARKER_LIST="${TASK_MARKERS[*]}"
    MONITOR_CMD="while true; do done_count=0; for marker in ${MARKER_LIST}; do [ -f \"\$marker\" ] && done_count=\$((done_count+1)); done; if [ \"\$done_count\" -ge ${TASK_COUNT} ]; then echo \"All submitted tasks finished. Killing session ${SESSION_NAME}.\"; tmux -2 kill-session -t ${SESSION_NAME}; break; fi; sleep 10; done"
    $TMUX_CMD run-shell -t $SESSION_NAME -b "bash -lc '$MONITOR_CMD'"
    echo "Auto-kill monitor started for ${TASK_COUNT} task(s)."
else
    echo "No new tasks were submitted; auto-kill monitor was not started."
fi

echo "All tasks submitted to tmux."
echo "Use 'tmux attach -t $SESSION_NAME' to view the progress."
