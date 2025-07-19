data_trace=(synthetic-data twitter)
if [[ " ${data_trace[@]} " =~ " $1 " ]]; then
  data_trace=$1
else
  echo "[ERROR] No match found for the given data trace."
  exit
fi

CONTAINER_NAME="container-cudatorch"
WORKDIR="/home/yfaye/roomie"

QPS=(12000)
APPROACHES=(infaas usher roomie_heuristic_v2)

WORKER_HOSTS=(chuc-3 chuc-5 chuc-8)
CONTROLLER_HOSTS=(chirop-4 chirop-5)

DURATION=5 # minutes

trace_directory="data/${data_trace}"

stop_all_processes() {
  # Stop workers, by killing all running processes, i.e., kill -9 $(ps -ef | grep -E '[a-zA-Z_-]+.py' | awk '{print $2}')
  for host in "${WORKER_HOSTS[@]}"; do
    pid=$(ssh $host docker exec $CONTAINER_NAME ps -ef | grep -E '[a-zA-Z_-]+.py' | awk '{print $2}')
    if [[ -n "$pid" ]]; then
      ssh $host docker exec $CONTAINER_NAME kill -9 $pid 2>/dev/null
    fi
    pid=$(ssh $host ps -ef | grep trace_gpu_perfs | awk '{print $2}')
    if [[ -n "$pid" ]]; then
      ssh $host kill -9 $pid 2>/dev/null
    fi
  done
  # Stop the , by killing all running processes.
  for host in "${CONTROLLER_HOSTS[@]}"; do
    pid=$(ssh $host docker exec $CONTAINER_NAME ps -ef | grep -E '[a-zA-Z_-]+.py' | awk '{print $2}')
    if [[ -n "$pid" ]]; then
      ssh $host docker exec $CONTAINER_NAME kill -9 $pid 2>/dev/null
    fi
  done
}

progress_bar() {
  # arg duration in seconds.
  local duration=$1
  local start_time=$(date +%s)
  while [ $(($(date +%s) - start_time)) -lt $duration ]; do
    local elapsed_time=$(($(date +%s) - start_time))
    local progress=$((elapsed_time * 100 / duration))
    local remaining_time=$((duration - elapsed_time))
    local hours=$((remaining_time / 3600))
    local minutes=$((remaining_time % 3600 / 60))
    local seconds=$((remaining_time % 60))
    printf "\rProgress: %d%% [%s] - Elapsed: %02d:%02d - Remaining: %02d:%02d" $progress $(printf "%*s" $progress | tr ' ' '#') $((elapsed_time % 3600 / 60)) $((elapsed_time % 60)) $minutes $seconds
    sleep 0.1
  done
  printf "\n"
}

function handler() 
{
  echo "Aborted execution"
  echo "Will terminate all running processes and exit"
  stop_all_processes
  exit 1
}

trap 'handler' EXIT

cd $WORKDIR

# Build argument options to generate the configuration.
worker_host_args=""
for item in "${WORKER_HOSTS[@]}"; do worker_host_args="$worker_host_args -w $item,nvidia_a100-sxm4-40gb:4"; done
controller_host_args="-m ${CONTROLLER_HOSTS[0]}"
query_host_args="-q ${CONTROLLER_HOSTS[1]}"

experiment_date=$(date +'%B_%d_%Y_%H%M')
# Collect PID of processes to wait for.
pids=()

for qps in "${QPS[@]}"; do
  echo "--- Query per second $qps ---"
  
  stop_all_processes
  for approach in "${APPROACHES[@]}"; do
    echo "¤¤¤ About to run experiment for $approach ¤¤¤"
    # Remove the log folder.
    rm -rf logger/$approach

    # Setup the configuration.
    python3 configure.py -c $approach -d $DURATION -p $trace_directory -Q $QPS $controller_host_args $query_host_args $worker_host_args

    # Run workers
    for host in "${WORKER_HOSTS[@]}"; do
      files=($(ls config/$approach/$host*json))
      for file in "${files[@]}"; do
        ssh $host docker exec -w /usmb/roomie $CONTAINER_NAME python3 main.py $file -E &
        pids+=($!)
      done
      # Start GPU performance profiling.
      filename="${host}"
      target="$WORKDIR/logger/$approach/$host"
      ssh $host python3 $WORKDIR/trace_gpu_perfs.py -d $target -f $filename -g dGPU &
    done

    # Run the 
    for host in "${CONTROLLER_HOSTS[@]}"; do
      ssh $host docker exec -w /usmb/roomie $CONTAINER_NAME python3 main.py config/$approach/$host.json -E &
      pids+=($!)
    done

    start_time=$(date +%s)
    sleep ${DURATION}m
    # progress_bar $((DURATION * 60))
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    minutes=$((elapsed_time / 60))
    seconds=$((elapsed_time % 60))
    echo "[$approach] - Elapsed time: $minutes minutes $seconds seconds"
    
    # Terminate all processes
    stop_all_processes
  done

  # Rename logger folder
  target_name="${data_trace}_logger_${experiment_date}_agg${qps}"
  cp -r logger $target_name
done

echo "--- All experiments completed successfully ---"

stop_all_processes