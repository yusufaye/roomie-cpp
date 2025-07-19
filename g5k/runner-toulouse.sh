data_trace=(synthetic-data twitter)
if [[ " ${data_trace[@]} " =~ " $1 " ]]; then
  data_trace=$1
else
  echo "[ERROR] No match found for the given data trace."
  exit
fi

CONTAINER_NAME="container-cudatorch"
WORKDIR="/home/yfaye/roomie"

QPS=(400 800 1000 2000 4000)
APPROACHES=(infaas usher roomie_heuristic_v2)

WORKER_HOSTS=(estats-1.toulouse.grid5000.fr estats-2.toulouse.grid5000.fr estats-3.toulouse.grid5000.fr estats-4.toulouse.grid5000.fr estats-5.toulouse.grid5000.fr estats-6.toulouse.grid5000.fr estats-7.toulouse.grid5000.fr estats-8.toulouse.grid5000.fr estats-9.toulouse.grid5000.fr estats-10.toulouse.grid5000.fr estats-11.toulouse.grid5000.fr estats-12.toulouse.grid5000.fr)

CONTROLLER_HOSTS=(montcalm-5.toulouse.grid5000.fr montcalm-6.toulouse.grid5000.fr)

DURATION=5 # minutes

trace_directory="data/${data_trace}"

stop_all_processes() {
  # Stop workers, by killing all running processes, i.e., kill -9 $(ps -ef | grep -E '[a-zA-Z_-]+.py' | awk '{print $2}')
  for host in "${WORKER_HOSTS[@]}"; do
    pid=$(ssh root@$host docker exec $CONTAINER_NAME ps -ef | grep -E '[a-zA-Z_-]+.py' | awk '{print $2}')
    if [[ -n "$pid" ]]; then
      ssh root@$host docker exec $CONTAINER_NAME kill -9 $pid 2>/dev/null
    fi
    pid=$(ssh root@$host ps -ef | grep trace_gpu_perfs | awk '{print $2}')
    if [[ -n "$pid" ]]; then
      ssh root@$host kill -9 $pid 2>/dev/null
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
for item in "${WORKER_HOSTS[@]}"; do worker_host_args="$worker_host_args -w $item,xavier:1"; done
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
        ssh root@$host docker exec -w /usmb/roomie $CONTAINER_NAME python3 main.py $file -I &
        pids+=($!)
      done
      # Start GPU performance profiling.
      filename="${host}"
      target="$WORKDIR/logger/$approach/$host"
      ssh root@$host python3 $WORKDIR/trace_gpu_perfs.py -d $target -f $filename -g iGPU &
    done

    # Run the 
    for host in "${CONTROLLER_HOSTS[@]}"; do
      ssh $host docker exec -w /usmb/roomie $CONTAINER_NAME python3 main.py config/$approach/$host.json -I &
      pids+=($!)
    done

    start_time=$(date +%s)
    # progress_bar $((DURATION * 60))
    sleep ${DURATION}m
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