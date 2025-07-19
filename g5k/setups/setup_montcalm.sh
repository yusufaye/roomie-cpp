if [ $# -ne 1 ]; then
	echo "[ERROR] Please provide job-id"
 	exit 1
fi

job_id=$1
response=$(curl -X GET https://api.grid5000.fr/stable/sites/toulouse/jobs/$job_id)
hostnames=$(echo "$response" | jq '.assigned_nodes' | jq -r '.[]' | tr ',' '\n')
for host in $hostnames; do
  echo ">>> $host"
done

g5k() {
  command=$1
  pids=()
  for host in $hostnames; do
    ssh $host $command &
    pids+=($!)
  done
  wait "${pids[@]}"
}

start_time=$(date +%s)

echo "=== 1/4. Setup docker. ==="
command="g5k-setup-docker -t"
g5k "$command"

echo "=== 2/4. Run containers. ==="
command="docker run -dt --name container-cudatorch --network host --privileged=true --mount type=bind,source=/home/yfaye,target=/usmb --workdir=/usmb yusufaye/debian11-python:v3.9.2"
g5k "$command"

echo "=== 3/4. Set Jetson mode. ==="
command="docker exec container-cudatorch pip3 install numpy==1.24.4"
g5k "$command"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))
seconds=$((elapsed_time % 60))

echo -e "=== 4/4. Done seting up for hosts: $hostnames \nElapsed time: $minutes min $seconds sec. ==="