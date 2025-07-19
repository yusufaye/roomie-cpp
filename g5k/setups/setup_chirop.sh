if [ $# -ne 1 ]; then
	echo "[ERROR] Please provide job-id"
 	exit 1
fi

job_id=$1
response=$(curl -X GET https://api.grid5000.fr/stable/sites/lille/jobs/$job_id)
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
command="docker run -dt --name container-cudatorch --network host --privileged=true --mount type=bind,source=/home/yfaye,target=/usmb --workdir=/usmb yusufaye/roomie:v22.04ubuntu"
g5k "$command"

echo "=== 3/4. Install dependencies. ==="
command="docker exec -w /usmb/src container-cudatorch pip3 install -r DEPENDENCIES.txt"
g5k "$command"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))
seconds=$((elapsed_time % 60))

echo -e "=== 4/4. Done seting up for hosts: $hostnames \nElapsed time: $minutes min $seconds sec. ==="