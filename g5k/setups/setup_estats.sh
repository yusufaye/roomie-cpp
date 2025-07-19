G5K_HOSTNAMES=(estats-1.toulouse.grid5000.fr estats-2.toulouse.grid5000.fr estats-3.toulouse.grid5000.fr estats-4.toulouse.grid5000.fr estats-5.toulouse.grid5000.fr estats-6.toulouse.grid5000.fr estats-7.toulouse.grid5000.fr estats-8.toulouse.grid5000.fr estats-9.toulouse.grid5000.fr estats-10.toulouse.grid5000.fr estats-11.toulouse.grid5000.fr estats-12.toulouse.grid5000.fr)

g5k() {
  command=$1
  pids=()
  for host in "${G5K_HOSTNAMES[@]}"; do
    ssh root@$host $command &
    pids+=($!)
  done
  wait "${pids[@]}"
}


start_time=$(date +%s)

echo "=== 1/5. Run containers. ==="
command="docker run -dt --name container-cudatorch --network host --runtime nvidia --ipc=host --privileged=true --mount type=bind,source=/home/yfaye,target=/usmb --workdir=/usmb yusufaye/l4t-pytorch:r35.4.1"
g5k "$command"

echo "=== 2/5. Set Jetson mode. ==="
command="nvpmodel -m 0"
g5k "$command"

echo "=== 3/5. Install jetson-stats. ==="
command="sudo pip3 install -U jetson-stats==4.2.9"
g5k "$command"

echo "=== 4/5. Install dependencies. ==="
command="pip3 install -r /home/yfaye/src/DEPENDENCIES.txt"
g5k "$command"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))
seconds=$((elapsed_time % 60))

echo -e "=== 4/4. Done seting up for hosts: $hostnames \nElapsed time: $minutes min $seconds sec. ==="