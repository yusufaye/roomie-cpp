import re
import os
import glob
import json
import argparse

from typing import List




parser = argparse.ArgumentParser(description="Generate configuration\n E.g., python -c usher -w estats-1,xavier:4,nvidia_geforce_rtx_2080_ti:2,nvidia_a100")
parser.add_argument("-c", "--config", dest="config", type=str, choices=[ "roomie", "roomie_heuristic", "roomie_heuristic_v2", "usher", "infaas" ], required=True, help="Job id for workers")
parser.add_argument("-w", "--worker-host", dest="worker_hosts", type=str, required=True, action="append", help="Worker host")
parser.add_argument("-m", "--controller-host", dest="controller_host", type=str, required=True, help="Controller host")
parser.add_argument("-q", "--query-host", dest="query_hosts", type=str, required=True, action="append", help="Query generator host")
parser.add_argument("-d", "--duration", dest="duration", type=float, default=10, help="Duration of the simulation in minutes")
parser.add_argument("-Q", "--qps", dest="qps", type=int, required=True, help="Average query per second.")
parser.add_argument("-p", "--path", dest="path", type=str, required=True, help="Trace directory.")
parser.add_argument("-D", "--debug", action="store_true", help="Print logs at debug level (default info)")
parser.add_argument("-I", "--info", action="store_true", help="Print logs at info level (default info)")
parser.add_argument("-W", "--warn", action="store_true", help="Print logs at warn level (default info)")
parser.add_argument("-E", "--error", action="store_true", help="Print logs at error level (default info)")

args = vars(parser.parse_args())

worker_hosts: List[str] = args["worker_hosts"]
controller_host: str    = args["controller_host"]
query_hosts: List[str]  = args["query_hosts"]
config: str             = args["config"]
duration: float         = args["duration"] # minutes
qps: int                = args["qps"] # minutes
path: str               = args["path"] # directory where to find the traces.


class Engine:
  def __init__(self, name: str, type: str, host: str, port: int=8080, parameters: dict={}):
    self.id         = None
    self.name       = name
    self.host       = host
    self.port       = port
    self.type       = type
    self.parameters = parameters
    self.remote_engines = []
    
  def save(self, directory="config"):
    with open("{}/{}.json".format(directory, self.name), "w") as file:
      data = dict(
        id=self.id,
        host=self.host,
        port=self.port,
        type=self.type,
        parameters=self.parameters,
        remote_engines=self.remote_engines,
        )
      json.dump(data, file, indent=2)
  
  def __repr__(self):
    return """Engine(
      'host': {host},
      'port': {port},
      'type': {type},
      'parameters': {parameters},
      'remote_engines': {remote_engines},
    )""".format(host=self.host, port=self.port, type=self.type, parameters=self.parameters, remote_engines=self.remote_engines)

workers: List[Engine] = []
for worker_host in worker_hosts:
  # E.g., estats-1,xavier:4,nvidia_geforce_rtx_2080_ti:2,nvidia_a100
  instances = worker_host.split(',')
  worker_host, hardware_platforms = instances[0], instances[1:]
  options = []
  for hardware_platform in hardware_platforms:
    options += re.findall(r'^([\w\d-]+):?([\d]+)?', hardware_platform)
  for i, (hardware_platform, gpu_devices) in enumerate(options):
    gpu_devices = int(gpu_devices) if gpu_devices else 1
    options[i] = (hardware_platform, gpu_devices)
  for hardware_platform, device in options:
    for device in range(gpu_devices):
      worker = Engine(
        name="{}_{}".format(worker_host, device),
        type="WorkerExecutor",
        host=worker_host,
        port=(8081 + device),
        parameters={
          "log_dir": "logger/{}/{}".format(config, worker_host),
          "use_cuda_stream": ("roomie" in config),
          "hardware_platform": hardware_platform,
          "device": device,
          })
      workers.append(worker)

ControllerParameters = dict(
  infaas=dict(scheduling="INFaaSSchaduling", log_dir="logger/{}".format(config)),
  usher=dict(scheduling="UsherSchaduling", log_dir="logger/{}".format(config)),
  roomie=dict(scheduling="InterferenceAwareScheduling", log_dir="logger/{}".format(config)),
  roomie_heuristic=dict(scheduling="HeuristicInterferenceAwareScheduling", log_dir="logger/{}".format(config)),
  roomie_heuristic_v2=dict(scheduling="HeuristicInterferenceAwareSchedulingV2", log_dir="logger/{}".format(config)),
  )

controller = Engine(name="{}".format(controller_host), type="Controller", host=controller_host, parameters=ControllerParameters[config])
controller.remote_engines = [
  { "remote_host": worker.host, "remote_port": worker.port }
  for worker in workers
]
### Query generators
GeneratorParam = [{
  "duration": duration,
    "domain": [
      "alexnet",
      "squeezenet1_1",
      "shufflenet_v2_x2_0",
      "googlenet",
      "resnet18",
      "inception_v3",
      "densenet121",
      "convnext_tiny",
      "resnet101",
      "mobilenet_v3_large",
      "mobilenet_v2",
      "maxvit_t",
      "efficientnet_v2_l",
      "wide_resnet101_2",
      "convnext_base",
      "convnext_large",
      "fcos_resnet50_fpn",
      "fasterrcnn_resnet50_fpn",
      "retinanet_resnet50_fpn_v2",
      # "ssd300_vgg16",
      ],
    "qps": qps,
    "path": path,
  }
]

query_generators: List[Engine] = []
names = []
for query_gen_host, parameters in zip(query_hosts, GeneratorParam):
  engine = Engine(name="{}".format(query_gen_host), type="PoissonZipfQueryGenerator", host=query_gen_host, port=None, parameters=parameters)
  query_generators.append(engine)
for item in query_generators:
  item.remote_engines = [ { "remote_host": controller.host, "remote_port": controller.port } ]
for item in workers:
  item.remote_engines = [ { "remote_host": controller.host, "remote_port": (controller.port + 1) } ]

# 1. Save and copy worker to the remote devices.
config_dir = "config/{}".format(config)
os.makedirs(config_dir, exist_ok=True)

for item in glob.glob("{}/*.json".format(config_dir)):
  os.remove(item)
engines = [ controller ] + workers + query_generators
for engine in engines:
  engine.save(directory=config_dir)