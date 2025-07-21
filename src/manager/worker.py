import os
import time
import queue
import aiocsv
import logging
import asyncio
import logging
import aiofiles
import warnings
try:
  import torch
except: pass
from typing import Dict, List, Any

from networking.message import Message
from networking.port import OutPort, InPort
warnings.simplefilter(action="ignore", category=FutureWarning)


class Logger:
  def __init__(self, directory: str, filename: str) -> None:
    self.filename   = filename
    self.directory  = directory
    os.makedirs(self.directory, exist_ok=True)
    self.path = "%s/%s"%(self.directory, self.filename)
    self.queue = asyncio.Queue()
    
  def log(self, row: dict) -> None:
    self.queue.put_nowait(row)
    
  async def run(self, header: List[str], freq=5, mode: str="w"):
    async with aiofiles.open(file=self.path, mode=mode) as file:
      writer = aiocsv.AsyncDictWriter(file, delimiter=",", lineterminator="\n", fieldnames=header)
      await writer.writeheader()
      while True:
        await asyncio.sleep(freq)
        size = self.queue.qsize()
        if size == 0: continue
        rows: List[Dict[str, Any]] = [ await self.queue.get() for _ in range(size) ]
        await writer.writerows(rows)


class Model:
  def __init__(self, id: int, name: str, batch_size: int):
    self.id = id
    self.name = name
    self.batch_size = batch_size
    self.throughput = 0.0
    self.qsize: int = 0
    self.input_rates = [0.0] * 10

class WorkerEngine:
  def __init__(self):
    super().__init__()
    self.inference_queue: Dict[int, queue.Queue]  = {} # -> variant_id
    self.running_variant: Dict[int, asyncio.Task]  = {} # -> variant_id
    self.variants: Dict[int, Model] = {}
    self.num_received: Dict[int, int] = {}
    self.deployment_queue = asyncio.Queue()
    self.csv_logger   = None
    self.directory    = "logger"
    self.interval     = 5 # seconds
    self.outgoing: List[OutPort]  = []
    self.incoming: List[InPort]   = []
  
  def configure(self, config):
    self.id: int = config["id"]
    self.config = config
    self.parameters: dict = self.config["parameters"]
    if "log_dir" in self.parameters.keys():
      self.directory = self.parameters["log_dir"]
    if self.config["host"] and self.config["port"]:
      self.incoming.append(InPort(host=self.config["host"], port=self.config["port"], callback=self.push))
    for item in self.config["remote_engines"]:
      outport = OutPort(remote_host=item["remote_host"], remote_port=item["remote_port"])
      self.outgoing.append(outport)
    self.hardware_platform: str = self.parameters["hardware_platform"]
    if "device" in self.parameters.keys():
      self.device = int(self.parameters["device"])
    else:
      self.device = 0
    logging.debug(f"👉[WORKER] Given device is {self.device}👈")
    if not torch.cuda.is_available():
      logging.error(">>> CUDA is not available <<<")
      exit()
    
  async def push(self, msg: Message):
    # logging.debug(f"✉️[WORKER] Recv: {msg}")
    if msg.type == "QUERY":
      if msg.data["variant_id"] in self.inference_queue.keys():
        self.inference_queue[msg.data["variant_id"]].put_nowait(msg)
        self.num_received[msg.data["variant_id"]] += msg.data["batch_size"]
    elif msg.type == "DEPLOY":
      await self.deployment_queue.put(msg)
    elif msg.type == "STOP":
      self.inference_queue[msg.data["variant_id"]].put_nowait(None)
    elif msg.type == "HELLO":
      logging.debug(f"👉[WORKER] Hello messge received: {msg}")
      self.id = msg.data["worker_id"]
      self.engine_name = f"WorkerEngine-{self.id}"
      free_memory, total_memory = torch.cuda.mem_get_info()
      logging.debug(f"Total memory: {total_memory / (1024.0 * 1024)} MB | Free memory: {free_memory / (1024.0 * 1024)} MB")
      self.total_mem = total_memory
      hello_msg = Message (0, "HELLO", {"worker_id": self.id, "total_mem": self.total_mem})
      await self.outgoing[0].send(hello_msg)
      self.logger = Logger(self.config["parameters"]["log_dir"], self.id)
      self.logger_task = asyncio.create_task(self.logger.run(header=["timestamp", "worker_id", "variant_id", "name", "throughput", "batch_size"]))
    else:
      ValueError("Unknown the given value of the type -> {}.".format(msg.type))
    
  async def monitor_incoming_data(self):
    while True:
      input_rate = {}
      for key, num_recv in self.num_received.items():
        input_rate[key] = num_recv
      await asyncio.sleep(1)
      for key, num_recv in input_rate.items():
        self.variants[key].input_rates[1:]  = self.variants[key].input_rates[0:-1]
        self.variants[key].input_rates[0]   = self.num_received[key] - num_recv
        self.variants[key].qsize            = self.inference_queue[key].qsize()
  
  async def monitor_daemon(self):
    try:
      while True:
        await asyncio.sleep(self.interval)
        data = []
        for variant in self.variants.values():
          data.append({
            "variant_id": variant.id,
            "variant_name": variant.name,
            "throughput": variant.throughput,
            "input_rate": variant.input_rates,
            "qsize": variant.qsize,
          })
        data = { "worker_id": self.id, "variants": data }
        msg = Message(0, "PROFILE_DATA", data)
        await self.outgoing[0].send(msg)
        logging.info(f"👉[WORKER] Monitoring with {data}")
    except Exception as e:
      logging.error(f"⛔️ Error with monitor daemon\n\t{e}")
  
  async def deployment_daemon(self):
    try:
      while True:
        msg: Message = await self.deployment_queue.get()
        model_ = Model(id=int(msg.data["id"]), name=msg.data["name"], batch_size=int(msg.data["batch_size"]))
        self.variants[model_.id] = model_
        queue_ = queue.Queue()
        self.num_received[model_.id] = 0
        self.inference_queue[model_.id] = queue_
        self.running_variant[model_.id] = asyncio.create_task(asyncio.to_thread(self.run_inference, model_.id))
        # self.running_variant[model_.id] = asyncio.create_task(asyncio.get_running_loop().run_in_executor(None, self.run_inference, model_))
    except Exception as e:
      logging.error(f"⛔️ Error on deploying daemon\n\t{e}")
      
  def run_inference(self, key: int):
    stream = torch.cuda.Stream(device=self.device)
    model: torch.nn.Module = torch.load("/usmb/roomie/data/models/{}.pth".format(self.variants[key].name))
    model = model.eval().cuda(device=self.device)
    free_memory, _ = torch.cuda.mem_get_info()
    logging.info(f"⚠️ [worker] New deployment\n\t| Name: {self.variants[key].name}\n\t| Batch-size: {self.variants[key].batch_size}\n\t| Free-memory: {free_memory / (1024.0 * 1024)} MB")
    try:
      with torch.cuda.stream(stream):
        input_tensor = torch.randn(size=(self.variants[key].batch_size, 3, 224, 224), device=self.device)
        with torch.no_grad():
          while True:
            msg: Message = self.inference_queue[key].get()
            if msg == None:
              return
            elapsed = time.time()
            model(input_tensor)
            self.variants[key].throughput = self.variants[key].batch_size / (time.time() - elapsed)
            logging.info(f"Inference: worker-id={self.id}, id={self.variants[key].id}, name={self.variants[key].name}, thr={self.variants[key].throughput}")
            "timestamp", "worker_id", "variant_id", "name", "throughput", "batch_size"
            self.logger.log({
              "timestamp": time.time(),
              # "start_at": msg.timestamp,
              "worker_id": self.id,
              "variant_id": self.variants[key].id,
              "name": self.variants[key].name,
              "throughput": self.variants[key].throughput,
              "batch_size": self.variants[key].batch_size,
              })
    except Exception as e:
      logging.error(f"⛔️ Error during inference\n\t{e}")
  
  async def start(self):
    tasks = [ self.monitor_incoming_data(), self.monitor_daemon(), self.deployment_daemon() ]
    for port in self.incoming + self.outgoing:
      tasks += port.get_runners()  
    
    try:
      await asyncio.gather(*tasks)
    except Exception as e:
      logging.error("=== RUN ERROR ===\n{}".format(e))
      # os.remove(self.directory)
      raise e