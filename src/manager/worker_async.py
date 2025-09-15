import os
import time
import queue
import aiocsv
import logging
import asyncio
import logging
import aiofiles
import warnings

import concurrent
try:
  import torch
except: pass
from typing import Dict, List, Tuple

from networking.message import Message
from networking.async_port import OutPort, InPort
warnings.simplefilter(action="ignore", category=FutureWarning)


class Logger:
  def __init__(self, header: List[str], directory: str, filename: str) -> None:
    self.filename   = filename
    self.directory  = directory
    self.header = header
    os.makedirs(self.directory, exist_ok=True)
    self.path = "%s/%s"%(self.directory, self.filename)
    self.queue = asyncio.Queue()
    
  def log(self, row: dict) -> None:
    self.queue.put_nowait(row)
    
  async def write(self, write_header=False):
    async with aiofiles.open(file=self.path, mode="a") as file:
      writer = aiocsv.AsyncDictWriter(file, delimiter=",", lineterminator="\n", fieldnames=self.header)
      if write_header:
        await writer.writeheader()
      size = self.queue.qsize()
      if size == 0: return
      rows = [ await self.queue.get() for _ in range(size) ]
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
    self.tasks: Dict[int, asyncio.Task] = {} # -> variant_id
    self.variants: Dict[int, Model]   = {}
    self.num_received: Dict[int, int] = {}
    self.deployment_queue = asyncio.Queue()
    self.stop_queue = asyncio.Queue()
    self.inference_tracer = None
    self.directory  = "logger"
    self.interval   = 5 # seconds
    self.outgoing: List[OutPort]  = []
    self.incoming: List[InPort]   = []
    self.queue = asyncio.Queue()
  
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
    self.gpu_name = torch.cuda.get_device_name(self.device).lower()
    self.hardware_platform = "_".join(self.gpu_name.split(" "))
    
  async def push(self, msg: Message):
    await self.queue.put(msg)
      
  def get_memory(self) -> Tuple[int, int]:
    free_memory, total_memory = torch.cuda.mem_get_info()
    if torch.cuda.is_available():
      if "nvidia" in self.gpu_name:
        return free_memory, total_memory
      else:
        return free_memory // 2, total_memory // 3
    else:
      logging.warning("GPU is not available")
      return 0, 0
    
  async def start(self):
    while True:
      msg: Message = await self.queue.get()
      if msg.type == "QUERY":
        if msg.data["variant_id"] in self.inference_queue.keys():
          self.inference_queue[msg.data["variant_id"]].put_nowait(msg)
          self.num_received[msg.data["variant_id"]] += msg.data["batch_size"]
        self.ingress_tracer.log({
          "timestamp": time.time(),
          "controller_timestamp": msg.timestamp,
          "query_gen_timestamp": msg.data["query_gen_timestamp"],
          "worker_timestamp": msg.data["worker_timestamp"],
          "worker_id": self.id,
          "variant_id": msg.data["variant_id"],
          "batch_size": msg.data["batch_size"],
          })
      elif msg.type == "DEPLOY":
        await self.deployment_queue.put(msg)
      elif msg.type == "STOP":
        await self.stop_queue.put(msg)
      elif msg.type == "HELLO":
        logging.debug(f"👉[WORKER] Hello messge received: {msg}")
        self.id = msg.data["worker_id"]
        self.engine_name = f"WorkerEngine-{self.id}"
        free_memory, total_memory = self.get_memory()
        logging.debug(f"Total memory: {total_memory / (1024.0 * 1024)} MB | Free memory: {free_memory / (1024.0 * 1024)} MB")
        # Get properties of the first CUDA device
        device_props = torch.cuda.get_device_properties(self.device)
        self.major = device_props.major
        self.minor = device_props.minor
        self.total_mem = total_memory
        hello_msg = Message (0, "HELLO", {"worker_id": self.id, "hardware_platform": self.hardware_platform, "gpu_name": self.gpu_name, "total_mem": self.total_mem, "major": self.major , "minor": self.minor})
        await self.outgoing[0].send(hello_msg)
        self.ingress_tracer = Logger(
          header=[ "timestamp", "controller_timestamp", "query_gen_timestamp", "worker_timestamp", "worker_id", "variant_id", "batch_size" ],
          directory=self.config["parameters"]["log_dir"],
          filename="worker-ingress-trace-{}.csv".format(self.id),
          )
        await self.ingress_tracer.write(write_header=True)
        self.inference_tracer = Logger(
          header=[ "timestamp", "query_gen_timestamp", "worker_timestamp", "worker_id", "variant_id", "name", "throughput", "batch_size", "qsize", ],
          directory=self.config["parameters"]["log_dir"],
          filename="worker-inference-{}.csv".format(self.id),
          )
        await self.inference_tracer.write(write_header=True)
      else:
        ValueError("Unknown the given value of the type -> {}.".format(msg.type))
    
  async def monitor_incoming_data(self):
    while True:
      try:
        input_rate = {}
        for key, num_recv in self.num_received.items():
          input_rate[key] = num_recv
        await asyncio.sleep(1)
        for key, num_recv in input_rate.items():
          self.variants[key].input_rates[1:]  = self.variants[key].input_rates[0:-1]
          self.variants[key].input_rates[0]   = self.num_received[key] - num_recv
          self.variants[key].qsize            = self.get_qsize(key)
      except: pass
  
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
            "input_rates": variant.input_rates,
            "qsize": variant.qsize,
          })
        data = { "worker_id": self.id, "variants": data }
        msg = Message(0, "PROFILE_DATA", data)
        await self.outgoing[0].send(msg)
        logging.debug(f"👉[WORKER] Monitoring with {data}")
    except Exception as e:
      logging.error(f"⛔️ Error with monitor daemon\n\t{e}")
  
  async def logger_daemon(self):
    while True:
      await asyncio.sleep(5)
      if self.inference_tracer == None: continue
      await self.inference_tracer.write()
  
  async def stop_daemon(self):
    try:
      while True:
        msg: Message = await self.stop_queue.get()
        key = msg.data["variant_id"]
        if key not in self.tasks.keys():
          # not deployed yet
          await asyncio.sleep(2)
          await self.stop_queue.put(msg)
        else:
          self.inference_queue[key].put_nowait(None)
          self.tasks[key].cancel()
          logging.debug(f"⚠️ About to stop variant {key}")
          try:
            if key in self.inference_queue.keys():
              del self.inference_queue[key]
            if key in self.tasks.keys():
              del self.tasks[key]
            if key in self.num_received.keys():
              del self.num_received[key]
            if key in self.variants.keys():
              del self.variants[key]
          except: pass
    except Exception as e:
      logging.error(f"⛔️ Error while trying to stop variant\n\t{e}")
  
  async def deployment_daemon(self):
    try:
      while True:
        msg: Message = await self.deployment_queue.get()
        model = Model(id=int(msg.data["id"]), name=msg.data["name"], batch_size=int(msg.data["batch_size"]))
        self.variants[model.id] = model
        self.num_received[model.id] = 0
        self.inference_queue[model.id] = queue.Queue()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.tasks[model.id] = asyncio.get_running_loop().run_in_executor(executor, self.run_inference, model.id)
        # self.tasks[model.id] = asyncio.get_running_loop().run_in_executor(None, self.run_inference, model.id)
        # msg = Message(0, "DEPLOYED", { "worker_id": self.id, "free_memory": free_memory, "total_memory": total_memory })
        # await self.outgoing[0].send(msg)
    except Exception as e:
      logging.error(f"⛔️ Error on deploying daemon\n\t{e}")
      
  def run_inference(self, key: int):
    try:
      stream = torch.cuda.Stream(device=self.device)
      module: torch.nn.Module = torch.load("/usmb/roomie/data/models/{}.pth".format(self.variants[key].name))
      module = module.eval().cuda(device=self.device).to(torch.float32)
      # free_memory, _ = self.get_memory()
      # logging.debug(f"⚠️ [worker] New deployment on Worker-{self.id} device {self.device}\n\t| Name: {self.variants[key].name}\n\t| Batch-size: {self.variants[key].batch_size}\n\t| Free-memory: {free_memory / (1024.0 * 1024)} MB")
      with torch.cuda.stream(stream):
        input_tensor = torch.randn(size=(self.variants[key].batch_size, 3, 224, 224), device=self.device).to(torch.float32)
        with torch.no_grad():
          while True:
            msg: Message = self.inference_queue[key].get()
            if msg == None:
              return
            elapsed = time.time()
            try:
              module(input_tensor)
            except RuntimeError as e: pass
            elapsed = time.time() - elapsed
            self.variants[key].throughput = self.variants[key].batch_size / elapsed
            self.inference_tracer.log({
              "timestamp": time.time(),
              "query_gen_timestamp": msg.data["query_gen_timestamp"],
              "worker_timestamp": msg.data["worker_timestamp"],
              "name": self.variants[key].name,
              "worker_id": self.id,
              "variant_id": msg.data["variant_id"],
              "batch_size": msg.data["batch_size"],
              "throughput": self.variants[key].throughput,
              "qsize": self.get_qsize(key),
              })
    except Exception as e:
      if isinstance(e, KeyError):
        pass
      elif isinstance(e, RuntimeError):
        logging.error(f"⛔️ [{key}] Error during inference, remaining {[item.id for item in self.variants.values()]}\n\t{type(e)}\n\t{e}")
        exit(1)
      else:
        logging.error(f"⛔️ Unknown type of exception\n\t{type(e)}\n\t{e}")
        exit(1)
  
  def get_qsize(self, key: int) -> int:
    return self.inference_queue[key].qsize() * self.variants[key].batch_size
  
  async def ingress_logger_daemon(self):
    while True:
      await asyncio.sleep(5)
      if self.ingress_tracer == None: continue
      await self.ingress_tracer.write()
  
  async def inference_logger_daemon(self):
    while True:
      await asyncio.sleep(5)
      if self.inference_tracer == None: continue
      await self.inference_tracer.write()
  
  async def run(self):
    tasks = [ self.start(), self.ingress_logger_daemon(), self.inference_logger_daemon(), self.monitor_incoming_data(), self.monitor_daemon(), self.deployment_daemon(), self.stop_daemon() ]
    for port in self.incoming + self.outgoing:
      tasks += port.get_runners()  
    
    try:
      await asyncio.gather(*tasks)
    except Exception as e:
      logging.error("=== RUN ERROR ===\n{}".format(e))
      # os.remove(self.directory)
      raise e