import os
import csv
import time
import queue
import logging
import logging
import warnings
import threading

try:
  import torch
except: pass
from typing import Dict, List, Tuple

from networking.message import Message
from networking.port import OutPort, InPort
warnings.simplefilter(action="ignore", category=FutureWarning)


class Logger(threading.Thread):
  def __init__(self, header: List[str], directory: str, filename: str) -> None:
    super().__init__()
    self.filename   = filename
    self.directory  = directory
    self.header = header
    os.makedirs(self.directory, exist_ok=True)
    self.path = "%s/%s"%(self.directory, self.filename)
    self.queue = queue.Queue()
    
  def log(self, row: dict) -> None:
    self.queue.put(row)
    
  def run(self):
    with open(file=self.path, mode="a") as file:
      writer = csv.DictWriter(file, delimiter=",", lineterminator="\n", fieldnames=self.header)
      writer.writeheader()
    while True:
      time.sleep(5)
      with open(file=self.path, mode="a") as file:
        writer = csv.DictWriter(file, delimiter=",", lineterminator="\n", fieldnames=self.header)
        qsize = self.queue.qsize()
        if qsize == 0:
          continue
        rows = [ self.queue.get() for _ in range(qsize) ]
        writer.writerows(rows)

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
    self.running: Dict[int, threading.Thread]     = {} # -> variant_id
    self.variants: Dict[int, Model]   = {}
    self.num_received: Dict[int, int] = {}
    self.deployment_queue = queue.Queue()
    self.stop_queue       = queue.Queue()
    self.logger = None
    self.directory  = "logger"
    self.interval   = 5 # seconds
    self.outgoing: List[OutPort]  = []
    self.incoming: List[InPort]   = []
    self.queue = queue.Queue()
  
  def configure(self, config):
    self.id: int = config["id"]
    self.config = config
    self.parameters: dict = self.config["parameters"]
    if "log_dir" in self.parameters.keys():
      self.directory = self.parameters["log_dir"]
    if self.config["host"] and self.config["port"]:
      inport = InPort(host=self.config["host"], port=self.config["port"], callback=self.push)
      self.incoming.append(inport)
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
    
  def push(self, msg: Message):
    self.queue.put(msg)
  
  def start(self):
    while True:
      msg: Message = self.queue.get()
      logging.debug(f"✉️[WORKER] Recv: {msg}")
      if msg.type == "QUERY":
        if msg.data["variant_id"] in self.inference_queue.keys():
          self.inference_queue[msg.data["variant_id"]].put(msg)
          self.num_received[msg.data["variant_id"]] += msg.data["batch_size"]
      elif msg.type == "DEPLOY":
        self.deployment_queue.put(msg)
      elif msg.type == "STOP":
        self.stop_queue.put(msg)
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
        self.outgoing[0].send(hello_msg)
        self.logger = Logger(
          header=["timestamp", "controller_timestamp", "query_gen_timestamp", "worker_timestamp", "worker_id", "variant_id", "name", "throughput", "batch_size", "qsize" ],
          directory=self.config["parameters"]["log_dir"],
          filename="worker-{}.csv".format(self.id),
          )
        self.logger.start()
      else:
        ValueError("Unknown the given value of the type -> {}.".format(msg.type))
    
  def monitor_incoming_data(self):
    while True:
      try:
        input_rate = {}
        for key, num_recv in self.num_received.items():
          input_rate[key] = num_recv
        time.sleep(1)
        for key, num_recv in input_rate.items():
          self.variants[key].input_rates[1:]  = self.variants[key].input_rates[0:-1]
          self.variants[key].input_rates[0]   = self.num_received[key] - num_recv
          self.variants[key].qsize            = self.inference_queue[key].qsize()
      except: pass
  
  def monitor_daemon(self):
    try:
      while True:
        time.sleep(self.interval)
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
        self.outgoing[0].send(msg)
        logging.debug(f"👉[WORKER] Monitoring with {data}")
    except Exception as e:
      logging.error(f"⛔️ Error with monitor daemon\n\t{e}")
  
  def stop_daemon(self):
    try:
      while True:
        msg: Message = self.stop_queue.get()
        key = msg.data["variant_id"]
        if key not in self.running.keys():
          # not deployed yet
          time.sleep(2)
          self.stop_queue.put(msg)
        else:
          self.inference_queue[key].put_nowait(None)
          logging.debug(f"⚠️ About to stop variant {key}")
          try:
            if key in self.inference_queue.keys():
              del self.inference_queue[key]
            if key in self.running.keys():
              del self.running[key]
            if key in self.num_received.keys():
              del self.num_received[key]
            if key in self.variants.keys():
              del self.variants[key]
          except: pass
    except Exception as e:
      logging.error(f"⛔️ Error while trying to stop variant\n\t{e}")
  
  def deployment_daemon(self):
    try:
      while True:
        msg: Message = self.deployment_queue.get()
        model_ = Model(id=int(msg.data["id"]), name=msg.data["name"], batch_size=int(msg.data["batch_size"]))
        self.variants[model_.id] = model_
        self.num_received[model_.id] = 0
        self.inference_queue[model_.id] = queue.Queue()
        task = threading.Thread(target=self.run_inference, kwargs={ "model": model_ })
        task.start()
        self.running[model_.id] = task
        # msg = Message(0, "DEPLOYED", { "worker_id": self.id, "free_memory": free_memory, "total_memory": total_memory })
        # await self.outgoing[0].send(msg)
    except Exception as e:
      logging.error(f"⛔️ Error on deploying daemon\n\t{e}")
      
  def run_inference(self, model: Model):
    try:
      stream = torch.cuda.Stream(device=self.device)
      module: torch.nn.Module = torch.load("/usmb/roomie/data/models/{}.pth".format(model.name))
      module = module.eval().cuda(device=self.device)
      free_memory, _ = self.get_memory()
      logging.debug(f"⚠️ [worker] New deployment on Worker-{self.id} device {self.device}\n\t| Name: {model.name}\n\t| Batch-size: {model.batch_size}\n\t| Free-memory: {free_memory / (1024.0 * 1024)} MB")
      with torch.cuda.stream(stream):
        input_tensor = torch.randn(size=(model.batch_size, 3, 224, 224), device=self.device)
        with torch.no_grad():
          while True:
            msg: Message = self.inference_queue[model.id].get()
            if msg == None:
              return
            elapsed = time.time()
            try:
              module(input_tensor)
            except RuntimeError as e: pass
            elapsed = time.time() - elapsed
            model.throughput = model.batch_size / elapsed
            self.logger.log({
              "timestamp": time.time(),
              "controller_timestamp": msg.timestamp,
              "query_gen_timestamp": msg.data["timestamp"],
              "worker_timestamp": msg.data["worker_timestamp"],
              "worker_id": self.id,
              "variant_id": model.id,
              "name": model.name,
              "throughput": model.throughput,
              "batch_size": msg.data["batch_size"],
              "qsize": self.inference_queue[model.id].qsize(),
              })
            # logging.info(f"{model.name}\tIn-sys: {insys_elapsed:.5f}\tThr: {model.throughput:.0f}\tQsize: {self.inference_queue[model.id].qsize()}")
    except Exception as e:
      if isinstance(e, KeyError):
        pass
      elif isinstance(e, RuntimeError):
        logging.error(f"⛔️ [{model.name}] Error during inference, remaining {[item.id for item in self.variants.values()]}\n\t{type(e)}\n\t{e}")
        exit(1)
      else:
        logging.error(f"⛔️ Unknown type of exception\n\t{type(e)}\n\t{e}")
        exit(1)
  
  def run(self):
    tasks = [ 
      threading.Thread(target=self.monitor_incoming_data, kwargs={}), 
      threading.Thread(target=self.monitor_daemon, kwargs={}), 
      threading.Thread(target=self.start, kwargs={}), 
      threading.Thread(target=self.deployment_daemon, kwargs={}), 
      threading.Thread(target=self.stop_daemon, kwargs={}) ]
    tasks += self.incoming + self.outgoing
    for task in tasks:
      task.start()
    for task in tasks:
      task.join()  