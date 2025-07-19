import os
import aiocsv
import asyncio
import logging
import aiofiles
import argparse
import platform
from typing import Any, Dict, List


class CSVWriter:
  def __init__(self, directory: str, filename: str, mode: str='a') -> None:
    self.directory = directory
    self.filename = filename
    self.mode = mode
    self.key = 0
    os.makedirs(self.directory, exist_ok=True) # create the path directory if not exist
    self.path = '%s/%s_gpu_trace.csv'%(self.directory, self.filename)
    
  async def write(self, *, rows: List[Dict[str, Any]]=[]) -> None:
    async with aiofiles.open(file=self.path, mode=self.mode) as file:
      self.key += 1
      # open the csv writer
      fieldnames = rows[0].keys()
      writer = aiocsv.AsyncDictWriter(file, delimiter=',', lineterminator='\n', fieldnames=fieldnames)
      if self.key == 1: await writer.writeheader()
      # write all data to the csv file
      await writer.writerows(rows)

class Logger:
  def __init__(self, directory: str, filename: str) -> None:
    self._csv_writer  = CSVWriter(directory=directory, filename=filename)
    self.queue        = asyncio.Queue()
    
  def log(self, row: dict) -> None:
    self.queue.put_nowait(row)
    
  async def write(self):
    size = self.queue.qsize()
    if size == 0:
      return
    rows = [ await self.queue.get() for _ in range(size) ]
    await self._csv_writer.write(rows=rows)



parser = argparse.ArgumentParser(description="Collect Nvidia GPU trace.")
parser.add_argument("-d", "--directory", dest="directory", type=str, required=True, help="Target directory")
parser.add_argument("-f", "--filename", dest="filename", type=str, required=True, help="Filename")
parser.add_argument("-i", "--interval", dest="interval", type=str, default=0.5, help="Interval of time to collect data.")
parser.add_argument("-g", "--gpu_type", dest="gpu_type", type=str, required=True, choices=["iGPU", "dGPU"], help="Profiling choice either for Jetson or for others GPU types.")
parser.add_argument("-D", "--debug", action="store_true", help="Print logs at debug level (default info)")
parser.add_argument("-I", "--info", action="store_true", help="Print logs at info level (default info)")
parser.add_argument("-W", "--warn", action="store_true", help="Print logs at warn level (default info)")
parser.add_argument("-E", "--error", action="store_true", help="Print logs at error level (default info)")

args = vars(parser.parse_args())

if args["debug"]:
  logging.getLogger().setLevel(logging.DEBUG)
elif args["info"]:
  logging.getLogger().setLevel(logging.INFO)
elif args["warn"]:
  logging.getLogger().setLevel(logging.WARN)
elif args["error"]:
  logging.getLogger().setLevel(logging.ERROR)
else:
  logging.getLogger().setLevel(logging.INFO)


directory: str  = args["directory"]
filename: str   = args["filename"]
interval: float = args["interval"]
gpu_type: float = args["gpu_type"]

logger = Logger(directory=directory, filename=filename)




async def logger_loop(freq=5):
  while True:
    await asyncio.sleep(freq)
    if not logger: continue
    await logger.write()

def profiling4dGPU():
  from nvitop import Device, ResourceMetricCollector, collect_in_background
  
  def on_collect(metrics):  # will be called periodically
    logger.log(metrics)
    return True
  
  # Record metrics to the logger in the background every 5 seconds.
  # It will collect 5-second mean/min/max for each metric.
  collect_in_background(
    on_collect, # will be called periodically
    ResourceMetricCollector(Device.cuda.all()),
    interval=.5,
  )

def profiling4iGPU():
  from jtop import jtop

  with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
    while jetson.ok():
      # Read tegra stats
      logger.log(jetson.stats)
      
async def main():
  tracer = None
  if gpu_type == "iGPU":
    tracer = profiling4iGPU
    logging.debug("--- Will starting tracing for iGPU. ---")
  elif gpu_type == "dGPU":
    tracer = profiling4dGPU
    logging.debug("--- Will starting tracing for dGPU. ---")
  else:
    logging.error(">>> The given gpu type is not supported.")
    raise ValueError("The given gpu type is not supported.")
  tasks = [
    asyncio.get_running_loop().run_in_executor(None, tracer),
    logger_loop(),
    ]
  await asyncio.gather(*tasks)
  

if __name__ == "__main__":
  try:
    asyncio.get_event_loop().run_until_complete(main())
  except Exception as e:
    msg = f"=== Error running GPU profiling at host {platform.node()}. ==="
    length = len(msg)
    header = bottom = "="*length
    logging.error(f"\n\n{header}\n{msg}\n{bottom}\n{e}")