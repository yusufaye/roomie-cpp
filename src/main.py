import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'

import sys
import json
import asyncio
import logging
import argparse

from manager.worker import WorkerEngine

logging.basicConfig(
  format="%(asctime)s %(levelname)s: %(message)s",
  level=logging.DEBUG,
  datefmt="%H:%M:%S",
  stream=sys.stderr,
  )

# For server logs
logging.getLogger("websockets.server").setLevel(logging.ERROR)

# For client logs
logging.getLogger("websockets.client").setLevel(logging.ERROR)

def main():
  parser = argparse.ArgumentParser(description="Execute video analytics processing pipeline.")
  parser.add_argument("config", metavar="config", type=str, help="Script to run on the server side")
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
  
  filename = str(args["config"])
  logging.debug("Loading configuration {}".format(filename))
  with open(filename, "r") as f:
    config = json.loads(f.read())
    
  engine = WorkerEngine()
  engine.configure(config=config)
  
  asyncio.get_event_loop().run_until_complete(engine.start()) # For Python 3.8.10
  # asyncio.run(engine.start())
  
if __name__== "__main__":
  try:
    main()
  except asyncio.exceptions.CancelledError:
    pass