import logging
import asyncio
import websockets

from networking.message import Message



class OutPort:
  def __init__(self, remote_host, remote_port):
    logging.debug(f"[OutPort] Remote host: {remote_host}, remote port: {remote_port}")
    self.remote_host = remote_host
    self.remote_port = remote_port
    self.uri = f"ws://{self.remote_host}:{self.remote_port}"
    self.queue = asyncio.Queue()

  async def send(self, msg: str):
    await self.queue.put(msg)

  async def connect(self):
    while True:
      try:
        async with websockets.connect(self.uri) as websocket:
          logging.debug(f"✅[OutPort] Connected successfully to {self.uri}!")
          try:
            while True:
              msg: Message = await self.queue.get()
              message = msg.marshal()
              await websocket.send(message)
          except Exception as e:
            logging.error(f"⛔️ Connection lost to {self.uri}\n\t{e}")
            break
      except (ConnectionRefusedError, OSError) as e:
          logging.error(f"⛔️[OutPort] Connection failed to host {self.remote_host} and port {self.remote_port}. Retrying in 3 seconds...")
          await asyncio.sleep(3)
    
  
  def get_runners(self):
    return [ self.connect() ]


class InPort:
  def __init__(self, host, port, callback):
    logging.debug(f"[InPort] Host: {host}, port: {port}")
    self.host     = "" # host don't set the host.
    self.port     = port
    self.callback = callback

  async def handler(self, websocket):
    try:
      async for message in websocket:
        msg = Message()
        msg.unmarshal(message)
        await self.callback(msg)
    except Exception as e:
      logging.error(f"⛔️[InPort] Error while handling message\n\t{e}")

  async def connect(self):
    try:
      async with websockets.serve(self.handler, self.host, self.port):
        logging.debug(f"✅[InPort] Client connected to ws://{self.host}:{self.port}.")
        await asyncio.Future()  # run forever
    except Exception as e:
      logging.error(f"⛔️[InPort] Connection failed to host {self.host} and port {self.port}\n{e}")
      

  def get_runners(self):
    return [ self.connect() ]
