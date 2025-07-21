import json
from enum import Enum
import logging
from typing import Dict


class Type(Enum):
  QUERY           = "QUERY" 
  REGISTER        = "REGISTER" 
  PROFILE_DATA    = "PROFILE_DATA" 
  WARMING_DONE    = "WARMING_DONE" 
  DEPLOY_EXECUTOR = "DEPLOY_EXECUTOR"
  STOP_EXECUTOR   = "STOP_EXECUTOR"
  HELLO           = "HELLO"
  FINISHED        = "FINISHED"

class Message:
  def __init__(self, timestamp: float=None, type: Type=None, data: Dict[str, str]=None):
    """Message wrapper.
    """
    self.timestamp  = timestamp
    self.type       = type
    self.data       = data

  def marshal(self) -> str:
    obj = {
      "timestamp": self.timestamp,
      "type": self.type,
      "data": self.data, 
    }
    message = json.dumps(obj)
    return message

  def unmarshal(self, message) -> None:
    try:
      obj: dict = json.loads(message)
    except Exception as e:
      raise e
    self.timestamp = obj["timestamp"]
    self.type: str = obj["type"]
    self.data: Dict[str, str] = obj["data"]
    # logging.warning(f"✉️ [WORKER]\n\tmessage: {message}\n\tunmarshal: {obj}\n\tdata: {self.data}")
        