#!/usr/bin/env python3
import time
from cereal import messaging, log
from selfdrive.manager.process_config import managed_processes

if __name__ == "__main__":
  pm = messaging.PubMaster(['pandaStates'])

  msgs = messaging.new_message('pandaStates', 1)
  msgs.pandaStates[0].ignitionLine = True
  msgs.pandaStates[0].pandaType = log.PandaState.PandaType.uno

  try:
    while True:
      time.sleep(1 / 100)  # continually send, rate doesn't matter
      pm.send('pandaStates', msgs)
  except KeyboardInterrupt:
    msgs.pandaStates[0].ignitionLine = False
    pm.send('pandaStates', msgs)

