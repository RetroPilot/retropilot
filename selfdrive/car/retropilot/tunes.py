#!/usr/bin/env python3
from enum import Enum


class LongTunes(Enum):
  PEDAL = 0
  ACTUATOR = 1

class LatTunes(Enum):
  PID_A = 1


###### LONG ######
def set_long_tune(tune, name):
  # Improved longitudinal tune
  if name == LongTunes.PEDAL:
    tune.deadzoneBP = [0., 8.05]
    tune.deadzoneV = [.0, .14]
    tune.kpBP = [0., 5., 20.]
    tune.kpV = [1.3, 1.0, 0.7]
    tune.kiBP = [0., 5., 12., 20., 27.]
    tune.kiV = [.35, .23, .20, .17, .1]
  # Default longitudinal tune
  elif name == LongTunes.ACTUATOR:
    tune.deadzoneBP = [0., 9.]
    tune.deadzoneV = [0., .75]
    tune.kpBP = [0., 5., 35.]
    tune.kiBP = [0., 35.]
    tune.kpV = [3.6, 2.4, 1.5]
    tune.kiV = [0.54, 0.36]
  else:
    raise NotImplementedError('This longitudinal tune does not exist')


###### LAT ######
def set_lat_tune(tune, name):
  tune.init('pid')
  tune.pid.kiBP = [0.0]
  tune.pid.kpBP = [0.0]
  if name == LatTunes.PID_A:
    tune.pid.kpV = [0.05]
    tune.pid.kiV = [0.05]
    tune.pid.kf = 0.00003
  else:
    raise NotImplementedError('This PID tune does not exist')
