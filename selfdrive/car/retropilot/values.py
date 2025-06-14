# flake8: noqa

from selfdrive.car import dbc_dict
from cereal import car
Ecu = car.CarParams.Ecu

DetectedEcus = {
  "GasInterceptor": False,
  "GasActuator": False,
  "SteerInterceptor": False,
  "SteerActuator": False,
  "SteerActuatorSSC": False,
  "iBooster": False,
  "BrakeActuator": False,
  "RelayCore": False,
  "Radar": False,
}

# edit this to 
RelayMsg = {
  "L_TURN": 0b00000001,
  "R_TURN": 0b00000010,
  "HAZARD": 0b00000011,
  "TAIL": 0b00000100,
  "HEAD": 0b00001100,
  "BRIGHTS": 0b00011000,
}
  
# Steer torque limits
class SteerLimitParams:
  STEER_MAX = 350
  STEER_DELTA_UP = 7       # 1.5s time to peak torque
  STEER_DELTA_DOWN = 7     # always lower than 45 otherwise the Rav4 faults (Prius seems ok with 50)
  STEER_ERROR_MAX = STEER_MAX     # max delta between torque cmd and torque motor

class CAR:
  RETROFIT = "RETROPILOT RETROFIT"


FINGERPRINTS = {
  CAR.RETROFIT: [{
     0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8,
   }],
}

STEER_THRESHOLD = 100

DBC = {
    CAR.RETROFIT: dbc_dict('ocelot_controls', 'toyota_tss2_adas'),
}
