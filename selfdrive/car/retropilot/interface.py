#!/usr/bin/env python3
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.car.retropilot.tunes import LatTunes, LongTunes, set_long_tune, set_lat_tune
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase
from selfdrive.car.retropilot.values import DetectedEcus

EventName = car.CarEvent.EventName

class CarInterface(CarInterfaceBase):
  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 3.0

  @staticmethod
  def myround(x, base=5):
    return base * round(x/base)

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=[]):  # pylint: disable=dangerous-default-value

    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)

    ret.carName = "retropilot"
    #TODO: ocelot panda safety. allOutput is kinda cursed
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.retropilot)]
    ret.safetyConfigs[0].safetyParam = 100

    ret.steerActuatorDelay = 0.12  # Default delay, Prius has larger delay
    ret.steerLimitTimer = 0.4
    ret.stoppingControl = True

    ret.openpilotLongitudinalControl = True

    ECU_FP = {
      "GasInterceptor": 0x201,
      "GasActuator": 0x401,
      "SteerInterceptor": 0x301,
      "SteerActuator": 0x12F,
      "SteerActuatorSSC": 0x22F,
      "iBooster": 0x20F,
      "RelayCore": 0x601,
    }

    for ecu, addr in ECU_FP.items():
      if addr in fingerprint[0]:
        DetectedEcus[ecu] = True

    ret.enableGasInterceptor = DetectedEcus["GasInterceptor"]

    # tuning

    ret.lateralTuning.init('pid')
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
    ret.minEnableSpeed = -1.

    ret.wheelbase = 2.70
    ret.steerRatio = 18.27
    tire_stiffness_factor = 0.444
    ret.mass = 2860. * CV.LB_TO_KG + STD_CARGO_KG
    set_lat_tune(ret.lateralTuning, LatTunes.PID_A)

    # end tuning

    ret.steerRateCost = 1.
    ret.centerToFront = ret.wheelbase * 0.44
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)
    # TODO: figure out how to enable aftermarket BSM radars
    ret.enableBsm = False
    if ret.enableGasInterceptor:
      set_long_tune(ret.longitudinalTuning, LongTunes.PEDAL)
    else:
      set_long_tune(ret.longitudinalTuning, LongTunes.ACTUATOR)

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    # ******************* do can recv *******************
    self.cp.update_strings(can_strings)
    self.cp_body.update_strings(can_strings)

    ret = self.CS.update(self.cp)

    ret.canValid = True #self.cp.can_valid
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    # events
    events = self.create_common_events(ret)

    ret.events = events.to_msg()

    self.CS.out = ret.as_reader()
    return self.CS.out

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c):

    # simple!
    ret = self.CC.update(c.enabled, c.active, self.CS, self.frame,
                               c.actuators, c.bodycontrol)

    self.frame += 1
    return ret
