from cereal import car
# from common.numpy_fast import mean
# from opendbc.can.can_define import CANDefine
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.config import Conversions as CV
from selfdrive.car.retropilot.values import DBC, DetectedEcus

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    print(DBC[CP.carFingerprint]['pt'])
    # can_define = CANDefine(DBC[CP.carFingerprint]['pt'])
    # self.shifter_values = can_define.dv["GEAR_PACKET"]['GEAR']
    self.shifter_values = "D"
    self.setSpeed = 0
    self.armed = False
    self.enabled = False
    self.enabled_last = True

  def update(self, cp):
    ret = car.CarState.new_message()
    #Car specific information
    print(DetectedEcus)

    if DetectedEcus["RelayCore"]:
      ret.leftBlinker = (cp.vl["RELAY_CORE_STATUS"]['RELAY_STATUS'] >> 7) & 1
      ret.rightBlinker = (cp.vl["RELAY_CORE_STATUS"]['RELAY_STATUS'] >> 6) & 1

    # if self.CP.carFingerprint == CAR.SMART_ROADSTER_COUPE:
    #     ret.doorOpen = False #any([cp_body.vl["BODYCONTROL"]['RIGHT_DOOR'], cp_body.vl["BODYCONTROL"]['LEFT_DOOR']]) != 0
    #     ret.seatbeltUnlatched = False
    #     ret.espDisabled = False #cp_body.vl["ABS"]['ESP_STATUS']
    #     ret.brakeLights = False #cp_body.vl["ABS"]['BRAKEPEDAL']
    #     can_gear = 0 #int(cp_body.vl["GEARBOX"]['GEARPOSITION'])
    #     ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    # ret.wheelSpeeds.fl = (cp.vl["WHEEL_SPEEDS"]['WHEEL_FL'] * 0.01) * 1.23 * CV.KPH_TO_MS
    # ret.wheelSpeeds.fr = (cp.vl["WHEEL_SPEEDS"]['WHEEL_FR'] * 0.01) * 1.23 * CV.KPH_TO_MS
    # ret.wheelSpeeds.rl = (cp.vl["WHEEL_SPEEDS"]['WHEEL_FL'] * 0.01) * 1.23 * CV.KPH_TO_MS
    # ret.wheelSpeeds.rr = (cp.vl["WHEEL_SPEEDS"]['WHEEL_FR'] * 0.01) * 1.23 * CV.KPH_TO_MS
    # ret.vEgoRaw = mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr])
    
    # Brakes
    if DetectedEcus["iBooster"]:
      ret.brakePressed = bool(cp.vl["IBOOSTER_BRAKE_STATUS"]['BRAKE_APPLIED'])
    if DetectedEcus["BrakeActuator"]:
      ret.brakePressed = bool(cp.vl["ACTUATOR_BRAKE_STATUS"]['BRAKE_APPLIED'])
    if self.enabled and ret.brakePressed:
      self.enabled = False

    # Gas
    if DetectedEcus["GasInterceptor"]:
      ret.gas = (cp.vl["PEDAL_GAS_SENSOR"]['PED_GAS'] + cp.vl["PEDAL_GAS_SENSOR"]['PED_GAS2']) / 2. #TODO: get divisor, offset, scalar from a param
      ret.gasPressed = ret.gas > 15
    if DetectedEcus["GasActuator"]:
      ret.gas = cp.vl["ACTUATOR_GAS_SENSOR"]['THROTTLE_POS'] #TODO: get scalar and offset from a param
      ret.gasPressed = False

    vss_us = cp.vl["SPEED"]['VSS_PULSE_US']

    if vss_us > 0:
      hz = 1e6 / vss_us
      ret.vEgoRaw = hz * (3600/4000) * CV.MPH_TO_MS
    else:
      ret.vEgoRaw = 0.0

    #calculate speed from wheel speeds
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.001
    
    # Steering
    if DetectedEcus["SteerInterceptor"]:
      #TODO: get divisor, offset, scalar from a param
      ret.steeringTorque = (cp.vl["INTERCEPTOR_STEERING_SENSOR"]['TRQ_2'] - cp.vl["INTERCEPTOR_STEERING_SENSOR"]['TRQ_1']) / 2 
      ret.steeringPressed = abs(ret.steeringTorque) > 1000 
      ret.steerWarning = cp.vl["INTERCEPTOR_STEERING_SENSOR"]['STATE'] != 0
      ret.steeringTorqueEps = ret.steeringTorque * 100
    if DetectedEcus["SteerActuator"]:
      #TODO: get divisor, offset, scalar from a param
      ret.steeringTorque = cp.vl["ACTUATOR_STEERING_STATUS"]['STEERING_TORQUE_DRIVER']
      ret.steeringPressed = abs(ret.steeringTorque) > 1000 
      ret.steerWarning = cp.vl["ACTUATOR_STEERING_STATUS"]['STEERING_OK'] != 0
      ret.steerTorqueEps = cp.vl["ACTUATOR_STEERING_STATUS"]['STEERING_TORQUE_EPS'] 
    if DetectedEcus["SteerActuatorSSC"]:
      # TODO: implement SSC debug for warning
      ret.steeringTorque = 0
      ret.steeringTorqueEps = cp.vl["STEERING_STATUS_SSC"]['STEERING_TORQUE']
      ret.steerWarning = False
      ret.steeringPressed = False

    # Ocelot SAS
    ret.steeringAngleDeg = cp.vl["STEER_ANGLE_SENSOR"]['STEER_ANGLE']
    ret.steeringRateDeg = cp.vl["STEER_ANGLE_SENSOR"]['STEER_RATE']

    ret.cruiseState.standstill = False
    ret.cruiseState.nonAdaptive = False

    if cp.vl["CRUISE"]["ON_OFF"]:
      self.armed = not(self.armed)
      if self.armed:
        if self.enabled:
          self.enabled_last = True
          if cp.vl["CRUISE"]["RES_UP"]:
            self.setSpeed += 5*CV.MPH_TO_MS
          if cp.vl["CRUISE"]["SET_DOWN"] and self.setSpeed >= 10*CV.MPH_TO_MS:
            self.setSpeed -= 5*CV.MPH_TO_MS
          if cp.vl["CRUISE"]["CANCEL"]:
            self.enabled = False
          self.enabled_last = True
        elif cp.vl["CRUISE"]["SET_DOWN"]:
          self.setSpeed = ret.vEgo
          self.enabled = True
        elif cp.vl["CRUISE"]["RES_UP"] and self.enabled_last:
          self.enabled = True

    ret.cruiseState.available = self.armed
    ret.cruiseState.enabled = self.enabled
    ret.cruiseState.speed = self.setSpeed

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      ("STEER_ANGLE", "STEER_ANGLE_SENSOR"),
      ("STEER_RATE", "STEER_ANGLE_SENSOR"),
      ("VSS_PULSE_US", "SPEED"),
      ("CAN_SPEED", "SPEED"),
      ("MODE", "SPEED"),
      ("ADC_1", "CRUISE"),
      ("ADC_2", "CRUISE"),
      ("ON_OFF", "CRUISE"),
      ("RES_UP", "CRUISE"),
      ("SET_DOWN", "CRUISE"),
      ("CANCEL", "CRUISE"),
      ("MODE", "CRUISE"),
    ]
    checks = [
      ("STEER_ANGLE_SENSOR", 20),
      ("SPEED", 20),
      ("CRUISE", 20),
      ("ENGINE", 20),
    ]

    if DetectedEcus["GasInterceptor"]:
      signals += [
        ("PED_GAS", "PEDAL_GAS_SENSOR"),
        ("PED_GAS2", "PEDAL_GAS_SENSOR"),
        ("STATE", "PEDAL_GAS_SENSOR"),
      ]
      checks += [
        ("PEDAL_GAS_SENSOR", 20)
      ]
    if DetectedEcus["GasActuator"]:
      signals += [
        ("THROTTLE_POS", "ACTUATOR_GAS_SENSOR"),
      ]
      checks += [
        ("ACTUATOR_GAS_SENSOR", 20)
      ]
    if DetectedEcus["SteerInterceptor"]:
      signals += [
        ("TRQ_1", "INTERCEPTOR_STEERING_SENSOR"),
        ("TRQ_2", "INTERCEPTOR_STEERING_SENSOR"),
        ("STATE", "INTERCEPTOR_STEERING_SENSOR"),
      ]
      checks += [
        ("INTERCEPTOR_STEERING_SENSOR", 20)
      ]
    if DetectedEcus["SteerActuator"]:
      signals += [
        ("STEERING_TORQUE_EPS", "ACTUATOR_STEERING_STATUS"),
        ("STEERING_TORQUE_DRIVER", "ACTUATOR_STEERING_STATUS"),
        ("STEERING_OK", "ACTUATOR_STEERING_STATUS"),
        ("STATUS", "ACTUATOR_STEERING_STATUS"),
      ]
      checks += [
        ("ACTUATOR_STEERING_STATUS", 20)
      ]
    if DetectedEcus["SteerActuatorSSC"]:
      signals += [
        ("STEERING_ANGLE", "STEERING_STATUS_SSC"),
        ("STEERING_SPEED", "STEERING_STATUS_SSC"),
        ("STEERING_TORQUE", "STEERING_STATUS_SSC"),
        ("CONTROL_STATUS", "STEERING_STATUS_SSC"),
      ]
      checks += [
        ("STEERING_STATUS_SSC", 20)
      ]
    if DetectedEcus["iBooster"]:
      signals += [
        ("BRAKE_APPLIED", "IBOOSTER_STATUS"),
        ("DRIVER_BRAKE_APPLIED", "IBOOSTER_STATUS"),
        ("BRAKE_OK", "IBOOSTER_STATUS"),
        ("STATUS", "IBOOSTER_STATUS"),
      ]
      checks += [
        ("IBOOSTER_STATUS", 20)
      ]
    if DetectedEcus["BrakeActuator"]:
      signals += [
        ("DRIVER_BRAKE_APPLIED", "ACTUATOR_BRAKE_STATUS"),
        ("BRAKE_OK", "ACTUATOR_BRAKE_STATUS"),
        ("STATUS", "ACTUATOR_BRAKE_STATUS"),
      ]
      checks += [
        ("ACTUATOR_BRAKE_STATUS", 20)
      ]
    if DetectedEcus["RelayCore"]:
      signals += [
        ("RELAY_CORE_RELAY_STATUSSTATUS", "RELAY_CORE_COMMAND"),
      ]
      checks += [
        ("RELAY_CORE_COMMAND", 20)
      ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

  @staticmethod
  def get_body_can_parser(CP):

    signals = [
    ]

    checks = [
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 1)
