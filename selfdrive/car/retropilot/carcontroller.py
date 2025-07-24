from cereal import car
from common.numpy_fast import clip
from selfdrive.car import apply_toyota_steer_torque_limits #, make_can_msg
from selfdrive.car.retropilot.ocelotcan import create_gas_interceptor_command, create_gas_actuator_command, \
                                           create_steer_interceptor_command, create_iBooster_cmd, create_relay_command
from selfdrive.car.retropilot.values import SteerLimitParams
from opendbc.can.packer import CANPacker
from selfdrive.car.retropilot.values import DetectedEcus

VisualAlert = car.CarControl.HUDControl.VisualAlert

def compute_gas_brake(accel):
  gb = float(accel) / 4.8
  return clip(gb, 0.0, 1.0), clip(-gb, 0.0, 1.0)

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.last_steer = 0
    self.steer_rate_limited = False

    self.alert_active = False

    self.accel = 0
    self.speed = 0
    self.gas = 0
    self.brake = 0

    self.packer = CANPacker(dbc_name)

  def update(self, enabled, active, CS, frame, actuators, relays):
    can_sends = []
    # *** compute control surfaces ***
    # if not enabled, everything should be 0
    if not enabled:
      apply_steer = 0
      apply_steer_req = 0
      apply_gas = 0
      apply_brake = 0
    else:
      apply_steer_req = 1

    if active:
      apply_brake = 0.0
      apply_gas = clip(actuators.accel, 0.0, 1.0)
      if actuators.accel < 0:
        apply_brake = clip(-actuators.accel, 0.0, 1.0)
    else:
      apply_gas = 0.0
      apply_brake = 0.0

    # don't gas and brake
    if CS.out.gas > 450:
      apply_brake = 0

    # print("enabled: ", enabled, "active: ", active, "actuators: ", apply_gas, apply_brake, actuators.steer)
    # for ecu, present in DetectedEcus.items():
    #   if present:
    #     print(f"Detected ECU: {ecu}")

    # steer torque (on interceptor, max torque should scale inversely with speed)
    steer_lim = SteerLimitParams.STEER_MAX * (1 - (CS.out.vEgo / 90)) 
    new_steer = int(round(actuators.steer * steer_lim))
    apply_steer = apply_toyota_steer_torque_limits(new_steer, self.last_steer, CS.out.steeringTorqueEps, SteerLimitParams)
    self.steer_rate_limited = new_steer != apply_steer

    self.last_steer = apply_steer

    # 50Hz Messages
    if (frame % 2 == 0):
      if DetectedEcus["GasInterceptor"]:
        can_sends.append(create_gas_interceptor_command(self.packer, apply_gas, frame//2))
      if DetectedEcus["GasActuator"]:
        can_sends.append(create_gas_actuator_command(self.packer, enabled, apply_gas, frame//2))
      # if DetectedEcus["SteerActuator"]:
      #   can_sends.append(create_steer_actuator_command(self.packer, apply_steer, apply_steer_req, frame//2))
      if DetectedEcus["RelayCore"]:
        can_sends.append(create_relay_command(self.packer, enabled, relays.relayCoreCMD, frame//2))
    
    # 100Hz Messages
    if DetectedEcus["iBooster"]:
      can_sends.append(create_iBooster_cmd(self.packer, enabled, apply_brake, frame))
    if DetectedEcus["SteerInterceptor"]:
      can_sends.append(create_steer_interceptor_command(self.packer, apply_steer, apply_steer_req, frame))
      
    # #*** static msgs ***
    # TODO: add static messages here. stuff like radar if detected, etc
    # for (addr, ecu, cars, bus, fr_step, vl) in STATIC_MSGS:
    #   if frame % fr_step == 0 and ecu in self.fake_ecus and CS.CP.carFingerprint in cars:
    #     can_sends.append(make_can_msg(addr, vl, bus))

    self.accel = actuators.accel
    self.speed = CS.out.vEgo 
    self.gas = apply_gas
    self.brake = apply_brake

    new_actuators = actuators.copy()
    new_actuators.speed = self.speed
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas
    new_actuators.brake = self.brake

    return new_actuators, can_sends
