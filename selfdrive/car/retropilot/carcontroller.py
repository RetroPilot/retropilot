from cereal import car
from common.numpy_fast import clip
from selfdrive.car import apply_toyota_steer_torque_limits #, make_can_msg
from selfdrive.car.retropilot.ocelotcan import create_gas_interceptor_command, create_gas_actuator_command, \
                                           create_steer_interceptor_command, create_iBooster_cmd, create_relay_command
from selfdrive.car.retropilot.values import SteerLimitParams
from opendbc.can.packer import CANPacker

VisualAlert = car.CarControl.HUDControl.VisualAlert

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.last_steer = 0
    self.steer_rate_limited = False

    self.alert_active = False

    self.packer = CANPacker(dbc_name)

  def update(self, enabled, active, CS, frame, actuators):
    can_sends = []
    enabled = 1
    # *** compute control surfaces ***
    # if not enabled, everything should be 0
    if not enabled:
      apply_steer = 0
      apply_steer_req = 0
      apply_gas = 0
      apply_brake = 0
    else:
      apply_steer_req = 1

    # gas and brake
    apply_gas = clip(actuators.gas, 0., 1.)
    apply_brake = clip(actuators.brake, 0., 1.)
    # if (frame % 2 == 0):
      # detect whether to use gas interceptor or actuator
      # if CS.CP.enableGasInterceptor:
      #   can_sends.append(create_gas_interceptor_command(self.packer, apply_gas, frame//2))
      # if CS.CP.enableGasActuator:
      #   can_sends.append(create_gas_actuator_command(self.packer, apply_gas, frame//2))
      # if CS.CP.enableiBooster:
      #   can_sends.append(create_iBooster_cmd(self.packer, enabled, apply_brake, frame//2))
      # can_sends.append(create_iBooster_cmd(self.packer, enabled, apply_brake, frame//2))

    # steer torque
    new_steer = int(round(actuators.steer * SteerLimitParams.STEER_MAX))
    apply_steer = apply_toyota_steer_torque_limits(new_steer, self.last_steer, CS.out.steeringTorqueEps, SteerLimitParams)
    self.steer_rate_limited = new_steer != apply_steer

    self.last_steer = apply_steer

    # send steering command. currently only support interceptor
    # if CS.CP.enableSteerInterceptor:
    can_sends.append(create_steer_interceptor_command(self.packer, apply_steer, apply_steer_req, frame))
    can_sends.append(create_gas_actuator_command(self.packer, enabled, apply_gas, frame))
    can_sends.append(create_gas_interceptor_command(self.packer, apply_gas, frame))
    can_sends.append(create_iBooster_cmd(self.packer, enabled, apply_brake, frame))
    can_sends.append(create_relay_command(self.packer, enabled, 3, frame))

    # #*** static msgs ***
    # TODO: add static messages here. stuff like radar if detected, etc
    # for (addr, ecu, cars, bus, fr_step, vl) in STATIC_MSGS:
    #   if frame % fr_step == 0 and ecu in self.fake_ecus and CS.CP.carFingerprint in cars:
    #     can_sends.append(make_can_msg(addr, vl, bus))

    new_actuators = actuators.copy()
    new_actuators.steer = apply_steer
    new_actuators.brake = apply_brake
    new_actuators.gas = apply_gas

    return new_actuators, can_sends
