# def create_steer_command(packer, steer, mode, raw_cnt):
#   """Creates a CAN message for the Seb Smith EPAS Steer Command."""

#   values = {
#     "STEER_MODE": mode,
#     "REQUESTED_STEER_TORQUE": steer,
#     "COUNTER": raw_cnt,
#   }
#   return packer.make_can_msg("OCELOT_STEERING_COMMAND", 0, values)

MAX_TORQUE = 350. # cannot be over 1000

def create_steer_interceptor_command(packer, torque, enable, idx):

  values = {
    "ENABLE": enable,
    "COUNTER": idx & 0xF,
  }

  if enable:
    values["TORQUE_COMMAND1"] = 1510 + (torque)
    values["TORQUE_COMMAND2"] = 1510 - (torque) 

  return packer.make_can_msg("INTERCEPTOR_STEERING_COMMAND", 0, values)

def create_gas_interceptor_command(packer, gas_amount, idx):
  # Common gas pedal msg generator
  enable = gas_amount > 0.001

  values = {
    "ENABLE": enable,
    "COUNTER": idx & 0xF,
  }

  if enable:
    # TODO: parameterize these values - they vary based on the car
    values["GAS_COMMAND"] = (gas_amount * 2400) + 850.
    values["GAS_COMMAND2"] = (gas_amount * 2000) + 480.

  return packer.make_can_msg("PEDAL_GAS_COMMAND", 0, values)

def create_gas_actuator_command(packer, enabled, gas_amount, idx):
  values = {
    "ENABLE": enabled,
    "COUNTER": idx & 0xF,
  }
  if enabled:
    values["THROTTLE_REQ"] = (gas_amount * 1500)

  return packer.make_can_msg("ACTUATOR_GAS_COMMAND", 0, values)

def create_iBooster_cmd(packer, enabled, brake, raw_cnt):
  values = {
    "BRAKE_POSITION_COMMAND" : brake * 7,
    "BRAKE_RELATIVE_COMMAND": 0, #brake * 252,
    "BRAKE_MODE": enabled * 2.,
    "COUNTER" : raw_cnt,
  }
  return packer.make_can_msg("IBOOSTER_BRAKE_COMMAND", 0, values)

def create_relay_command(packer, enabled, relay, idx):
  values = {
    "RELAY_COMMAND": relay,
    "ENABLE": enabled,
    "COUNTER": idx & 0xF,
  }
  return packer.make_can_msg("RELAY_CORE_COMMAND", 0, values)
