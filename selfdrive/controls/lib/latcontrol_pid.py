import math
import json
import os

from selfdrive.controls.lib.pid import PIController
from selfdrive.controls.lib.drive_helpers import get_steer_max
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from cereal import log


TUNE_FILE = "/data/tune_pid.json"


class LatControlPID(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)

    self.kp = CP.lateralTuning.pid.kpV[0]
    self.ki = CP.lateralTuning.pid.kiV[0]
    self.kf = CP.lateralTuning.pid.kf

    self.kpBP = CP.lateralTuning.pid.kpBP
    self.kiBP = CP.lateralTuning.pid.kiBP

    self.pid = PIController((self.kpBP, [self.kp]),
                            (self.kiBP, [self.ki]),
                            k_f=self.kf, pos_limit=1.0, neg_limit=-1.0)
    self.get_steer_feedforward = CI.get_steer_feedforward_function()

  def reset(self):
    super().reset()
    self.pid.reset()

  def _reload_tune(self):
    if not os.path.exists(TUNE_FILE):
      return
    try:
      with open(TUNE_FILE, "r") as f:
        data = json.load(f)
        kp = float(data.get("kp", self.kp))
        ki = float(data.get("ki", self.ki))
        kf = float(data.get("kf", self.kf))

        if kp != self.kp or ki != self.ki or kf != self.kf:
          print(f"[Tuning] Reloading PID: kp={kp}, ki={ki}, kf={kf}")
          self.kp, self.ki, self.kf = kp, ki, kf
          self.pid = PIController((self.kpBP, [self.kp]),
                                  (self.kiBP, [self.ki]),
                                  k_f=self.kf, pos_limit=1.0, neg_limit=-1.0)
    except Exception as e:
      print(f"[Tuning] Failed to read {TUNE_FILE}: {e}")

  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate):
    pid_log = log.ControlsState.LateralPIDState.new_message()
    pid_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    pid_log.steeringRateDeg = float(CS.steeringRateDeg)

    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg

    pid_log.steeringAngleDesiredDeg = angle_steers_des
    pid_log.angleError = angle_steers_des - CS.steeringAngleDeg

    if CS.vEgo < MIN_STEER_SPEED or not active:
      output_steer = 0.0
      pid_log.active = False
      self.pid.reset()

      # Live tuning only when not active
      self._reload_tune()
    else:
      steers_max = get_steer_max(CP, CS.vEgo)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max

      steer_feedforward = self.get_steer_feedforward(angle_steers_des_no_offset, CS.vEgo)
      deadzone = 0.0

      output_steer = self.pid.update(angle_steers_des, CS.steeringAngleDeg, override=CS.steeringPressed,
                                     feedforward=steer_feedforward, speed=CS.vEgo, deadzone=deadzone)
      pid_log.active = True
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      pid_log.f = self.pid.f
      pid_log.output = output_steer
      pid_log.saturated = self._check_saturation(steers_max - abs(output_steer) < 1e-3, CS)

    return output_steer, angle_steers_des, pid_log

