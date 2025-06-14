#!/usr/bin/env python3
import argparse
import time
import threading

from inputs import get_gamepad
import cereal.messaging as messaging
from common.numpy_fast import interp, clip
from common.params import Params
from tools.lib.kbhit import KBHit


class Keyboard:
  def __init__(self):
    self.kb = KBHit()
    self.axis_increment = 0.05
    self.axes_map = {'w': 'gb', 's': 'gb', 'a': 'steer', 'd': 'steer'}
    self.axes_values = {'gb': 0., 'steer': 0.}
    self.cancel = False

  def update(self):
    if not self.kb.kbhit():
      return False
    key = self.kb.getch().lower()
    self.cancel = False
    if key == 'r':
      self.axes_values = {ax: 0. for ax in self.axes_values}
    elif key == 'c':
      self.cancel = True
    elif key in self.axes_map:
      axis = self.axes_map[key]
      incr = self.axis_increment if key in ['w', 'a'] else -self.axis_increment
      self.axes_values[axis] = clip(self.axes_values[axis] + incr, -1, 1)
    else:
      return False
    return True


class Joystick:
  def __init__(self):
    self.min_axis_value = 0
    self.max_axis_value = 255
    self.axes_values = {'ABS_Y': 0., 'ABS_RX': 0.}

    # Button mappings (all treated as real-time)
    self.button_codes = ['BTN_SOUTH', 'BTN_EAST', 'BTN_WEST', 'BTN_TL', 'BTN_TR']
    self.button_states = {code: False for code in self.button_codes}
    self.button_lock = threading.Lock()

  def _handle_event(self, event):
    with self.button_lock:
      if event.code in self.button_states:
        self.button_states[event.code] = (event.state != 0)
      elif event.code in self.axes_values:
        self.max_axis_value = max(event.state, self.max_axis_value)
        self.min_axis_value = min(event.state, self.min_axis_value)
        norm = -interp(event.state, [self.min_axis_value, self.max_axis_value], [-1., 1.])
        self.axes_values[event.code] = norm if abs(norm) > 0.05 else 0.

  def start_thread(self):
    def run():
      while True:
        try:
          events = get_gamepad()
          for e in events:
            self._handle_event(e)
        except Exception:
          pass
    threading.Thread(target=run, daemon=True).start()

  def update(self):
    return True  # Always ready

  def get_button_states(self):
    with self.button_lock:
      return [self.button_states[b] for b in self.button_codes]


def joystick_thread(use_keyboard):
  Params().put_bool('JoystickDebugMode', True)
  joystick_sock = messaging.pub_sock('testJoystick')
  joystick = Keyboard() if use_keyboard else Joystick()
  if not use_keyboard:
    joystick.start_thread()

  rate_hz = 20
  dt = 1.0 / rate_hz

  while True:
    start = time.monotonic()
    joystick.update()

    dat = messaging.new_message('testJoystick')
    dat.testJoystick.axes = [joystick.axes_values[a] for a in joystick.axes_values]
    dat.testJoystick.buttons = joystick.get_button_states()
    joystick_sock.send(dat.to_bytes())

    print(', '.join(f'{k}: {round(v, 3)}' for k, v in joystick.axes_values.items()),
          "buttons:", joystick.get_button_states())

    elapsed = time.monotonic() - start
    time.sleep(max(0, dt - elapsed))


def main():
  parser = argparse.ArgumentParser(description='Publishes events from your joystick to control your car.\n' +
                                               'openpilot must be offroad before starting joysticked.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--keyboard', action='store_true', help='Use your keyboard instead of a joystick')
  args = parser.parse_args()

  if args.keyboard:
    print('Gas/brake control: `W` and `S` keys')
    print('Steering control: `A` and `D` keys')
    print('Buttons:')
    print('- `R`: Resets axes')
    print('- `C`: Cancel cruise control')
  else:
    print('Using joystick...')

  joystick_thread(args.keyboard)


if __name__ == '__main__':
  main()
