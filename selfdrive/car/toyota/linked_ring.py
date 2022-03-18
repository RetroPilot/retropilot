
class Link(object):
  def __init__(self, value=0.0):
    self.next = None
    self.value = value


class LinkedRing(object):
  def __init__(self, length):
    self.sum = 0.0
    self.length = length
    self.current = Link()

    # Initialize all the nodes:
    last = self.current
    for _ in range(length - 1):  # one link is already created
      last.next = Link()
      last = last.next
    last.next = self.current  # close the ring

  def add_val(self, val):
    self.sum -= self.current.value
    self.sum += val
    self.current.value = val
    self.current = self.current.next

  def average(self):
    return self.sum / self.length

