import numpy as np

# training
class DataInput:
  def __init__(self, data, batch_size, k):
    self.k = k
    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    self.i += 1

    u, i, y, sl, new_sl, c = [], [], [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[4])
      y.append(t[5])
      c.append(t[6])
      sl.append(min(len(t[1]), self.k))
      new_sl.append(len(t[2]))
    max_new_sl = max(new_sl)

    hist_i = np.zeros([len(ts), self.k], np.int64)
    hist_t = np.zeros([len(ts), self.k], np.float32)
    hist_i_new = np.zeros([len(ts), max_new_sl], np.int64)

    kk = 0
    for t in ts:
      length = len(t[1])
      if length > self.k:
        for l in range(self.k):
          hist_i[kk][l] = t[1][length-self.k+l]
          hist_t[kk][l] = t[3][length-self.k+l]
      else:
        for l in range(length):
          hist_i[kk][l] = t[1][l]
          hist_t[kk][l] = t[3][l]
      for l in range(len(t[2])):
          hist_i_new[kk][l] = t[2][l]
      kk += 1

    return self.i, (u, i, y, hist_i, hist_i_new, hist_t, sl, new_sl, c)

# testing
class DataInputTest:
  def __init__(self, data, batch_size, k):
    self.k = k
    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    self.i += 1

    u, i, j, sl, new_sl, c = [], [], [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[4][0])
      j.append(t[4][1])
      c.append(t[5])
      sl.append(min(len(t[1]), self.k))
      new_sl.append(len(t[2]))
    max_new_sl = max(new_sl)

    hist_i = np.zeros([len(ts), self.k], np.int64)
    hist_t = np.zeros([len(ts), self.k], np.float32)
    hist_i_new = np.zeros([len(ts), max_new_sl], np.int64)

    kk = 0
    for t in ts:
      length = len(t[1])
      if length > self.k:
        for l in range(self.k):
          hist_i[kk][l] = t[1][length-self.k+l]
          hist_t[kk][l] = t[3][length-self.k+l]
      else:
        for l in range(length):
          hist_i[kk][l] = t[1][l]
          hist_t[kk][l] = t[3][l]
      for l in range(len(t[2])):
          hist_i_new[kk][l] = t[2][l]
      kk += 1

    return self.i, (u, i, j, hist_i, hist_i_new, hist_t, sl, new_sl, c)
