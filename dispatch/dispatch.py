
import numpy as np

from .load_images import LoadedImg

#
# L1 is the "manhattan" distance patch:
#   0 1 0
#   1 1 1
#   0 1 0
# Square is the "manhattan" distance patch:
#   1 1 1
#   1 1 1
#   1 1 1
#TODO: generalize to different dimensions
def patch_template(size=1, dim=3, type="L1"):
  if type not in ["L1","Linf","square"]:
    raise Exception("Patch type %s not supported" % type)
  template = []

  for i in range(-size,size+1):
    for j in range(-size,size+1):
      for k in range(-size,size+1):
        if type == "L1" and abs(i) + abs(j) + abs(k) <= size:
          template.append([i,j,k])
        elif type in ["Linf", "square"]:
          template.append([i,j,k])

  template = np.array(template)
  return template


class PatchMaker():
  '''
  does not respect padding
  '''
  def __init__(self, loaded_img_obj, n=None, template=patch_template(), flatten=True, shuffle=False, bounds_check=True, remove_boundary=False):
    self.loaded_img_obj = loaded_img_obj
    self.n = n
    self.template = template
    self.flatten = flatten
    self.shuffle = shuffle
    self.bounds_check = bounds_check
    self.remove_boundary = remove_boundary

  def __iter__(self):
    for patch_idx, indices in enumerate(self.loaded_img_obj.idx_iterator(shuffle=self.shuffle)):

      if self.n is not None and patch_idx > self.n:
        break

      patch_indices = indices + self.template

      if self.remove_boundary:
        if np.any(patch_indices >= self.loaded_img_obj.img_block.shape[0:3]) or np.any(patch_indices < 0):
          continue

      if self.bounds_check:
        if np.any(patch_indices >= self.loaded_img_obj.img_block.shape[0:3]) or np.any(patch_indices < 0):
          raise IndexError("mask induces patch outside bounds, use remove_boundary=True for naive fix")

      #TODO safety first here
      values = self.loaded_img_obj.img_block[ \
        patch_indices[:,0], \
        patch_indices[:,1], \
        patch_indices[:,2], \
        :\
      ]

      if self.flatten:
        values = values.flatten()
      yield values, indices







