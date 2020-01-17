
import numpy as np

from load_images import LoadedImg

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
  def __init__(self):
    pass

  def generate(self, loaded_img_obj, n=None, template=patch_template(), flatten=True):
    for patch_idx, indices in enumerate(loaded_img_obj.idx_iterator()):

      if n is not None and patch_idx > n:
        raise StopIteration

      patch_indices = indices + template

      #TODO safety first here
      values = loaded_img_obj.img_block[ \
        patch_indices[:,0], \
        patch_indices[:,1], \
        patch_indices[:,2], \
        :\
      ]

      if flatten:
        values = values.flatten()
      yield values, indices







