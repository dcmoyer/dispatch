
import numpy as np
import nibabel as nib

class LoadedImg():
  def __init__(self, filename, mask=None, type="nii"):
    if type.endswith("nii") or type.endswith("nii.gz"):
      self.img_block = nib.load(filename).get_fdata()
    else:
      raise Exception("File type not recognized")

    if mask:
      self.mask_block = nib.load(mask).get_fdata()
      if len(self.mask_block.shape) > 3:
        print(self.mask_block.shape)
        print("Warning, only 3d masks supported")
      elif len(self.mask_block.shape) < 3:
        raise Exception("2d or 1d mask???")

      if self.mask_block.shape[0:3] != self.img_block.shape[0:3]:
        raise Exception("image and mask differ in shape")

    else:
      self.mask_block = mask

  def iterator(self):
    np_ma_obj = np.ma.array( self.img_block, mask=self.mask_block )
    for vox in np_ma_obj:
      yield vox

  def idx_iterator(self):
    if self.mask_block is not None:

      where_mask_output = np.where(self.mask_block)
      for idx_0, idx_1, idx_2 in zip(*where_mask_output):
        yield (idx_0, idx_1, idx_2)

    else:

      #This is highly inefficient
      print("Warning, called idx_iterator without mask, this is probably unnecessary...")
      img_shape = self.img_block.shape[0:3]
      for i in range(img_shape[0]):
        for j in range(img_shape[1]):
          for k in range(img_shape[2]):
            yield (i,j,k)










