
import numpy as np
import nibabel as nib
import pathlib

def _maybe_load( volume_or_name ):
  if isinstance(volume_or_name,str) or isinstance(volume_or_name,pathlib.Path):

    if volume_or_name.endswith("nii") or volume_or_name.endswith("nii.gz"):
      return nib.load(volume_or_name).get_fdata()

    elif volume_or_name.endswith("npy"):
      return np.load(volume_or_name) #assumes single matrix saved, so don't mess it up :)

    elif volume_or_name.endswith("npz"):
      loaded_zip = np.load(volume_or_name)
      return loaded_zip[ loaded_zip.files[0] ]

    else:
      raise Exception("File type not recognized")

  elif isinstance(volume_or_name,np.ndarray):
    return volume_or_name

  else:
    raise TypeError("Image Loader only accepts .nii, .nii.gz, or numpy ndarrays.")

class LoadedImg():
  def __init__(self, img=None, mask=None ):

    self.img_block = _maybe_load( img )

    if mask is not None:
        self.mask_block = _maybe_load( mask )
    else:
        self.mask_block = None

    if self.mask_block is not None and len(self.mask_block.shape) < 3:
      raise Exception("2d or 1d mask???")

    if self.mask_block is not None and self.mask_block.shape[0:3] != self.img_block.shape[0:3]:
      raise Exception("image and mask differ in shape")

  def iterator(self):
    np_ma_obj = np.ma.array( self.img_block, mask=self.mask_block )
    for vox in np_ma_obj:
      yield vox

  def idx_iterator(self, shuffle=False):
    if self.mask_block is not None:

      where_mask_output = np.where(self.mask_block)
      where_mask_output = list(zip(*where_mask_output))
      if shuffle:
        np.random.shuffle(where_mask_output)
      for index in where_mask_output:
        yield index 

    else:

      #This is highly inefficient
      print("Warning, called idx_iterator without mask, this is probably unnecessary...")
      img_shape = self.img_block.shape[0:3]
      for idx in np.ndindex(*img_shape):
        yield idx
      #for i in range(img_shape[0]):
      #  for j in range(img_shape[1]):
      #    for k in range(img_shape[2]):
      #      yield (i,j,k)










