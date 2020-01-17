
import numpy as np
import tensorflow as tf
import nibabel as nib
from tensorflow.keras.utils import Sequence
import threading

LABEL_IS_IMAGE_TYPES = ["img", "nifti", "vol"]
LABEL_IS_FILE_LIST_TYPES = ["txt", "tsv", "file"]

##
## This is a base class take from keras-preprocessing to avoid extra
##  dependency. It essentially implements the basic needs for a python
##  iterator. 
##
class Iterator(Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batch` method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        repeat = (self.n + self.batch_size - 1) // self.n
        if self.shuffle:
            self.index_array = np.ravel([np.random.permutation(self.n) for _ in range(repeat)])
        else:
            self.index_array = np.ravel([np.arange(self.n)] * repeat)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError(f'Asked to retrieve element {idx}, but the Sequence has length {len(self)}')
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batch(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batch(index_array)

    def _get_batch(self, index_array):
        """Gets a batch of samples.

        # Arguments
            index_array: array of sample indices to include in batch.

        # Returns
            A batch of samples.
        """
        raise NotImplementedError



class MRI_DataGenerator( Iterator ):

    def __init__(self,
        list_of_samples,
        batch_size,
        batches_per_epoch=None,
        list_of_labels=None,
        label_type=None,
        load_files=False,
        shuffle=False,
        seed=None,
        preproc=None,
        preproc_params=None,
        labels_preproc=None,
        labels_preproc_params=None,
        patch_preproc=None,
        patch_preproc_params=None,
        augmenter=None,
        augment_params={},
        gridded_outputs=None,
        ):

        self.samples = list_of_samples
        self.n = len(list_of_samples)
        self.load_files = load_files

        self.input_files = list_of_samples
        self.label_files = list_of_labels
        self.label_type = label_type

        #augmentation
        self.augmenter = augmenter
        if augmenter is not None:
            self.image_transformer = augmenter(**augment_params)
        else:
            self.image_transformer = lambda x : x

        #
        self.batches_per_epoch = batches_per_epoch
        if self.batches_per_epoch is not None and batches_per_epoch <= 0:
            raise TypeError("batches per epoch should be positive int")
        elif self.batches_per_epoch is not None:
            self.real_n = self.batches_per_epoch * self.batch_size
        else:
            self.real_n = self.n

        #why did they alias this?
        self.index_generator = self._flow_index()

        ##

        if preproc is not None:
          self.preproc_func = lambda x: preproc(x, **preproc_params)
        else:
          self.preproc_func = lambda x: nib.load(x).get_fdata()

        if labels_preproc is not None:
            self.preproc_func_labels = lambda x: preproc(x, **preproc_params)
        else:
            if label_type in LABEL_IS_IMAGE_TYPES:
                self.preproc_func_labels = lambda x: nib.load(x).get_fdata()

        if self.load_files:
            #this loads everything into memory
            print("[simple_data_utils] Preloading.")

            self.inputs = []
            for filename in self.input_files:
              self.inputs.append(self.preproc_func(filename))

            if self.label_files is not None:
                self.labels = []
                if label_type in LABEL_IS_IMAGE_TYPES:
                    for filename in self.label_files:
                        self.labels.append(self.preproc_func_labels(filename))

            #self.inputs = [self.preproc_func(file) for file in self.input_files]
            #if self.label_files is not None:
            #    self.labels = [self.preproc_func_labels(file) for file in self.label_files]
            self.preproc_func = lambda x: x
            self.preproc_func_labels = lambda x: x
        else:
            self.inputs = self.input_files
            self.labels = self.label_files

        if label_type in LABEL_IS_FILE_LIST_TYPES:
            self.labels = np.loadtxt( self.label_files[0] )
        elif label_type == "array":
            self.labels = list_of_labels
        elif label_type in LABEL_IS_IMAGE_TYPES or label_type is None:
            pass
        else:
            raise Exception("[simple_data_utils] label_type not recongized.") 

        super().__init__(len(self.inputs), batch_size, shuffle, seed)

    def _set_index_array(self):
        #repeat = len(self)
        repeat = (self.real_n + self.batch_size - 1) // self.real_n
        if self.shuffle:
            self.index_array = np.ravel([ \
                np.random.permutation(self.real_n) for _ in range(repeat) \
            ])
        else:
            self.index_array = np.ravel([np.arange(self.real_n)] * repeat)

    def __len__(self):
        return (self.real_n + self.batch_size - 1) // self.batch_size  # round up

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.real_n
            if self.real_n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def _get_batch(self, index_array):

        batch = []
        batch_y = []

        for _, i in enumerate(index_array):
            #if load_files is true, this is an identity function, otherwise it's the 
            #preprocess function with preset args.
            x = self.preproc_func(self.inputs[i])
            if self.label_type is None or self.label_type is "AE":
                y = []
            elif self.label_type in LABEL_IS_FILE_LIST_TYPES or self.label_type == "array":
                y = self.labels[i]
            elif self.label_type in LABEL_IS_IMAGE_TYPES:
                y = self.preproc_func(self.labels[i])

            if self.augmenter is not None:
                if self.label_type in LABEL_IS_IMAGE_TYPES:
                    x, y = self.image_transformer.random_transform(x, y, seed=self.seed)
                else:
                    x = self.image_transformer.random_transform(x, seed=self.seed)

            batch.append(x)
            batch_y.append(y)

        if self.label_type is None:
            return np.asarray(batch)
        elif self.label_type is "AE":
            return np.asarray(batch), np.asarray(batch)
        elif self.label_type in LABEL_IS_IMAGE_TYPES:
            return np.asarray(batch), np.asarray(batch_y)

