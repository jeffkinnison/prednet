"""Loaders for CloudVolume, HDF5 layers, and image stacks.

Classes
-------
CloudVolumeSequenceGenerator
HDF5SequenceGenerator
ImageSequenceGenerator
"""

import os

from cloudvolume import CloudVolume
from keras import backend as K
from keras.preprocessing.image import Iterator
import numpy as np


class CloudVolumeSequenceGenerator(Iterator):
    """Generate image sequences from a CloudVolume layer.

    Parameters
    ----------
    path : str
        Path to the CloudVolume layer. Should point to the top-level directory
        of the layer.
    nt : int
        The number of frames/images per input. Controls the shape of the z-axis
        of each generated sample.
    batch_size : int
        The number of samples per batch.
    shuffle : bool

    """
    def __init__(self, path, nt, n_channels, xy_shape, batch_size=8, shuffle=False,
                 seed=None, output_mode='error', sequence_start_mode='all',
                 N_seq=None, xy_stride=None,
                 data_format=K.image_data_format()):
        self.X = CloudVolume(path)
        self.nt = nt
        self.n_channels = n_channels
        self.xy_shape = xy_shape
        self.xy_stride = (1, 1) if xy_stride is None else xy_stride
        self.total_stride = self.xy_stride + (self.nt,)
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            self.im_shape = \
                (self.nt, self.X.shape[3], self.X.shape[1], self.X.shape[0])
        else:
            self.im_shape = \
                (self.nt, self.X.shape[1], self.X.shape[0], self.X.shape[3])

        # Compute the number of nt-length sequences of shape xy_shape
        # Later, we will unravel the index to extract specific blocks
        self.blocked_shape = np.asarray([self.X.shape[0] - self.xy_shape[0],
                                         self.X.shape[1] - self.xy_shape[1],
                                         self.X.shape[2] - self.nt]).astype(np.float32)
        self.blocked_shape /= np.asarray([self.xy_stride[0], self.xy_stride[1], self.nt]).astype(np.float32)
        self.blocked_shape = np.ceil(self.blocked_shape).astype(np.int64)
        self.n_blocks = int(np.product(self.blocked_shape))
        self.possible_starts = np.arange(self.n_blocks)

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)

        # print("Possible Starts: ", self.possible_starts)

        if N_seq is not None and len(self.possible_starts) > N_seq:
            self.possible_starts = self.possible_starts[:N_seq]

        self.N_sequences = len(self.possible_starts)
        super(CloudVolumeSequenceGenerator, self).__init__(
            len(self.possible_starts), batch_size, shuffle, seed)

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            # print("Batch Index: ", self.batch_index)
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size, = next(self.index_generator), self.batch_size
        print(self.batch_index, current_index)
        shape = \
            (current_batch_size, self.nt, 1, self.xy_shape[1], self.xy_shape[0]) \
            if self.data_format == 'channels_first' else \
            (current_batch_size, self.nt, self.xy_shape[1], self.xy_shape[0], 1)
        batch_x = np.zeros(shape, dtype=np.float32)

        # print("Index Array: ", index_array[0])

        for i, idx in enumerate(index_array[0]):
            idx = self.possible_starts[idx]
            block_coords = np.unravel_index([idx], self.blocked_shape)
            # print("Block Data: ", idx, self.blocked_shape, block_coords)
            start = np.array([bc[0] * self.total_stride[j] for j, bc in enumerate(block_coords)], dtype=np.int32).ravel()
            end = start + np.asarray(self.xy_shape + (self.nt,))
            # print("Start/End: ", start, end)
            slices = tuple([slice(int(start[j]), int(end[j])) for j in range(3)])
            batch_entry = self.X[slices]
            batch_entry = np.transpose(batch_entry, axes=(2, 3, 1, 0) if self.data_format == 'channels_first' else (2, 1, 0, 3))
            batch_x[i] += self.preprocess(batch_entry)

        if self.output_mode == 'error':
            batch_y = np.zeros(current_batch_size, np.float32)
        else:
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255.0

    def create_all(self):
        # if self.data_format == 'channels_first':
        #     out = np.zeros((self.n_blocks, self.nt, self.n_channels, self.xy_shape[1], self.xy_shape[0]), dtype=np.float32)
        # else:
        #     out = np.zeros((self.n_blocks, self.nt, self.xy_shape[1], self.xy_shape[0], self.n_channels), dtype=np.float32)
        out = None
        for i in range(0, self.n_blocks, self.batch_size):
            if out is not None:
                out = np.concatenate([out, self.next()[0]], axis=0)
            else:
                out = self.next()[0]
        return out


class CloudVolumeSegmentationSequenceGenerator(Iterator):
    """Generate image sequences from a CloudVolume layer.

    Parameters
    ----------
    path : str
        Path to the CloudVolume layer. Should point to the top-level directory
        of the layer.
    nt : int
        The number of frames/images per input. Controls the shape of the z-axis
        of each generated sample.
    batch_size : int
        The number of samples per batch.
    shuffle : bool

    """
    def __init__(self, img_path, seg_path, nt, n_channels, xy_shape,
                 batch_size=8, shuffle=False, seed=None, output_mode='error',
                 sequence_start_mode='all', N_seq=None, xy_stride=None,
                 data_format=K.image_data_format()):
        self.X = CloudVolume(img_path)
        self.X_seg = CloudVolume(seg_path)
        assert all(self.X.shape == self.X_seg.shape), 'image and segmentation volumes must be the same shape'
        self.max_label = np.max(self.X_seg[:])
        # print(self.max_label)
        self.nt = nt
        self.n_channels = n_channels
        self.xy_shape = xy_shape
        self.xy_stride = (1, 1) if xy_stride is None else xy_stride
        self.total_stride = self.xy_stride + (self.nt,)
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode
        if self.data_format == 'channels_first':
            self.im_shape = \
                (self.nt, self.X.shape[3], self.X.shape[1], self.X.shape[0])
        else:
            self.im_shape = \
                (self.nt, self.X.shape[1], self.X.shape[0], self.X.shape[3])

        # Compute the number of nt-length sequences of shape xy_shape
        # Later, we will unravel the index to extract specific blocks
        self.blocked_shape = np.asarray([self.X.shape[0] - self.xy_shape[0],
                                         self.X.shape[1] - self.xy_shape[1],
                                         self.X.shape[2] - self.nt]).astype(np.float32)
        self.blocked_shape /= np.asarray([self.xy_stride[0], self.xy_stride[1], self.nt]).astype(np.float32)
        self.blocked_shape = np.ceil(self.blocked_shape).astype(np.int64)
        self.n_blocks = int(np.product(self.blocked_shape))
        self.possible_starts = np.arange(self.n_blocks)

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)

        # print("Possible Starts: ", self.possible_starts)

        if N_seq is not None and len(self.possible_starts) > N_seq:
            self.possible_starts = self.possible_starts[:N_seq]

        self.N_sequences = len(self.possible_starts)
        super(CloudVolumeSegmentationSequenceGenerator, self).__init__(
            len(self.possible_starts), batch_size, shuffle, seed)

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            # print("Batch Index: ", self.batch_index)
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size, = next(self.index_generator), self.batch_size
        # print(self.batch_index, current_index)
        shape = \
            (current_batch_size, self.nt, self.n_channels + 1, self.xy_shape[1], self.xy_shape[0]) \
            if self.data_format == 'channels_first' else \
            (current_batch_size, self.nt, self.xy_shape[1], self.xy_shape[0], self.n_channels + 1)
        batch_x = np.zeros(shape, dtype=np.float32)

        # print("Index Array: ", index_array[0])

        for i, idx in enumerate(index_array[0]):
            idx = self.possible_starts[idx]
            block_coords = np.unravel_index([idx], self.blocked_shape)
            # print("Block Data: ", idx, self.blocked_shape, block_coords)
            start = np.array([bc[0] * self.total_stride[j] for j, bc in enumerate(block_coords)], dtype=np.int32).ravel()
            end = start + np.asarray(self.xy_shape + (self.nt,))
            # print("Start/End: ", start, end)
            slices = tuple([slice(int(start[j]), int(end[j])) for j in range(3)])
            batch_entry = self.X[slices]
            batch_entry = np.transpose(batch_entry, axes=(2, 3, 1, 0) if self.data_format == 'channels_first' else (2, 1, 0, 3))
            batch_x[i, :, :, :, :-1] += self.preprocess(batch_entry)
            batch_entry = self.X_seg[slices]
            batch_entry = np.transpose(batch_entry, axes=(2, 3, 1, 0) if self.data_format == 'channels_first' else (2, 1, 0, 3))
            batch_x[i, :, :, :, -1:] += self.preprocess(batch_entry, norm_val=self.max_label)

        if self.output_mode == 'error':
            batch_y = np.zeros(current_batch_size, np.float32)
        else:
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X, norm_val=255.0):
        return X.astype(np.float32) / float(norm_val)

    def create_all(self):
        # if self.data_format == 'channels_first':
        #     out = np.zeros((self.n_blocks, self.nt, self.n_channels, self.xy_shape[1], self.xy_shape[0]), dtype=np.float32)
        # else:
        #     out = np.zeros((self.n_blocks, self.nt, self.xy_shape[1], self.xy_shape[0], self.n_channels), dtype=np.float32)
        out = None
        for i in range(0, self.n_blocks, self.batch_size):
            if out is not None:
                out = np.concatenate([out, self.next()[0]], axis=0)
            else:
                out = self.next()[0]
        return out


if __name__ == '__main__':
    path = 'file:///scratch0/ygw01/precomputed/image'

    loader = CloudVolumeSequenceGenerator(path, 10, 1, (64, 64), xy_stride=(48, 48), batch_size=1)

    cv = CloudVolume(path)

    for i in range(0, cv.shape[0] - 64, 48):
        endi = i + 64
        if endi > cv.shape[0]:
            endi = cv.shape[0]
        for j in range(0, cv.shape[1] - 64, 48):
            endj = j + 64
            if endj > cv.shape[1]:
                endj = cv.shape[1]
            for k in range(0, cv.shape[2] - 10, 10):
                endk = k + 10
                if endk > cv.shape[2]:
                    endk = cv.shape[2]
                loaded, _ = loader.next()
                # print(loaded.nonzero())
                correct = cv[i:endi, j:endj, k:endk]
                print((i, j, k), (endi, endj, endk))
                correct = np.transpose(correct, axes=[2, 3, 1, 0])
                correct = np.array([correct.astype(np.float32) / 255.0])
                # print(loaded)
                # print(correct)
                assert np.all(loaded == correct)
