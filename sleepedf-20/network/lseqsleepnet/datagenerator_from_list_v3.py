# Sequence data generation for sequence-to-sequence sleep staging.
# X1: raw EEG (3000, num_epoch) at 100Hz
# X2: EEG spectrogram (129, 29, num_epoch)
# X_mel: mel spectrogram from sonified EEG (n_mels, 29, num_epoch)

import os
import numpy as np
import h5py
import librosa

class DataGenerator3:
    def __init__(self, list_of_files, file_sizes,
                 data_shape_2=np.array([29, 129]),
                 seq_len=200,
                 sr=100,
                 n_mels=64):
        '''
        Mini-batch data generator with mel spectrogram from sonified EEG.
        Args:
            list_of_files : list of paths to .mat data files
            file_sizes    : number of sleep epochs per file
            data_shape_2  : [time_frames, freq_bins] of one EEG spectrogram epoch
            seq_len       : total sequence length (nsubseq * sub_seq_len)
            sr            : original EEG sampling rate (100Hz)
            n_mels        : number of mel frequency bins (64)
        '''
        self.list_of_files = list_of_files
        self.file_sizes = file_sizes
        self.data_shape_2 = data_shape_2      # [29, 129]
        self.X2 = None
        self.X_mel = None                      # mel spectrogram branch
        self.y = None
        self.label = None
        self.boundary_index = np.array([])
        self.seq_len = seq_len
        self.Ncat = 5
        self.pointer = 0
        self.data_index = None
        self.data_size = np.sum(self.file_sizes)

        self.sr = sr
        self.n_mels = n_mels
        self.target_sr = 8000               # sonification sample rate
        self.mel_time_frames = data_shape_2[0]  # 29 — match original spectrogram
        # mel spectrogram shape per epoch: (mel_time_frames, n_mels) = (29, 64)
        self.mel_shape = np.array([self.mel_time_frames, self.n_mels])

        self.read_mat_filelist()

    def read_mat_filelist(self):
        self.X2 = np.ndarray(
            [self.data_size, self.data_shape_2[0], self.data_shape_2[1]],
            dtype=np.float32)
        self.X_mel = np.ndarray(
            [self.data_size, self.mel_shape[0], self.mel_shape[1]],
            dtype=np.float32)
        self.y = np.ndarray([self.data_size, self.Ncat], dtype=np.float32)
        self.label = np.ndarray([self.data_size], dtype=np.float32)

        count = 0
        for i in range(len(self.list_of_files)):
            X2, X_mel, y, label = self.read_mat_file(self.list_of_files[i].strip())
            n = len(X2)
            self.X2[count:count + n] = X2.astype(np.float32)
            self.X_mel[count:count + n] = X_mel.astype(np.float32)
            self.y[count:count + n] = y.astype(np.float32)
            self.label[count:count + n] = label.astype(np.float32)
            self.boundary_index = np.append(
                self.boundary_index,
                np.arange(count, count + self.seq_len - 1))
            count += n

        self.data_index = np.arange(len(self.X2))
        mask = ~np.isin(self.data_index, self.boundary_index)
        self.data_index = self.data_index[mask]

    def _extract_mel_spectrogram(self, x1_all, cache_path=None):
        """
        Sonification pipeline:
        1. Check cache -> load instantly
        2. Upsample EEG 100Hz -> 8000Hz (sonification)
        3. Compute mel spectrogram with same number of time frames as original (29)
        4. Save cache

        Output shape per epoch: (mel_time_frames=29, n_mels=64)
        """
        if cache_path is not None and os.path.exists(cache_path):
            mel = np.load(cache_path)
            print(f"  Loaded mel spectrogram from cache: {cache_path}")
            return mel.astype(np.float32)

        print(f"  Extracting sonified mel spectrograms (will cache)...")
        num_epochs = x1_all.shape[0]
        mel_feats = np.zeros((num_epochs, self.mel_time_frames, self.n_mels),
                             dtype=np.float32)

        # Hop length to get exactly mel_time_frames=29 frames
        # Signal length after upsampling: 3000 * (8000/100) = 240000 samples
        signal_len = int(3000 * (self.target_sr / self.sr))
        hop_length = signal_len // self.mel_time_frames  # ~8275

        for i in range(num_epochs):
            eeg = x1_all[i].astype(np.float32)

            # Sonification: upsample EEG from 100Hz to 8000Hz
            eeg_sonified = librosa.resample(eeg, orig_sr=self.sr,
                                            target_sr=self.target_sr)

            # Mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=eeg_sonified,
                sr=self.target_sr,
                n_mels=self.n_mels,
                hop_length=hop_length,
                fmax=self.target_sr // 2)

            # Convert to log scale (dB)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Ensure exactly mel_time_frames time frames
            if mel_db.shape[1] >= self.mel_time_frames:
                mel_db = mel_db[:, :self.mel_time_frames]
            else:
                # pad if short
                pad = self.mel_time_frames - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='edge')

            # Transpose to (time_frames, n_mels) = (29, 64)
            mel_feats[i] = mel_db.T

        if cache_path is not None:
            np.save(cache_path, mel_feats)
            print(f"  Saved mel spectrogram to cache: {cache_path}")

        return mel_feats

    def read_mat_file(self, filename):
        with h5py.File(filename, 'r') as data:
            # EEG spectrogram: (129, 29, num_epoch) -> (num_epoch, 29, 129)
            X2 = np.array(data['X2'])
            X2 = np.transpose(X2, (2, 1, 0))

            # Raw EEG: (3000, num_epoch) -> (num_epoch, 3000)
            X1 = np.array(data['X1'])
            X1 = np.transpose(X1, (1, 0))

            y = np.array(data['y'])
            y = np.transpose(y, (1, 0))

            label = np.array(data['label'])
            label = np.transpose(label, (1, 0))
            label = np.squeeze(label)

        # Use .mel_feats.npy suffix for cache
        cache_path = filename.replace('.mat', '.mel_feats.npy')
        X_mel = self._extract_mel_spectrogram(X1, cache_path=cache_path)

        return X2, X_mel, y, label

    def normalize(self, meanX2, stdX2):
        """Normalize original EEG spectrogram."""
        X2 = self.X2
        X2 = np.reshape(X2, (self.data_size * self.data_shape_2[0],
                              self.data_shape_2[1]))
        X2 = (X2 - meanX2) / stdX2
        self.X2 = np.reshape(X2, (self.data_size,
                                   self.data_shape_2[0],
                                   self.data_shape_2[1]))

    def normalize_mel(self, mean_mel, std_mel):
        """Normalize mel spectrogram."""
        X_mel = self.X_mel
        X_mel = np.reshape(X_mel, (self.data_size * self.mel_shape[0],
                                    self.mel_shape[1]))
        X_mel = (X_mel - mean_mel) / (std_mel + 1e-8)
        self.X_mel = np.reshape(X_mel, (self.data_size,
                                         self.mel_shape[0],
                                         self.mel_shape[1]))

    def shuffle_data(self):
        idx = np.random.permutation(len(self.data_index))
        self.data_index = self.data_index[idx]

    def reset_pointer(self):
        self.pointer = 0

    def next_batch(self, batch_size):
        data_index = self.data_index[self.pointer:self.pointer + batch_size]
        self.pointer += batch_size

        # (batch, seq_len, time_frames, freq_bins)
        batch_x2 = np.ndarray(
            [batch_size, self.seq_len, self.data_shape_2[0], self.data_shape_2[1]],
            dtype=np.float32)
        # (batch, seq_len, mel_time_frames, n_mels)
        batch_mel = np.ndarray(
            [batch_size, self.seq_len, self.mel_shape[0], self.mel_shape[1]],
            dtype=np.float32)
        batch_y = np.ndarray(
            [batch_size, self.seq_len, self.y.shape[1]],
            dtype=np.float32)
        batch_label = np.ndarray(
            [batch_size, self.seq_len],
            dtype=np.float32)

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                idx = data_index[i] - (self.seq_len - 1) + n
                batch_x2[i, n] = self.X2[idx].squeeze()
                batch_mel[i, n] = self.X_mel[idx]
                batch_y[i, n] = self.y[idx]
                batch_label[i, n] = self.label[idx]

        return batch_x2, batch_mel, batch_y, batch_label

    def rest_batch(self, batch_size):
        data_index = self.data_index[self.pointer:len(self.data_index)]
        actual_len = len(self.data_index) - self.pointer
        self.pointer = len(self.data_index)

        batch_x2 = np.ndarray(
            [actual_len, self.seq_len, self.data_shape_2[0], self.data_shape_2[1]],
            dtype=np.float32)
        batch_mel = np.ndarray(
            [actual_len, self.seq_len, self.mel_shape[0], self.mel_shape[1]],
            dtype=np.float32)
        batch_y = np.ndarray(
            [actual_len, self.seq_len, self.y.shape[1]],
            dtype=np.float32)
        batch_label = np.ndarray(
            [actual_len, self.seq_len],
            dtype=np.float32)

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                idx = data_index[i] - (self.seq_len - 1) + n
                batch_x2[i, n] = self.X2[idx].squeeze()
                batch_mel[i, n] = self.X_mel[idx]
                batch_y[i, n] = self.y[idx]
                batch_label[i, n] = self.label[idx]

        return actual_len, batch_x2, batch_mel, batch_y, batch_label