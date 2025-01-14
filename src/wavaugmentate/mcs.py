#!/usr/bin/env python3

"""
This module defines multichannel audio flies augmentation class
MultiChannelSignal.
"""

from __future__ import annotations
import copy
import random
import logging as log
import numpy as np
from scipy.io import wavfile

ERROR_MARK = "Error: "
SUCCESS_MARK = "Done."

# Default sampling frequency, Hz.
DEF_FS = 44100

# Default signal durance in seconds.
DEF_SIGNAL_LEN = 5

random_noise_gen = np.random.default_rng()


def pause_measure(mask: np.ndarray[int]) -> list:
    """
    Measures pauses in multichannel sound.

    Args:
        mask (np.ndarray): A mask indicating the pauses in the multichannel
        sound.

    Returns:
        list: A list of lists containing pairs of (index, length) of pauses for
        each channel.  Length is in samples."""

    n_channels = mask.shape[0]
    pause_list = []
    out_list = []
    index = 0
    for i in range(0, n_channels):
        zero_count = 0
        prev_val = 1
        for j in range(0, mask.shape[1]):
            val = mask[i][j]
            if val == 0:
                if prev_val == 1:
                    index = j
                zero_count += 1
            else:
                if prev_val == 0:
                    pause_list.append((index, zero_count))
                    zero_count = 0
            prev_val = val
        out_list.append(pause_list)
        pause_list = []

    return out_list


class MultiChannelSignal:
    """
    Class provides support of multichannel sound
    data sintesized or loaded from WAV file.
    """

    def __init__(
        self, np_data: np.ndarray = None, sampling_rate: int = -1,
        seed: int = -1
    ):
        """
        Initializes a new instance of the MultiChannelSignal class.

        Args:
            np_data (np.ndarray, optional): The multichannel sound data.
            Defaults to None.
            sampling_rate (int, optional): The sample rate of the sound data.
            Defaults to -1.
            seed (int, optional): Value for seeding random generator.
             Defaults to -1.

        Returns:
            None
        """

        if np_data is None:
            self.data = None  # np.ndarray: Multichannel sound data field.
        else:
            self.data = (
                np_data  # .copy()
            )  # np.ndarray: Multichannel sound data field.
        self.path = ""  # Path to the sound file, from which the data was read.
        self.sample_rate = sampling_rate  # Sampling frequency, Hz.
        self.seed = seed  # Flag for seeding random generator.

    def copy(self) -> MultiChannelSignal:
        """Make deep copy of the MultiChannelSignal object."""

        return copy.deepcopy(self)

    def _channel_rms(
                     self, chan_index: int, last_index_of_sample: int = -1,
                     decimals: int = -1) -> float:
        """
        Calculate the root mean square (RMS) of a single channel signal.

        Args:
            chan_index (int): Index of the channel to calculate the RMS of.
            last_index_of_sample (int): The last index to consider when
              calculating RNS.  If -1, consider the entire array.
            decimals (int): Number of decimal places to round the RMS value.

        Returns:
            float: The RMS value of the input signal.
        """

        # Calculate the mean of the squares of the signal values
        mean_square = 0
        if chan_index > -1:
            mean_square = np.mean(self.data[chan_index,
                                            0:last_index_of_sample] ** 2)
        else:
            mean_square = np.mean(self.data[0:last_index_of_sample] ** 2)

        # Calculate the square root of the mean of the squares
        single_chan_rms = np.sqrt(mean_square)

        # Round the result to the specified number of decimal places
        if decimals > 0:
            single_chan_rms = round(single_chan_rms, decimals)

        # Return the result
        return single_chan_rms

    def rms(self, last_index_of_sample: int = -1, decimals: int = -1):
        """
        Calculate the root mean square (RMS) of a multichannel sound.

        Args:
            last_index_of_sample (int): The last index to consider when
             calculating the RMS.  If -1, consider the entire array.
             Defaults to -1.
            decimals (int): Number of decimal places to round the RMS value.
             If -1, do not round. Defaults to -1.

        Returns:
            list: A list of RMS values for each channel in the multichannel
            sound.
        """

        res = []
        shape_len = len(self.data.shape)
        if shape_len > 1:
            for i in range(0, self.data.shape[0]):
                chan_rms = self._channel_rms(i, last_index_of_sample, decimals)
                res.append(chan_rms)
        else:
            chan_rms = self._channel_rms(-1, last_index_of_sample, decimals)
            res.append(chan_rms)
        return res

    def shape(self) -> tuple:
        """
        Returns the shape of the multichannel sound object 'data' field.

        Returns:
            tuple: A tuple containing the shape of the multichannel sound data.
        """
        return self.data.shape

    def generate(
        self,
        frequency_list: list[int],
        duration: float = DEF_SIGNAL_LEN,
        sampling_rate: int = -1,
        mode="sine",
    ) -> MultiChannelSignal:
        """
        Generate a multichannel sound based on the given frequency list,
        duration, sample rate, and mode. The mode can be 'sine' or 'speech'. In
        'sine' mode, output multichannel sound will be a list of sine waves. In
        'speech' mode, output will be a list of speech like signals. In this
        mode input frequencies list will be used as basic tone frequency for
        corresponding channel, it should be in interval 600..300.

        Args:
            frequency_list (list): A list of frequencies to generate sound for.
            duration (float): The duration of the sound in seconds.
            sampling_rate (int): The sample rate of the sound. Defaults to -1.
            mode (str): The mode of sound generation. Can be 'sine' or 'speech'.
        Defaults to 'sine'. Mode 'speech' generates speech like signals.

        Returns:
            self (MultiChannelSignal): representing the generated multichannel
         sound.
        """

        if sampling_rate > 0:
            self.sample_rate = sampling_rate
        self.data = None
        samples = np.arange(duration * self.sample_rate) / self.sample_rate
        channels = []
        if mode == "sine":
            for freq in frequency_list:
                signal = np.sin(2 * np.pi * freq * samples)
                signal = np.float32(signal)
                channels.append(signal)
            self.data = np.array(channels)

        if mode == "speech":
            if self.seed != -1:
                random.seed(self.seed)
            for freq in frequency_list:
                if freq > 300 or freq < 60:
                    msg = "Use basic tone from interval 600..300 Hz."
                    log.error(msg)
                    raise ValueError(msg)

                # Formants:
                fbt = random.randint(freq, 300)  # 60–300 Гц
                freq_list = [
                    fbt,
                    random.randint(2 * fbt, 850),  # 150–850 Гц
                    random.randint(3 * fbt, 2500),  # 500–2500 Гц
                    random.randint(4 * fbt, 3500),  # 1500–3500 Гц
                    random.randint(5 * fbt, 4500)  # 2500–4500 Гц
                ]
                signal = 0
                amp = 1
                for frm in freq_list:
                    signal += amp * np.sin(2 * np.pi * frm * samples)
                    amp -= 0.1
                peak_amplitude = np.max(np.abs(signal))
                signal = signal / peak_amplitude
                signal = np.float32(signal)
                channels.append(signal)
                self.data = np.array(channels)
        return self

    def write(self, dest_path: str) -> MultiChannelSignal:
        """
        Writes the given multichannel sound data to a WAV file at the specified
        path.

        Args:
            dest_path (str): The path to the WAV file.

        Returns:
        self (MultiChannelSignal):  representing saved multichannel sound.
        """

        buf = self.data.T
        wavfile.write(dest_path, self.sample_rate, buf)
        return self

    def write_by_channel(self, dest_path: str) -> MultiChannelSignal:
        """
        Writes each channel of the multichannel sound data to a separate WAV
        files, 1 for each channel.

        File name will be modified to include the channel number. If path
        contains
        ./outputwav/sound_augmented.wav the output file names will be
        ./outputwav/sound_augmented_1.wav
        ./outputwav/sound_augmented_2.wav and so on.

        Args:
            dest_path (str): The path to the WAV file. The filename will be
            modified to include the channel number.

        Returns:
            self (MultiChannelSignal): The instance itself, allowing for
            method chaining.
        """

        trimmed_path = dest_path.split(".wav")
        for i in range(self.channels_count()):
            buf = self.data[i, :].T
            file_name = f"{trimmed_path[0]}_{i + 1}.wav"
            log.info("Writing %s...", file_name)
            wavfile.write(file_name, self.sample_rate, buf)
        return self

    def read(self, source_path: str) -> MultiChannelSignal:
        """
        Reads a multichannel sound from a WAV file.

        Args:
            source_path (str): The path to the WAV file.

        Returns:
            self(MultiChannelSignal): An object MultiChannelSignal containing
             the sample rate and the multichannel sound data.
        """

        self.sample_rate, buf = wavfile.read(source_path)
        if len(buf.shape) != 2:
            buf = np.expand_dims(buf, axis=1)
        self.path = source_path
        self.data = buf.T
        return self

    # Audio augmentation functions

    def pause_detect(self, relative_level: list[float]) -> np.ndarray[int]:
        """
            Detects pauses in a multichannel sound.

            Args:
            relative_level (list[float]): The list of relative levels for each
            channel, signal below this level will be marked as pause.

        Returns:
            np.ndarray[int]: The mask indicating the pauses in the multichannel
            sound.  The mask has the same shape as the input sound. It contains
            zeros and ones 0 - pause, 1 - not a pause.
        """

        rms_list = self.rms()
        modules_list = abs(self.data)
        mask = np.zeros(self.data.shape)

        for i in range(0, self.data.shape[0]):
            threshold = rms_list[i] * relative_level[i]
            mask[i] = np.clip(modules_list[i], a_min=threshold,
                              a_max=1.1 * threshold)
            mask[i] -= threshold
            mask[i] /= 0.09 * threshold
            mask[i] = np.clip(mask[i], a_min=0, a_max=1).astype(int)
        return mask

    def pause_shrink(self, mask: np.ndarray[int],
                     min_pause: list[int]) -> MultiChannelSignal:
        """
        Shrink pauses in multichannel sound.

        Args:
            mask (np.ndarray[int]): The mask indicating the pauses in the
            multichannel sound.
            min_pause (list[int]): The list of minimum pause lengths for
            each channel in object.

        Returns:
            self (MultiChannelSignal): The MultiChannelSignal object with
             pauses shrunk.
        """

        chans = self.data.shape[0]
        out_data = np.zeros_like(self.data, dtype=np.float32)
        for i in range(0, chans):
            k = 0
            zero_count = 0
            for j in range(0, self.data.shape[1]):
                if mask[i][j] == 0:
                    zero_count += 1
                    if zero_count < min_pause[i]:
                        out_data[i][k] = self.data[i][j]
                        k += 1
                else:
                    zero_count = 0
                    out_data[i][k] = self.data[i][j]
                    k += 1
        self.data = out_data
        return self

    def channels_count(self) -> int:
        """Returns the number of channels in the MultiChannelSignal object."""

        if self.data is None:
            return 0

        if len(self.data.shape) > 1:
            return self.data.shape[0]

        return 1

    def channels_len(self) -> int:
        """
        Returns the number of samples in one channel of MultiChannelSignal
        object.
        """

        if self.data is None:
            return 0

        if len(self.data.shape) > 1:
            return self.data.shape[1]

        return len(self.data)

    def split(self, channels_count: int) -> MultiChannelSignal:
        """
        Splits a multichannel signal (containing single channel) into multiple
        identical channels.

        Args:
            channels_count (int): The number of channels to split the signal
            into.

        Returns:
            self (MultiChannelSignal): The split multichannel signal, with
              each channel identical.
        """

        if self.channels_count() > 1:
            msg = "Can't split more than 1 channel signal."
            log.error(msg)
            raise ValueError(msg)

        out_data = None

        if len(self.data.shape) > 1:
            out_data = np.zeros(
                (channels_count, self.data.shape[1]), dtype=np.float32
            )
        else:
            out_data = np.zeros(
                (channels_count, len(self.data)), dtype=np.float32
            )

        for i in range(0, channels_count):
            out_data[i] = self.data
        self.data = out_data
        return self

    def merge(self) -> MultiChannelSignal:
        """
            Merges channels of MultiChannelSignal object into a single channel.

        Args:
            none

        Returns:
            self (MultiChannelSignal): The MultiChannelSignal object containing
              a single channel of merging result.
        """

        out_data = np.zeros(self.data.shape[1], dtype=np.float32)
        channels_count = self.data.shape[0]
        for i in range(0, channels_count):
            out_data += self.data[i]
        self.data = out_data
        return self

    def sum(self, signal2: MultiChannelSignal) -> MultiChannelSignal:
        """
        Sums two multichannel signals.

        Args:
            signal2 (MultiChannelSignal): The second multichannel sound
              signal.

        Returns:
            self (MultiChannelSignal): The sum of self.data and signal2.data
              signals as MultiChannelSignal.
        """

        out_data = self.data + signal2.data
        self.data = out_data
        return self

    def side_by_side(self,
                     signal2: MultiChannelSignal) -> MultiChannelSignal:
        """
        Concatenates two multichannel sound signals side by side.

        Args:
            signal2 (MultiChannelSignal): The second multichannel sound
              signal.

        Returns:
            self (MultiChannelSignal): The concatenated sound signal
              containing channels of both MultiChannelSignal objects.
        """

        out_data = np.zeros(
            (self.data.shape[0] + signal2.data.shape[0], self.data.shape[1]),
            dtype=np.float32,
        )
        out_data[0 : self.data.shape[0], :] = self.data
        out_data[self.data.shape[0] :, :] = signal2.data
        self.data = out_data
        return self

    def put(self, signal: MultiChannelSignal) -> MultiChannelSignal:
        """
        Updates the multichannel sound data and sample rate of the
        MultiChannelSignal instance.

        Args:
            signal (MultiChannelSignal): source of multichannel sound data.

        Returns:
            self (MultiChannelSignal): The updated MultiChannelSignal instance.
        """

        self.data = signal.data
        self.sample_rate = signal.sample_rate
        self.path = signal.path
        return self

    def get(self) -> np.ndarray:
        """
        Returns the multichannel sound data stored in the MultiChannelSignal
        instance.

        Returns:
            np.ndarray: The multichannel sound data.
        """
        return self.data

    def set_seed(self, seed: int = -1):
        """Set seeding value."""

        self.seed = seed

    def info(self) -> dict:
        """
        Returns a dictionary containing metadata about the audio data.

            The dictionary contains the following information:

            * path: The file path where the audio data was loaded from.
            * channels_count: The number of audio channels in the data
              (1 for mono, 2 and more for stereo and other).
            * sample_rate: The sampling rate at which the audio data is stored.
            * length_s: The duration of the audio data in seconds.

            If the data is not loaded, the `path`, `channels_count`, and
            `length_s` values will be -1. Otherwise,
            they will be populated with actual values.

        Returns:
        dict: A dictionary containing metadata about the audio data.
        """

        res = {
            "path": self.path,
            "channels_count": -1,
            "sample_rate": self.sample_rate,
            "length_s": -1,
        }
        if self.data is not None:
            length = self.data.shape[1] / self.sample_rate
            res["channels_count"] = self.channels_count()
            res["length_s"] = length
        return res

    # Alias Method Names
    rd = read
    wr = write
    wrbc = write_by_channel
    mrg = merge
    splt = split
    sbs = side_by_side
    pdt = pause_detect
    gen = generate
    cpy = copy
