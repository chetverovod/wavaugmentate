#!/usr/bin/env python3

"""
This module does multichannel audio flies augmentation.
"""

import argparse
import copy
from pathlib import Path
import os
import sys
import random
from scipy.io import wavfile
import numpy as np

# Default sampling frequency, Hz.
DEF_FS = 44100

random_noise_gen = np.random.default_rng()

class Mcs:
    """
    Class provides support of  multichannel sound
    data.
    """

    def __init__(self, mcs_data: np.ndarray = None, fs: int = -1):
        """
        Initializes a new instance of the Mcs class.

        Args:
            mcs_data (np.ndarray, optional): The multichannel sound data.
            Defaults to None.
            fs (int, optional): The sample rate of the sound data. Defaults
            to -1.

        Returns:
            None
        """

        self.data = mcs_data  # Multichannel sound data field.
        self.path = ""  # Path to the sound file, from which the data was read.
        self.sample_rate = fs  # Sampling frequency, Hz.

      
       

def _single_rms(signal_of_channel: np.ndarray, decimals: int) -> float:
    """
    Calculate the root mean square (RMS) of a single channel signal.

    Parameters:
        signal_of_channel (array): Input signal of a single channel.
        decimals (int): Number of decimal places to round the RMS value.

    Returns:
        float: The RMS value of the input signal.
    """

    r = np.sqrt(np.mean(signal_of_channel**2))
    if decimals > 0:
        r = round(r, decimals)
    return r


def rms(mcs_data: np.ndarray, last_index: int = -1, decimals: int = -1):
    """
    Calculate the root mean square (RMS) of a multichannel sound.

    Parameters:
        mcs_data (array): Input multichannel sound data.
        last_index (int): The last index to consider when calculating the RMS.
            If -1, consider the entire array. Defaults to -1.
        decimals (int): Number of decimal places to round the RMS value.
            If -1, do not round. Defaults to -1.

    Returns:
        list: A list of RMS values for each channel in the multichannel sound.
    """
    res = []
    shlen = len(mcs_data.shape)
    if shlen > 1:
        for i in range(0, mcs_data.shape[0]):
            ch = mcs_data[i]
            r = _single_rms(ch[0:last_index], decimals)
            res.append(r)
    else:
        r = _single_rms(mcs_data[0:last_index], decimals)
        res.append(r)
    return res


def generate(
    frequency_list: list[100, 200, 300, 400],
    duration: float,
    sample_rate=DEF_FS,
    mode="sine",
    seed: int = -1,
):
    """
    Generate a multichannel sound based on the given frequency list, duration,
    sample rate, and mode. The mode can be 'sine' or 'speech'. In 'sine' mode,
    output multichannel sound will be a list of sine waves. In 'speech' mode,
    output will be a list of speech like signals. In this mode input
    frequencies list will be used as basic tone frequency for corresponding
    channel, it should be in interval 600..300.

    Args:
    frequency_list (list): A list of frequencies to generate sound for.
    duration (float): The duration of the sound in seconds.
    sample_rate (int): The sample rate of the sound. Defaults to def_fs.
    mode (str): The mode of sound generation. Can be 'sine' or 'speech'.
    Defaults to 'sine'.
    seed (int): The seed for random number generation. Defaults to -1.

    Returns:
    multichannel_sound (numpy array): A numpy array representing the generated
    multichannel sound.
    """

    samples = np.arange(duration * sample_rate) / sample_rate
    channels = []
    if mode == "sine":
        for f in frequency_list:
            signal = np.sin(2 * np.pi * f * samples)
            signal = np.float32(signal)
            channels.append(signal)
            multichannel_sound = np.array(channels).copy()

    if mode == "speech":
        if seed != -1:
            random.seed(seed)
        for f in frequency_list:
            if f > 300 or f < 60:
                print(error_mark + "Use basic tone from interval 600..300 Hz")
                sys.exit(1)
            # Formants:
            fbt = random.randint(f, 300)  # 60–300 Гц
            frm1 = random.randint(2 * fbt, 850)  # 150–850 Гц
            frm2 = random.randint(3 * fbt, 2500)  # 500–2500 Гц
            frm3 = random.randint(4 * fbt, 3500)  # 1500–3500 Гц
            frm4 = random.randint(5 * fbt, 4500)  # 2500–4500 Гц
            freq_list = [fbt, frm1, frm2, frm3, frm4]
            signal = 0
            amp = 1
            for frm in freq_list:
                signal += amp * np.sin(2 * np.pi * frm * samples)
                amp -= 0.1
            p = np.max(np.abs(signal))
            signal = signal / p
            signal = np.float32(signal)
            channels.append(signal)
            multichannel_sound = np.array(channels).copy()

    return multichannel_sound


def write(path: str, mcs_data: np.ndarray, sample_rate: int = 44100):
    """
    Writes the given multichannel sound data to a WAV file at the specified
    path.

    Args:
        path (str): The path to the WAV file.
        mcs_data (np.ndarray): The multichannel sound data to write. The shape
        of the array should be (num_channels, num_samples).  sample_rate (int,
        optional): The sample rate of the sound data. Defaults to 44100.

    Returns:
        None
    """

    buf = mcs_data.T.copy()
    wavfile.write(path, sample_rate, buf)


def read(path: str) -> tuple[int, np.ndarray]:
    """
    Reads a multichannel sound from a WAV file.

    Args:
        path (str): The path to the WAV file.

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate and the
        multichannel sound data.
    """

    sample_rate, buf = wavfile.read(path)
    mcs_data = buf.T.copy()
    return sample_rate, mcs_data


def file_info(path: str) -> dict:
    """
    Returns a dictionary containing information about a WAV file.

    Args:
        path (str): The path to the WAV file.

    Returns:
        dict: A dictionary containing the following keys:
            - path (str): The path to the WAV file.
            - channels_count (int): The number of channels in the WAV file.
            - sample_rate (int): The sample rate of the WAV file.
            - length_s (float): The length of the WAV file in seconds.
    """

    sample_rate, buf = wavfile.read(path)
    length = buf.shape[0] / sample_rate

    return {
        "path": path,
        "channels_count": buf.shape[1],
        "sample_rate": sample_rate,
        "length_s": length,
    }


# Audio augmentation functions


def amplitude_ctrl(
    mcs_data: np.ndarray, amplitude_list: list[float]
) -> np.ndarray:
    """
    Apply amplitude control to a multichannel sound.

    Args:
        mcs_data (np.ndarray): The multichannel sound data.
        amplitude_list (list[float]): The list of amplitude coefficients to
        apply to each channel.

    Returns:
        np.ndarray: The amplitude-controlled multichannel sound.
    """

    channels = []
    for signal, amplitude in zip(mcs_data, amplitude_list):
        channels.append(signal * amplitude)
        multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def delay_ctrl(
    mcs_data: np.ndarray, delay_us_list: list[int], sampling_rate: int = DEF_FS
) -> np.ndarray:
    """
    Add delays of channels of multichannel sound. Output data become longer.
    Values of delay will be converted to count of samples.

    Parameters:
        mcs_data (np.ndarray): The multichannel sound data.
        delay_us_list (list[int]): The list of delay values in microseconds to
        apply to each channel. Each value should be a positive integer.
        sampling_rate (int): The sampling rate of the
        sound data. Defaults to def_fs.

    Returns:
        np.ndarray: The delayed multichannel sound.
    """

    channels = []
    # In samples.
    max_samples_delay = int(max(delay_us_list) * 1.0e-6 * sampling_rate)

    for signal, delay in zip(mcs_data, delay_us_list):
        samples_delay = int(delay * 1.0e-6 * sampling_rate)  # In samples.
        res = np.zeros(samples_delay)
        res = np.append(res, signal)
        if samples_delay < max_samples_delay:
            res = np.append(res, np.zeros(max_samples_delay - samples_delay))
        channels.append(res)
        multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def echo_ctrl(
    mcs_data,
    delay_us_list: list[int],
    amplitude_list: list[float],
    sampling_rate: int = DEF_FS,
) -> np.ndarray:
    """
    Add echo to multichannel sound. The output data become longer. To each
    channel will be added it's copy with corresponding delay delay and
    amplitude.  It looks like acoustic wave was reflected from the hard wall.

    Parameters:
        mcs_data (np.ndarray): The multichannel sound data.
        delay_us_list (list[int]): The list of delay values in microseconds to
            apply to each channel. Each value should be a positive integer.
        amplitude_list (list[float]): The list of amplitude coefficients to
            apply to each channel.
        sampling_rate (int): The sampling rate of the sound data. Defaults to
        def_fs.

    Returns:
        np.ndarray: The echoed multichannel sound.
    """

    a = amplitude_ctrl(mcs_data, amplitude_list)
    e = delay_ctrl(a, delay_us_list, sampling_rate)
    channels = []
    for d in mcs_data:
        zl = e.shape[1] - d.shape[0]
        channels.append(np.append(d, np.zeros(zl)))
    multichannel_sound = np.array(channels).copy() + e

    return multichannel_sound


def noise_ctrl(
    mcs_data: np.ndarray,
    noise_level_list: list[float],
    seed: int = -1,
) -> np.ndarray:
    """
    Apply noise to a multichannel sound.

    Parameters:
        mcs_data (np.ndarray): The multichannel sound data.
        noise_level_list (list[float]): The list of noise levels to apply to
        each channel.
        seed (int): The seed for random number generation. Defaults to -1.

    Returns:
        np.ndarray: The noise-controlled multichannel sound.
    """

    channels = []
    for signal, level in zip(mcs_data, noise_level_list):
        if seed != -1:
            random.seed(seed)
            n_noise = random_noise_gen.standard_normal(mcs_data.shape[1],)
        else:
            # TODO seed should be fixed for repeatable results
            n_noise = random_noise_gen.standard_normal(mcs_data.shape[1])
        noise = n_noise
        res = signal + level * noise
        channels.append(res)
    multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def pause_detect(
    mcs_data: np.ndarray, relative_level: list[float]
) -> np.ndarray[int]:
    """
    Detects pauses in a multichannel sound.

    Parameters:
        mcs_data (np.ndarray): The multichannel sound data.
        relative_level (list[float]): The list of relative levels for each
        channel, signal below this level will be marked as pause.

    Returns:
        np.ndarray: The mask indicating the pauses in the multichannel sound.
        The mask has the same shape as the input sound. It contains zeros and
        ones 0 - pause, 1 - not a pause.
    """

    r = rms(mcs_data)
    a = abs(mcs_data)
    mask = np.zeros(mcs_data.shape)

    for i in range(0, mcs_data.shape[0]):
        ll = r[i] * relative_level[i]
        mask[i] = np.clip(a[i], a_min=ll, a_max=1.1 * ll)
        mask[i] -= ll
        mask[i] /= 0.09 * ll
        mask[i] = np.clip(mask[i], a_min=0, a_max=1).astype(int)
    return mask


def pause_shrink(
    mcs_data: np.ndarray, mask: np.ndarray[int], min_pause: list[int]
) -> np.ndarray:
    """
    Shrink pauses in multichannel sound.

    Parameters:
        mcs_data (np.ndarray): The multichannel sound data.
        mask (np.ndarray): The mask indicating the pauses in the multichannel
        sound.
        min_pause (list[int]): The list of minimum pause lengths for
        each channel in samples.

    Returns:
        np.ndarray: The multichannel sound with pauses shrunk.
    """

    chans = mcs_data.shape[0]
    out_data = np.zeros_like(mcs_data, dtype=np.float32)
    for i in range(0, chans):
        k = 0
        zero_count = 0
        for j in range(0, mcs_data.shape[1]):
            if mask[i][j] == 0:
                zero_count += 1
                if zero_count < min_pause[i]:
                    out_data[i][k] = mcs_data[i][j]
                    k += 1
            else:
                zero_count = 0
                out_data[i][k] = mcs_data[i][j]
                k += 1
    return out_data


def pause_measure(mask: np.ndarray[int]) -> dict:
    """
    Measures pauses in multichannel sound.

    Parameters:
        mask (np.ndarray): A mask indicating the pauses in the multichannel
        sound.

    Returns:
        list: A list of lists containing pairs of (index, length) of pauses for
        each channel.  Length is in samples."""

    chans = mask.shape[0]
    pause_list = []
    out_list = []
    index = 0
    for i in range(0, chans):
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


def pause_set(
    mcs_data: np.ndarray, pause_map: list, pause_sz: list[int]
) -> np.ndarray:
    """
    Set pauses lengths in multichannel sound to selected values.

    Parameters:
        mcs_data (np.ndarray): The multichannel sound data.
        pause_map (list): A list of dictionaries containing pairs of (index,
        length) of pauses for each channel.
        pause_sz (list[int]): A list of pause sizes for each channel.

    Returns:
        np.ndarray: The multichannel sound with pauses shrunk.
    """
    chans = mcs_data.shape[0]
    out_list = []
    for i in range(0, chans):
        prev_index = 0
        local_list = []
        for p in pause_map[i]:
            index = p[0] + p[1]
            delta = index - prev_index
            if delta > 0:
                local_list.append(mcs_data[i][prev_index : prev_index + delta])
                stub = np.zeros(pause_sz[i])
                local_list.append(stub)
                prev_index = index
        out_list.append(local_list)
        a = []
        for L in out_list:
            a.append(np.concatenate(L).copy())
        max_len = -1
        for b in a:
            if len(b) > max_len:
                max_len = len(b)
        c = []
        for e in a:
            e = np.concatenate([e, np.zeros(max_len - len(e))]).copy()
            c.append(e)
    res = np.stack(c, axis=0).copy()
    return res


def split(mcs_data: np.ndarray, channels_count: int) -> np.ndarray:
    """
    Splits a multichannel signal (containing single channel) into multiple
    identical channels.

    Args:
        mcs_data (np.ndarray): The multichannel signal data.
        channels_count (int): The number of channels to split the signal into.

    Returns:
        np.ndarray: The split multichannel signal, with each channel identical.
    """

    out_data = np.zeros((channels_count, mcs_data.shape[1]), dtype=np.float32)
    for i in range(0, channels_count):
        out_data[i] = mcs_data.copy()

    return out_data


def merge(mcs_data: np.ndarray) -> np.ndarray:
    """
    Mixes channels of a multichannel sound into a single channel.

    Args:
        mcs_data (np.ndarray): The multichannel sound data.

    Returns:
        np.ndarray: The merged sound data, containing a single channel.
    """

    out_data = np.zeros(mcs_data.shape[1], dtype=np.float32)
    channels_count = mcs_data.shape[0]
    for i in range(0, channels_count):
        out_data += mcs_data[i]

    return out_data


def sum(mcs_data1: np.ndarray, mcs_data2: np.ndarray) -> np.ndarray:
    """
    Sums two multichannel sound signals.

    Parameters:
        mcs_data1 (np.ndarray): The first multichannel sound signal.
        mcs_data2 (np.ndarray): The second multichannel sound signal.

    Returns:
        np.ndarray: The sum of mcs_data1 and mcs_data2 signals.
    """

    out_data = mcs_data1 + mcs_data2

    return out_data.copy()


def side_by_side(mcs_data1: np.ndarray, mcs_data2: np.ndarray) -> np.ndarray:
    """
    Concatenates two multichannel sound signals side by side.

    Parameters:
        mcs_data1 (np.ndarray): The first multichannel sound signal.
        mcs_data2 (np.ndarray): The second multichannel sound signal.

    Returns:
        np.ndarray: The concatenated sound signal containing channels of both
        MCS.
    """

    out_data = np.zeros(
        (mcs_data1.shape[0] + mcs_data2.shape[0], mcs_data1.shape[1]),
        dtype=np.float32,
    )
    out_data[0 : mcs_data1.shape[0], :] = mcs_data1
    out_data[mcs_data1.shape[0] :, :] = mcs_data2
    return out_data.copy()


#  Chaining class


class WaChain:
    """
    Class provides support of chain operations with multichannel sound
    data.
    """

    def __init__(self, mcs_data: np.ndarray = None, fs: int = -1):
        """
        Initializes a new instance of the WaChain class.

        Args:
            mcs_data (np.ndarray, optional): The multichannel sound data.
            Defaults to None.
            fs (int, optional): The sample rate of the sound data. Defaults
            to -1.

        Returns:
            None
        """

        self.data = mcs_data  # Multichannel sound data field
        self.path = ""  # Path to the sound file, from which the data was read.
        self.sample_rate = fs  # Sampling frequency, Hz.
        self.chains=[]  # List of chains.

    def copy(self) -> "WaChain":
        """
        Creates a deep copy of the WaChain instance.

        Returns:
            WaChain: A deep copy of the WaChain instance.
        """
        return copy.deepcopy(self)

    def put(self, mcs_data: np.ndarray, fs: int = -1) -> "WaChain":
        """
        Updates the multichannel sound data and sample rate of the WaChain
        instance.

        Args:
            mcs_data (np.ndarray): source of multichannel sound data.
            fs (int, optional): The new sample rate. Defaults to -1.

        Returns:
            WaChain: The updated WaChain instance.
        """

        self.data = mcs_data.copy()
        self.sample_rate = fs
        return self

    def get(self):
        """
        Returns the multichannel sound data stored in the WaChain instance.

        Returns:
            np.ndarray: The multichannel sound data.
        """
        return self.data

    def gen(
        self,
        f_list: list[int],
        t: float,
        fs: int = DEF_FS,
        mode="sine",
        seed: int = -1,
    ) -> "WaChain":
        """
        Generates a multichannel sound using the given frequency list,
        duration, and sampling rate.

        Args:
            f_list (list[int]): The list of frequencies to generate the sound
            for each channel.
            t (float): The duration of the sound in seconds.
            fs (int, optional): The sampling rate of the sound data. Defaults
            to def_fs.
            mode (str, optional): The mode of sound generation. Can be
            'sine' or 'speech'. Defaults to 'sine'.
            seed (int, optional): The seed for random number generation.
            Defaults to -1.

        Returns:
            WaChain: The updated WaChain instance with the generated
            multichannel sound, allowing for method chaining.
        """

        self.data = generate(f_list, t, fs, mode, seed)
        self.sample_rate = fs
        return self

    def rd(self, path: str) -> "WaChain":
        """
        Reads data from a file at the specified path and updates the sample
        rate and data attributes.

        Args:
            path (str): Path to the file containing the data.

        Returns:
            sWaChain: The updated WaChain instance itself, allowing for method
            chaining.
        """

        self.sample_rate, self.data = read(path)
        return self

    def wr(self, path: str) -> "WaChain":
        """
        Writes the audio data to a file at the specified path.

        Args:
            path (str): The path to write the audio data to.

        Returns:
            WaChain: The current instance of WaChain, allowing for method
            chaining.
        """

        write(path, self.data, self.sample_rate)
        return self

    def amp(self, amplitude_list: list[float]) -> "WaChain":
        """
        Amplifies the audio data based on a custom amplitude control.

        Args:
            amplitude_list (list[float]): A list of amplitudes to apply to each
            corresponding chunk of audio data.

        Returns:
            WaChain: The current instance of WaChain, allowing for method
            chaining.
        """

        self.data = amplitude_ctrl(self.data, amplitude_list)
        return self

    def dly(self, delay_list: list[int]) -> "WaChain":
        """
        Delays the audio data based on a custom delay control.

        Args:
            delay_list (list[int]): A list of delays to apply to each
            corresponding chunk of audio data.

        Returns:
            WaChain: The current instance of WaChain, allowing for method
            chaining.
        """

        self.data = delay_ctrl(self.data, delay_list)
        return self

    def ns(self, noise_level_list, seed=-1) -> "WaChain":
        """
        Adds custom noise to the audio data.

        Args:
            noise_level_list (list[float]): A list of noise levels to apply to
            each corresponding chunk of audio data.
            seed (int): An optional random seed for reproducibility. Defaults
            to -1.

        Returns:
            WaChain: The current instance of WaChain, allowing for method
            chaining.
        """

        self.data = noise_ctrl(self.data, noise_level_list, seed)
        return self

    def echo(
        self,
        delay_us_list: list[int],
        amplitude_list: list[float],
        sampling_rate: int = DEF_FS,
    ) -> "WaChain":
        """
        Adds an echo effect to the audio data.

        Args:
            delay_us_list (list[int]): A list of delays in microseconds for
            each corresponding chunk of audio data.
            amplitude_list (list[float]): A list of amplitudes to apply to each
            corresponding chunk of echo data.
            sampling_rate (int): The sampling rate at which to add the echo
            effect.  Defaults to `def_fs`.

        Returns:
            WaChain: The current instance of WaChain, allowing for method
            chaining.
        """

        self.data = echo_ctrl(
            self.data, delay_us_list, amplitude_list, sampling_rate
        )
        return self

    def rms(self, last_index: int = -1, decimals: int = -1) -> list[float]:
        """
        Calculates the root mean square (RMS) of the audio data.

        Args:
            last_index (int): The index up to which to calculate the RMS.
            Defaults to -1, meaning all data.
            decimals (int): The number of decimal places to round the
            result to.
            Defaults to -1, meaning no rounding.

        Returns:
            list[float]: A list containing the RMS values for each
            corresponding chunk of audio data.
        """

        return rms(self.data, last_index, decimals)

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
            res["channels_count"] = self.data.shape[0]
            res["length_s"] = length
        return res

    def sum(self, mcs_data: np.ndarray) -> "WaChain":
        """
        Sums two multichannel sound signals side by side.

        Args:
            mcs_data (np.ndarray): The second multichannel sound signal.

        Returns:
            WaChain: The updated WaChain instance with the summed multichannel
            sound, allowing for method chaining.
        """

        self.data = sum(self.data, mcs_data)
        return self

    def mrg(self) -> "WaChain":
        """
        Merges all channels to single and returns mono MCS.

        Returns:
            WaChain: The updated WaChain instance with the merged multichannel
            sound, allowing for method chaining.
        """

        self.data = merge(self.data)
        return self

    def splt(self, channels_count: int) -> "WaChain":
        """
        Splits a multichannel signal (containing single channel) into multiple
        identical channels.

        Args:
            channels_count (int): The number of channels to split the signal
            into.

        Returns:
            WaChain: The updated WaChain instance with the split multichannel
            sound, allowing for method chaining.
        """

        self.data = split(self.data, channels_count)
        return self

    def sbs(self, mcs_data: np.ndarray) -> "WaChain":
        """
        Concatenates two multichannel sound signals side by side.

        Args:
            mcs_data (np.ndarray): The second multichannel sound signal.

        Returns:
            WaChain: The updated WaChain instance with the concatenated
            multichannel sound, allowing for method chaining.
        """

        self.data = side_by_side(self.data, mcs_data)
        return self

    def pdt(self, relative_level: list[float]) -> "WaChain":
        """
        Detects pauses in a multichannel sound based on a custom
        relative level value.

        Args:
            relative_level (list[float]): A list of relative levels for each
            corresponding channel, signal below this level will be marked as
            pause.

        Returns:
            WaChain: The updated WaChain instance with the pause detection
            result, allowing for method chaining.
        """
        self.data = pause_detect(self.data, relative_level)
        return self

    def achn(self, list_of_chains: list[str]) -> "WaChain":
        """
        Add chain to list of chains.

        Args:
            list_of_chains (list[str]): A list of chains to add.

        Returns:
            WaChain: The updated WaChain instance with added chains.
            result, allowing for method chaining.
        """

        for c in list_of_chains:
            self.chains.append( c.strip())
        return self

    def evl(self) -> "WaChain":
        """
        Add chain to list of chains.

        Args:
            list_of_chains (list[str]): A list of chains to add.

        Returns:
            WaChain: The updated WaChain instance with added chains.
            result, allowing for method chaining.
        """
        str(eval(cmd_prefix + c.strip())) # It is need for chain commands.

        for s in list_of_chains:
            self.chains.append(s)
        return self


# CLI interface functions
error_mark = "Error: "
success_mark = "Done."

prog_name = os.path.basename(__file__).split(".")[0]

application_info = f"{prog_name} application provides functions for \
multichannel WAV audio data augmentation."


def check_amp_list(ls: list[str]) -> None:
    """
    Checks if all elements in the given amplitudes list are valid numbers.

    Args:
        ls (list): The list of elements to check.

    Returns:
        None

    Raises:
        ValueError: If the list contains a non-number element.
        SystemExit: Exits the program with a status code of 3 if a non-number
        element is found.
    """
    for n in ls:
        try:
            float(n)
        except ValueError:
            print(
                f"{error_mark}Amplitude list"
                f" contains non number element: <{n}>."
            )
            sys.exit(3)


def check_delay_list(ls: list[str]) -> None:
    """
    Checks if all elements in the given delays list are valid integers.

    Args:
        ls (list): The list of elements to check.

    Returns:
        None

    Raises:
        ValueError: If the list contains a non-integer element.
        SystemExit: Exits the program with a status code of 1 if a non-integer
        element is found.
    """
    for n in ls:
        try:
            int(n)
        except ValueError:
            print(
                f"{error_mark}Delays list"
                f" contains non integer element: <{n}>."
            )
            sys.exit(1)


def print_help_and_info():
    """Function prints info about application"""

    print(application_info)
    sys.exit(0)


def chain_hdr(args):
    """
    Processes the chain code from the given arguments and executes the
    corresponding WaChain commands.

    Args:
        args: The arguments containing the chain code to be executed.

    Returns:
        None

    Raises:
        SystemExit: Exits the program with a status code of 0 after
        successful execution.
    """
    if args.chain_code is None:
        return
    c = args.chain_code.strip()
    print("chain:", c)
    w = WaChain()
    cmd_prefix = "w."
    str(eval(cmd_prefix + c.strip())) # It is need for chain commands.
    print(success_mark)
    w.info()
    sys.exit(0)


def input_path_hdr(args):
    """Function checks presence of input file"""
    if args.in_path is None:
        print_help_and_info()
    if not os.path.exists(args.in_path) or not os.path.isfile(args.in_path):
        print(f"{error_mark}Input file <{args.in_path}> not found.")
        sys.exit(1)


def is_file_creatable(fullpath: str) -> bool:
    """
    Checks if a file can be created at the given full path.

    Args:
        fullpath (str): The full path where the file is to be created.

    Returns:
        bool: True if the file can be created, False otherwise.

    Raises:
        Exception: If the file cannot be created.
        SystemExit: If the path does not exist.
    """

    # Split the path
    path, _ = os.path.split(fullpath)
    isdir = os.path.isdir(path)
    if isdir:
        try:
            Path(fullpath).touch(mode=0o777, exist_ok=True)
        except Exception:
            print(f"{error_mark}Can't create file <{fullpath}>.")
            raise
    else:
        print(f"{error_mark}Path <{path}> is not exists.")
        sys.exit(1)

    return True


def output_path_hdr(args):
    """Function checks of output file name and path."""

    if not is_file_creatable(args.out_path):
        print(f"{error_mark}Can't create file <{args.out_path}>.")
        sys.exit(1)


def file_info_hdr(args):
    """Function prints info about input audio file."""

    print()
    if args.info:
        for key, value in file_info(args.path).items():
            print(f"{key}: {value}")
        sys.exit(0)


def amplitude_hdr(args):
    """Function makes CLI amplitude augmentation."""

    if args.amplitude_list is None:
        return

    amplitude_list = args.amplitude_list.split(",")
    check_amp_list(amplitude_list)

    float_list = [float(i) for i in amplitude_list]
    print(f"amplitudes: {float_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(float_list):
        print(
            f"{error_mark}Amplitude list length <{len(float_list)}>"
            " does not match number of channels. It should have"
            f" <{info['channels_count']}> elements."
        )
        sys.exit(2)
    _, mcs_data = read(args.in_path)
    res_data = amplitude_ctrl(mcs_data, float_list)
    write(args.out_path, res_data, info["sample_rate"])
    print(success_mark)
    sys.exit(0)


def noise_hdr(args):
    """Function makes CLI noise augmentation."""

    if args.noise_list is None:
        return

    noise_list = args.noise_list.split(",")
    check_amp_list(noise_list)

    float_list = [float(i) for i in noise_list]
    print(f"noise levels: {float_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(float_list):
        print(
            f"{error_mark}Noise list length <{len(float_list)}>"
            " does not match number of channels. It should have"
            f" <{info['channels_count']}> elements."
        )
        sys.exit(2)
    _, mcs_data = read(args.in_path)
    res_data = noise_ctrl(mcs_data, float_list)
    write(args.out_path, res_data, info["sample_rate"])
    print(success_mark)
    sys.exit(0)


def echo_hdr(args):
    """Function makes CLI echo augmentation."""

    if args.echo_list is None:
        return

    lists = args.echo_list.split("/")
    if len(lists) != 2:
        print(
            f"{error_mark}Can't distinguish delay and amplitude"
            f"lists <{args.echo_list}>."
        )
        sys.exit(1)

    delay_list = lists[0].split(",")
    amplitude_list = lists[1].split(",")
    if len(amplitude_list) != len(delay_list):
        print(
            f"{error_mark}Can't delay and amplitude lists lengths"
            f"differ <{args.echo_list}>."
        )
        sys.exit(2)

    check_delay_list(delay_list)
    check_amp_list(amplitude_list)

    int_list = [int(i) for i in delay_list]
    print(f"delays: {int_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(int_list):
        print(
            f"{error_mark}Delay list length <{len(int_list)}>"
            " does not match number of channels. It should have"
            f" <{info['channels_count']}> elements."
        )
        sys.exit(2)

    float_list = [float(i) for i in amplitude_list]
    print(f"amplitudes: {float_list}")

    _, mcs_data = read(args.in_path)
    res_data = echo_ctrl(mcs_data, int_list, float_list)

    write(args.out_path, res_data, info["sample_rate"])
    print(success_mark)
    sys.exit(0)


def delay_hdr(args):
    """Function makes CLIdelay augmentation."""

    if args.delay_list is None:
        return

    delay_list = args.delay_list.split(",")
    check_delay_list(delay_list)

    int_list = [int(i) for i in delay_list]
    print(f"delays: {int_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(int_list):
        print(
            f"{error_mark}Delays list length <{len(int_list)}>"
            " does not match number of channels. It should have"
            f" <{info['channels_count']}> elements."
        )
        sys.exit(2)
    _, mcs_data = read(args.in_path)
    res_data = delay_ctrl(mcs_data, int_list)
    write(args.out_path, res_data, info["sample_rate"])
    print(success_mark)
    sys.exit(0)


def parse_args():
    """CLI options parsing."""

    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="WAV audio files augmentation utility.",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("-i", dest="in_path", help="Input audio" " file path.")
    parser.add_argument("-o", dest="out_path", help="Output audio file path.")
    parser.add_argument(
        "--info",
        dest="info",
        action="store_true",
        help="Print info about input audio file.",
    )
    parser.add_argument(
        "--amp",
        "-a",
        dest="amplitude_list",
        help="Change amplitude (volume)"
        " of channels in audio file. Provide coefficients for"
        ' every channel, example:\n\t -a "0.1, 0.2, 0.3, -1"',
    )
    parser.add_argument(
        "--echo",
        "-e",
        dest="echo_list",
        help="Add echo to channels in audio file."
        " of channels in audio file. Provide coefficients"
        "  and delays (in microseconds) of "
        " reflected signal for every channel, example:\n\t"
        '      -e "0.1, 0.2, 0.3, -1 / 100, 200, 0, 300"',
    )
    parser.add_argument(
        "--dly",
        "-d",
        dest="delay_list",
        type=str,
        help="Add time delays"
        " to channels in audio file. Provide delay for"
        ' every channel in microseconds, example:\n\t \
                            -d "100, 200, 300, 0"',
    )
    parser.add_argument(
        "--ns",
        "-n",
        dest="noise_list",
        help="Add normal noise"
        " to channels in audio file. Provide coefficients for"
        ' every channel, example:\n\t -n "0.1, 0.2, 0.3, -1"',
    )
    parser.add_argument(
        "--chain",
        "-c",
        dest="chain_code",
        type=str,
        help="Execute chain of transformations."
        " example:\n\t"
        '-c "gen([100,250,100], 3, 44100).amp([0.1, 0.2, 0.3])'
        '.wr("./sines.wav")"',
    )

    return parser.parse_args()


def main():
    """CLI arguments parsing."""
    if sys.version_info[0:2] != (3, 11):
        raise Exception('Requires python 3.11')
    args = parse_args()
    chain_hdr(args)
    input_path_hdr(args)
    file_info_hdr(args)
    output_path_hdr(args)
    amplitude_hdr(args)
    noise_hdr(args)
    delay_hdr(args)
    echo_hdr(args)


if __name__ == "__main__":
    main()
