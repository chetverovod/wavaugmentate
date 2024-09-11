#!/usr/bin/env python3

"""
This module defines multichannel audio flies augmentation class Mcs.
"""

import copy
import random
import sys
from typing import List, Tuple

import numpy as np
from scipy.io import wavfile

ERROR_MARK = "Error: "
SUCCESS_MARK = "Done."

# Default sampling frequency, Hz.
DEF_FS = 44100

# Default signal durance in seconds.
DEF_SIGNAL_LEN = 5

random_noise_gen = np.random.default_rng()


def delay_syntez(
    delay_us_list: List[int],
    delay_deviation_list: List[int] = None,
    seed: int = -1,
) -> List[int]:
    """Make delays list"""

    d_list = delay_us_list
    if delay_deviation_list is not None:
        d_list = []
        for delay, dev in zip(delay_us_list, delay_deviation_list):
            if dev > 0:
                left = delay - dev
                if left < 0:
                    print(
                        f"{ERROR_MARK}"
                        f"deviation value {dev} can give negative delay."
                    )
                    sys.exit(1)
                right = delay + dev
                if seed != -1:
                    local_ng = np.random.default_rng(seed=seed)
                    d_list.append(local_ng.integers(left, right))
                else:
                    d_list.append(random_noise_gen.integers(left, right))
    return d_list


def pause_measure(mask: np.ndarray[int]) -> dict:
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


class Mcs:
    """
    Class provides support of  multichannel sound
    data.
    """

    def __init__(
        self, np_data: np.ndarray = None, samp_rt: int = -1, seed: int = -1
    ):
        """
        Initializes a new instance of the Mcs class.

        Args:
            mcs_data (np.ndarray, optional): The multichannel sound data.
            Defaults to None.
            fs (int, optional): The sample rate of the sound data. Defaults
            to -1.
            seed (int): Value for seeding random generator. Defaults to -1.

        Returns:
            None
        """

        if np_data is None:
            self.data = None  # np.ndarray: Multichannel sound data field.
        else:
            self.data = (
                np_data.copy()
            )  # np.ndarray: Multichannel sound data field.
        self.path = ""  # Path to the sound file, from which the data was read.
        self.sample_rate = samp_rt  # Sampling frequency, Hz.
        self.chains = []  # List of chains.
        self.seed = seed  # Flag for seeding random generator.

    def copy(self) -> "Mcs":
        """Deep Copy of the Mcs object."""

        return copy.deepcopy(self)

    def __channel_rms(
        self, chan_index: int, last_index: int, decimals: int
    ) -> float:
        """
        Calculate the root mean square (RMS) of a single channel signal.

        Args
            signal_of_channel (array): Input signal of a single channel.
            decimals (int): Number of decimal places to round the RMS value.

        Returns:
            float: The RMS value of the input signal.
        """
        # Calculate the mean of the squares of the signal values
        mean_square = 0
        if chan_index > -1:
            mean_square = np.mean(self.data[chan_index, 0:last_index] ** 2)
        else:
            mean_square = np.mean(self.data[0:last_index] ** 2)

        # Calculate the square root of the mean of the squares
        single_chan_rms = np.sqrt(mean_square)

        # Round the result to the specified number of decimal places
        if decimals > 0:
            single_chan_rms = round(single_chan_rms, decimals)

        # Return the result
        return single_chan_rms

    def rms(self, last_index: int = -1, decimals: int = -1):
        """
        Calculate the root mean square (RMS) of a multichannel sound.

        Args:
            last_index (int): The last index to consider when calculating the RMS.
            If -1, consider the entire array. Defaults to -1.
            decimals (int): Number of decimal places to round the RMS value.
            If -1, do not round. Defaults to -1.

        Returns:
            list: A list of RMS values for each channel in the multichannel sound.

        """

        res = []
        shape_len = len(self.data.shape)
        if shape_len > 1:
            for i in range(0, self.data.shape[0]):
                chan_rms = self.__channel_rms(i, last_index, decimals)
                res.append(chan_rms)
        else:
            chan_rms = self.__channel_rms(-1, last_index, decimals)
            res.append(chan_rms)
        return res

    def shape(self) -> Tuple:
        """
        Returns the shape of the multichannel sound data.

        Returns:
            Tuple: A tuple containing the shape of the multichannel sound data.
        """
        return self.data.shape

    def generate(
        self,
        frequency_list: List[int],
        duration: float = DEF_SIGNAL_LEN,
        samp_rt: int = -1,
        mode="sine",
    ) -> "Mcs":
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
        fs (int): The sample rate of the sound. Defaults to -1.
        mode (str): The mode of sound generation. Can be 'sine' or 'speech'.
        Defaults to 'sine'.

        Returns:
        self (Mcs):  representing the generated multichannel sound.
        """

        if samp_rt > 0:
            self.sample_rate = samp_rt
        self.data = None
        samples = np.arange(duration * self.sample_rate) / self.sample_rate
        channels = []
        if mode == "sine":
            for freq in frequency_list:
                signal = np.sin(2 * np.pi * freq * samples)
                signal = np.float32(signal)
                channels.append(signal)
            self.data = np.array(channels).copy()

        if mode == "speech":
            if self.seed != -1:
                random.seed(self.seed)
            for freq in frequency_list:
                if freq > 300 or freq < 60:
                    print(
                        ERROR_MARK + "Use basic tone from interval 600..300 Hz"
                    )
                    sys.exit(1)

                # Formants:
                fbt = random.randint(freq, 300)  # 60–300 Гц
                freq_list = [fbt]
                freq_list.append(random.randint(2 * fbt, 850))  # 150–850 Гц
                freq_list.append(random.randint(3 * fbt, 2500))  # 500–2500 Гц
                freq_list.append(random.randint(4 * fbt, 3500))  # 1500–3500 Гц
                freq_list.append(random.randint(5 * fbt, 4500))  # 2500–4500 Гц
                signal = 0
                amp = 1
                for frm in freq_list:
                    signal += amp * np.sin(2 * np.pi * frm * samples)
                    amp -= 0.1
                peak_amplitude = np.max(np.abs(signal))
                signal = signal / peak_amplitude
                signal = np.float32(signal)
                channels.append(signal)
                self.data = np.array(channels).copy()
        return self

    def write(self, path: str) -> "Mcs":
        """
            Writes the given multichannel sound data to a WAV file at the specified
            path.

        Args:
            path (str): The path to the WAV file.
            mcs_data (np.ndarray): The multichannel sound data to write. The shape
            of the array should be (num_channels, num_samples).

        Returns:
        self (Mcs):  representing saved multichannel sound.
        """

        buf = self.data.T.copy()
        wavfile.write(path, self.sample_rate, buf)
        return self

    def write_by_channel(self, path: str) -> "Mcs":
        """
        Writes each channel of the multichannel sound data to a separate WAV
        files, 1 for each channel.

        File name will be modified to include the channel number. If path contains
        ./outputwav/sound_augmented.wav the output file names will be
        ./outputwav/sound_augmented_1.wav
        ./outputwav/sound_augmented_2.wav and so on.

        Args:
            path (str): The path to the WAV file. The filename will be modified
            to include the channel number.

        Returns:
            self (Mcs): The Mcs instance itself, allowing for method chaining.
        """

        trimmed_path = path.split(".wav")
        for i in range(self.channels_count()):
            buf = self.data[i, :].T.copy()
            file_name = trimmed_path[0] + f"_{i + 1}.wav"
            print(f"Writing {file_name}...")
            wavfile.write(file_name, self.sample_rate, buf)
        return self

    def read(self, path: str) -> "Mcs":
        """
        Reads a multichannel sound from a WAV file.

        Args:
            path (str): The path to the WAV file.

        Returns:
            tuple[int, np.ndarray]: A tuple containing the sample rate and the
            multichannel sound data.
        """

        self.sample_rate, buf = wavfile.read(path)
        if len(buf.shape) != 2:
            buf = np.expand_dims(buf, axis=1)
        self.path = path
        self.data = buf.T.copy()
        return self

    # Audio augmentation functions

    def amplitude_ctrl(
        self,
        amplitude_list: List[float],
        amplitude_deviation_list: List[float] = None,
    ) -> "Mcs":
        """
        Apply amplitude control to a multichannel sound. If
        amplitude_deviation_list is defined, you can get different
        versions of tha same mcs data.

        Args:
            amplitude_list (List[float]): The list of amplitude coefficients to
            apply to each channel.
            amplitude_deviation_list (List[float]): If exists, sets amplitude values
            random with uniform distribution in range
            [amplitude - deviation, amplitude + deviation)].

        Returns:
            self (Mcs): The amplitude-controlled multichannel sound.
        """
        if self.channels_count() != len(amplitude_list):
            print(
                ERROR_MARK
                + "Amplitude list length does not match number of channels."
            )
            sys.exit(1)

        amp_list = amplitude_list
        if amplitude_deviation_list is not None:
            if self.channels_count() != len(amplitude_deviation_list):
                print(
                    ERROR_MARK
                    + "Amplitude deviation list length does not match number of channels."
                )
                sys.exit(1)

            amp_list = []
            for amplitude, dev in zip(
                amplitude_list, amplitude_deviation_list
            ):
                if dev > 0:
                    left = amplitude - dev
                    right = amplitude + dev
                    if self.seed != -1:
                        local_ng = np.random.default_rng(seed=self.seed)
                        amp_list.append(local_ng.uniform(left, right))
                    else:
                        amp_list.append(random_noise_gen.uniform(left, right))

        channels = []
        for signal, ampl in zip(self.data, amp_list):
            channels.append(signal * ampl)

        self.data = np.array(channels).copy()
        return self

    def delay_ctrl(
        self,
        delay_us_list: List[int],
        delay_deviation_list: List[int] = None,
    ) -> "Mcs":
        """
            Add delays of channels of multichannel sound. Output data become longer.
            Values of delay will be converted to count of samples.

        Args:
            delay_us_list (List[int]): The list of delay values in microseconds to
            apply to each channel. Each value should be a positive integer.
            sound data.
            delay_deviation_list (List[int]): If exists, the list of delay
            deviations makes delays uniformly distributed.

        Returns:
            self (Mcs): The delayed multichannel sound.

        """

        if self.channels_count() != len(delay_us_list):
            print(
                ERROR_MARK
                + "Delay list length does not match number of channels."
            )
            sys.exit(1)

        if delay_deviation_list is not None:
            if self.channels_count() != len(delay_deviation_list):
                print(
                    ERROR_MARK
                    + "Delay deviation list length does not match number of channels."
                )
                sys.exit(1)

        d_list = delay_syntez(delay_us_list, delay_deviation_list, self.seed)
        channels = []
        # In samples.
        max_samples_delay = int(max(d_list) * 1.0e-6 * self.sample_rate)

        for signal, delay in zip(self.data, d_list):
            samples_delay = int(
                delay * 1.0e-6 * self.sample_rate
            )  # In samples.
            res = np.zeros(samples_delay)
            res = np.append(res, signal)
            if samples_delay < max_samples_delay:
                res = np.append(
                    res, np.zeros(max_samples_delay - samples_delay)
                )
            channels.append(res)
        self.data = np.array(channels).copy()
        return self

    def echo_ctrl(
        self,
        delay_us_list: List[int],
        amplitude_list: List[float],
        delay_deviation_list: List[int] = None,
        amplitude_deviation_list: List[float] = None,
    ) -> "Mcs":
        """
        Add echo to multichannel sound. The output data become longer. To each
        channel will be added it's copy with corresponding delay delay and
        amplitude. It looks like acoustic wave was reflected from the hard wall.

        Args:
            delay_us_list (List[int]): The list of delay values in microseconds to
                apply to each channel. Each value should be a positive integer.
            amplitude_list (List[float]): The list of amplitude coefficients to
                apply to each channel.
            delay_deviation_list (List[int]): If exists gives random deviation of
            reflection delay.
            amplitude_deviation_list (List[float]): If exists gives random
            deviation of reflection amplitude.
            seed (int): If exists seeds random generator.

        Returns:
            self (Mcs): The echoed multichannel sound.
        """
        amplitude_change = self.copy()
        amplitude_change.amplitude_ctrl(amplitude_list, amplitude_deviation_list)
        delay_change = amplitude_change.copy()
        delay_change.delay_ctrl(delay_us_list, delay_deviation_list)
        channels = []
        for single_channel in self.data:
            zeros_len = delay_change.data.shape[1] - single_channel.data.shape[0]
            channels.append(np.append(single_channel, np.zeros(zeros_len)))
        self.data = np.array(channels).copy() + delay_change.data.copy()

        return self

    def noise_ctrl(
        self,
        noise_level_list: List[float],
    ) -> "Mcs":
        """
        Apply noise to a multichannel sound.

        Args:
            noise_level_list (List[float]): The list of noise levels to apply to
            each channel.
            seed (int): The seed for random number generation. Defaults to -1.

        Returns:
            self (Mcs): The noise-controlled multichannel sound.
        """

        channels = []
        for signal, level in zip(self.data, noise_level_list):
            if self.seed != -1:
                local_ng = np.random.default_rng(seed=self.seed)
                n_noise = local_ng.standard_normal(
                    self.data.shape[1],
                )
            else:
                n_noise = random_noise_gen.standard_normal(self.data.shape[1])
            noise = n_noise
            res = signal + level * noise
            channels.append(res)
        self.data = np.array(channels).copy()
        return self

    def pause_detect(self, relative_level: List[float]) -> np.ndarray[int]:
        """
            Detects pauses in a multichannel sound.

            Args:
            mcs_data (np.ndarray): The multichannel sound data.
            relative_level (List[float]): The list of relative levels for each
            channel, signal below this level will be marked as pause.

        Returns:
            np.ndarray: The mask indicating the pauses in the multichannel sound.
            The mask has the same shape as the input sound. It contains zeros and
            ones 0 - pause, 1 - not a pause.
        """

        rms_list = self.rms()
        modules_list = abs(self.data)
        mask = np.zeros(self.data.shape)

        for i in range(0, self.data.shape[0]):
            threshold = rms_list[i] * relative_level[i]
            mask[i] = np.clip(modules_list[i], a_min=threshold, a_max=1.1 * threshold)
            mask[i] -= threshold
            mask[i] /= 0.09 * threshold
            mask[i] = np.clip(mask[i], a_min=0, a_max=1).astype(int)
        return mask

    def pause_shrink(
        self, mask: np.ndarray[int], min_pause: List[int]
    ) -> "Mcs":
        """
        Shrink pauses in multichannel sound.

        Args:
            mask (np.ndarray): The mask indicating the pauses in the multichannel
            sound.
            min_pause (List[int]): The list of minimum pause lengths for
            each channel in samples.

        Returns:
            self (Mcs): The multichannel sound with pauses shrunk.
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
        self.data = out_data.copy()
        return self

    def pause_set(self, pause_map: list, pause_sz: List[int]) -> "Mcs":
        """
        Set pauses lengths in multichannel sound to selected values.

        Args:
            pause_map (list): A list of dictionaries containing pairs of (index,
            length) of pauses for each channel.
            pause_sz (List[int]): A list of pause sizes for each channel.

        Returns:
            self (Mcs): The multichannel sound with pauses shrunk.
        """

        out_list = []
        for i in range(0, self.data.shape[0]):
            prev_index = 0
            local_list = []
            for pause_info in pause_map[i]:
                index = pause_info[0] + pause_info[1]
                delta = index - prev_index
                if delta > 0:
                    local_list.append(
                        self.data[i][prev_index : prev_index + delta]
                    )
                    stub = np.zeros(pause_sz[i])
                    local_list.append(stub)
                    prev_index = index

            out_list.append(local_list)
            a_list = []
            for elem in out_list:
                a_list.append(np.concatenate(elem).copy())

            max_len = -1
            for elem in a_list:
                max_len = max(max_len, len(elem))

            channels_list = []
            for elem in a_list:
                elem = np.concatenate(
                    [elem, np.zeros(max_len - len(elem))]
                ).copy()
                channels_list.append(elem)
        self.data = np.stack(channels_list, axis=0).copy()
        return self

    def channels_count(self) -> int:
        """Returns the number of channels in the multichannel signal."""

        channels_count = 0
        if self.data is not None:
            if len(self.data.shape) > 1:
                channels_count = self.data.shape[0]
            else:
                channels_count = 1
        return channels_count

    def split(self, channels_count: int) -> "Mcs":
        """
        Splits a multichannel signal (containing single channel) into multiple
        identical channels.

        Args:
            channels_count (int): The number of channels to split the signal into.

        Returns:
            self (Mcs): The split multichannel signal, with each channel identical.
        """

        if self.channels_count() > 1:
            print(ERROR_MARK, "Can't split more than 1 channel signal.")
            sys.exit(1)

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
            out_data[i] = self.data.copy()
        self.data = out_data
        return self

    def merge(self) -> "Mcs":
        """
            Mixes channels of a multichannel sound into a single channel.

        Args:
            none

        Returns:
            self (Mcs): The merged sound data, containing a single channel.
        """

        out_data = np.zeros(self.data.shape[1], dtype=np.float32)
        channels_count = self.data.shape[0]
        for i in range(0, channels_count):
            out_data += self.data[i]
        self.data = out_data.copy()
        return self

    def sum(self, mcs_data2: "Mcs") -> "Mcs":
        """
        Sums two multichannel sound signals.

        Args:
            mcs_data2 (Mcs): The second multichannel sound signal.

        Returns:
            self (Mcs): The sum of self._data and mcs_data2 signals as Mcs.
        """

        out_data = self.data.copy() + mcs_data2.data.copy()
        self.data = out_data
        return self

    def side_by_side(self, mcs_data2: "Mcs") -> "Mcs":
        """
        Concatenates two multichannel sound signals side by side.

        Args:
            mcs_data2 (Mcs): The second multichannel sound signal.

        Returns:
            self (Mcs): The concatenated sound signal containing channels of both
            MCS.
        """

        out_data = np.zeros(
            (self.data.shape[0] + mcs_data2.data.shape[0], self.data.shape[1]),
            dtype=np.float32,
        )
        out_data[0 : self.data.shape[0], :] = self.data
        out_data[self.data.shape[0] :, :] = mcs_data2.data
        self.data = out_data.copy()
        return self

    def put(self, mcs: "Mcs") -> "Mcs":
        """
        Updates the multichannel sound data and sample rate of the Mcs
        instance.

        Args:
            mcs_data (Mcs): source of multichannel sound data.
            fs (int, optional): The new sample rate. Defaults to -1.

        Returns:
            Mcs: The updated Mcs instance.
        """

        self.data = mcs.data.copy()
        self.sample_rate = mcs.sample_rate
        self.path = mcs.path
        return self

    def get(self) -> np.ndarray:
        """
        Returns the multichannel sound data stored in the Mcs instance.

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

    def add_chain(self, list_of_chains: List[str]) -> "Mcs":
        """
        Add chain to list of chains.

        Args:
            list_of_chains (List[str]): A list of chains to add.

        Returns:
            self (Mcs): The updated Mcs instance with added chains.
            result, allowing for method chaining.
        """

        for chain in list_of_chains:
            self.chains.append(chain.strip())
        return self

    def eval(self) -> list["Mcs"]:
        """
        Evaluate list of chains.

        Args:
            none

        Returns:
            self (Mcs): The updated Mcs instance with added chains.
            result, allowing for method chaining.
        """

        res = []
        _ = self.copy()
        print("_sample_rate:", _.sample_rate)
        cmd_prefix = "_."
        for chain in self.chains:
            cmd_line = cmd_prefix + chain
            print("cmd_line:", cmd_line)
            res.append(eval(cmd_line))  # It is need for chain commands.
        return res

    def read_file_apply_chains(self, path: str) -> list["Mcs"]:
        """
        Reads data from a file at the specified path and updates the sample
        rate and data attributes and applies chains if they exist in object.

        Args:
            path (str): Path to the file containing the data.

        Returns:
            self (Mcs): The updated Mcs instance itself, allowing for method
            chaining.
        """

        self.read(path)
        res = self.eval()
        return res

    # Alias Method Names
    rd = read
    wr = write
    wrbc = write_by_channel
    amp = amplitude_ctrl
    dly = delay_ctrl
    echo = echo_ctrl
    ns = noise_ctrl
    mrg = merge
    splt = split
    sbs = side_by_side
    pdt = pause_detect
    achn = add_chain
    rdac = read_file_apply_chains
    gen = generate
    cpy = copy
