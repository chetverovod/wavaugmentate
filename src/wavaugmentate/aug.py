#!/usr/bin/env python3

"""
This module defines multichannel audio flies augmentation class Mcs.
"""

import copy
import sys
from typing import List

import numpy as np
import mcs as ms
from mcs import Mcs


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
                        f"{ms.ERROR_MARK}"
                        f"deviation value {dev} can give negative delay."
                    )
                    sys.exit(1)
                right = delay + dev
                if seed != -1:
                    local_ng = np.random.default_rng(seed=seed)
                    d_list.append(local_ng.integers(left, right))
                else:
                    d_list.append(ms.random_noise_gen.integers(left, right))
    return d_list


class Aug:
    """
    Class provides augmentation of  multichannel sound
    data (Mcs class objects).
    """

    def __init__(self, signal: "Mcs" = None) -> None:
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
        if signal is not None:
            self.signal = signal.copy()
        else:
            self.signal = Mcs()

        self.chains = []  # List of chains.

    def info(self) -> dict:
        """
        Returns a dictionary containing information about the object.
        
        The dictionary includes the information returned by the `info()` method
        of the `signal` attribute of the object, as well as a key-value pair
        where the key is 'chains' and the value is the `chains` attribute of the
        object.
        
        Returns:
            dict: A dictionary containing information about the object.
        """
        res = self.signal.info()
        res['chains'] = self.chains
        return res

    def set_seed(self, seed: int = -1):
        """Set seeding value."""

        self.signal.set_seed(seed)

    def put(self, signal: "Mcs") -> "Aug":
        """
        Updates the multichannel sound data and sample rate of the Mcs
        instance.

        Args:
            mcs_data (Mcs): source of multichannel sound data.
            fs (int, optional): The new sample rate. Defaults to -1.

        Returns:
            Mcs: The updated Mcs instance.
        """

        self.signal = signal.copy()
        return self

    def get(self) -> "Mcs":
        """
        Returns the multichannel sound data stored in the Mcs instance.

        Returns:
            np.ndarray: The multichannel sound data.
        """
        return self.signal

    def generate(
        self,
        frequency_list: List[int],
        duration: float = ms.DEF_SIGNAL_LEN,
        samp_rt: int = -1,
        mode="sine",
    ) -> "Aug":
        """
        Generates a multichannel sound based on the given frequency list, duration, 
        sample rate, and mode.

        Args:
            frequency_list (List[int]): A list of frequencies to generate sound for.
            duration (float): The duration of the sound in seconds. Defaults to ms.DEF_SIGNAL_LEN.
            samp_rt (int): The sample rate of the sound. Defaults to -1.
            mode (str): The mode of sound generation. Can be 'sine' or 'speech'. Defaults to 'sine'.

        Returns:
            Aug: The generated multichannel sound.
        """

        self.signal = self.signal.generate(
            frequency_list, duration, samp_rt, mode
        )
        return self

    # Audio augmentation functions
    def amplitude_ctrl(
        self,
        amplitude_list: List[float],
        amplitude_deviation_list: List[float] = None,
    ) -> "Aug":
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
            self (Aug): The amplitude-controlled multichannel sound.
        """

        obj = self.signal.copy()
        if obj.channels_count() != len(amplitude_list):
            print(
                ms.ERROR_MARK
                + "Amplitude list length does not match number of channels."
            )
            sys.exit(1)

        amp_list = amplitude_list
        if amplitude_deviation_list is not None:
            if obj.channels_count() != len(amplitude_deviation_list):
                print(
                    ms.ERROR_MARK
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
                    if obj.seed != -1:
                        local_ng = np.random.default_rng(seed=obj.seed)
                        amp_list.append(local_ng.uniform(left, right))
                    else:
                        amp_list.append(ms.random_noise_gen.uniform(left, right))

        channels = []
        for signal, ampl in zip(obj.data, amp_list):
            channels.append(signal * ampl)

        self.signal.data = np.array(channels).copy()
        return self

    def delay_ctrl(
        self,
        delay_us_list: List[int],
        delay_deviation_list: List[int] = None,
    ) -> "Aug":
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
            self (Aug): The delayed multichannel sound.

        """

        obj = self.signal.copy()
        if obj.channels_count() != len(delay_us_list):
            print(
                ms.ERROR_MARK
                + "Delay list length does not match number of channels."
            )
            sys.exit(1)

        if delay_deviation_list is not None:
            if obj.channels_count() != len(delay_deviation_list):
                print(
                    ms.ERROR_MARK
                    + "Delay deviation list length does not match number of channels."
                )
                sys.exit(1)

        d_list = delay_syntez(delay_us_list, delay_deviation_list, obj.seed)
        channels = []
        # In samples.
        max_samples_delay = int(max(d_list) * 1.0e-6 * obj.sample_rate)

        for signal, delay in zip(obj.data, d_list):
            samples_delay = int(
                delay * 1.0e-6 * obj.sample_rate
            )  # In samples.
            res = np.zeros(samples_delay)
            res = np.append(res, signal)
            if samples_delay < max_samples_delay:
                res = np.append(
                    res, np.zeros(max_samples_delay - samples_delay)
                )
            channels.append(res)
        self.signal.data = np.array(channels).copy()
        return self

    def echo_ctrl(
        self,
        delay_us_list: List[int],
        amplitude_list: List[float],
        delay_deviation_list: List[int] = None,
        amplitude_deviation_list: List[float] = None,
    ) -> "Aug":
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
            self (Aug): The echoed multichannel sound.
        """

        amplitude_change = self.copy()
        amplitude_change.amplitude_ctrl(amplitude_list, amplitude_deviation_list)
        delay_change = amplitude_change.copy()
        delay_change.delay_ctrl(delay_us_list, delay_deviation_list)
        channels = []
        for single_channel in self.signal.data:
            zeros_len = delay_change.signal.data.shape[1] - single_channel.data.shape[0]
            channels.append(np.append(single_channel, np.zeros(zeros_len)))
        self.signal.data = np.array(channels).copy() + delay_change.signal.data.copy()

        return self

    def noise_ctrl(
        self,
        noise_level_list: List[float],
    ) -> "Aug":
        """
        Apply noise to a multichannel sound.

        Args:
            noise_level_list (List[float]): The list of noise levels to apply to
            each channel.
            seed (int): The seed for random number generation. Defaults to -1.

        Returns:
            self (Aug): The noise-controlled multichannel sound.
        """

        obj = self.signal.copy()
        channels = []
        for signal, level in zip(obj.data, noise_level_list):
            if obj.seed != -1:
                local_ng = np.random.default_rng(seed=obj.seed)
                n_noise = local_ng.standard_normal(
                    obj.data.shape[1],
                )
            else:
                n_noise = ms.random_noise_gen.standard_normal(obj.data.shape[1])
            noise = n_noise
            res = signal + level * noise
            channels.append(res)
        self.signal.data = np.array(channels).copy()
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

        obj = self.signal.copy()
        mask = obj.pause_detect(relative_level)
        return mask

    def pause_shrink(
        self, mask: np.ndarray[int], min_pause: List[int]
    ) -> "Aug":
        """
        Shrink pauses in multichannel sound.

        Args:
            mask (np.ndarray): The mask indicating the pauses in the multichannel
            sound.
            min_pause (List[int]): The list of minimum pause lengths for
            each channel in samples.

        Returns:
            self (Aug): The multichannel sound with pauses shrunk.
        """

        obj = self.signal.copy()
        print('obj.shape =', obj.data.shape)
        chans = obj.data.shape[0]
        out_data = np.zeros_like(obj.data, dtype=np.float32)
        for i in range(0, chans):
            k = 0
            zero_count = 0
            for j in range(0, obj.data.shape[1]):
                if mask[i][j] == 0:
                    zero_count += 1
                    if zero_count < min_pause[i]:
                        out_data[i][k] = obj.data[i][j]
                        k += 1
                else:
                    zero_count = 0
                    out_data[i][k] = obj.data[i][j]
                    k += 1
        self.signal.data = out_data.copy()
        return self

    def pause_set(self, pause_map: list, pause_sz: List[int]) -> "Aug":
        """
        Set pauses lengths in multichannel sound to selected values.

        Args:
            pause_map (list): A list of dictionaries containing pairs of (index,
            length) of pauses for each channel.
            pause_sz (List[int]): A list of pause sizes for each channel.

        Returns:
            self (Aug): The multichannel sound with pauses shrunk.
        """

        obj = self.signal.copy()
        out_list = []
        for i in range(0, obj.data.shape[0]):
            prev_index = 0
            local_list = []
            for pause_info in pause_map[i]:
                index = pause_info[0] + pause_info[1]
                delta = index - prev_index
                if delta > 0:
                    local_list.append(
                        obj.data[i][prev_index : prev_index + delta]
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
        self.signal.data = np.stack(channels_list, axis=0).copy()
        return self

    def split(self, channels_count: int) -> "Aug":
        """
        Splits a multichannel signal (containing single channel) into multiple
        identical channels.

        Args:
            channels_count (int): The number of channels to split the signal into.

        Returns:
            self (Aug): The split multichannel signal, with each channel identical.
        """

        self.signal = self.signal.split(channels_count)
        return self

    def merge(self) -> "Aug":
        """
            Mixes channels of a multichannel sound into a single channel.

        Args:
            none

        Returns:
            self (Aug): The merged sound data, containing a single channel.
        """
        self.signal = self.signal.merge()
        return self

    def sum(self, mcs_data2: "Mcs") -> "Aug":
        """
        Sums two multichannel sound signals.

        Args:
            mcs_data2 (Mcs): The second multichannel sound signal.

        Returns:
            self (Aug): The sum of self._data and mcs_data2 signals as Mcs.
        """

        self.signal.sum(mcs_data2)
        return self

    def side_by_side(self, mcs_data2: "Mcs") -> "Aug":
        """
        Concatenates two multichannel sound signals side by side.

        Args:
            mcs_data2 (Mcs): The second multichannel sound signal.

        Returns:
            self (Aug): The concatenated sound signal containing channels of both
            MCS.
        """
        self.signal.side_by_side(mcs_data2)
        return self

    def add_chain(self, list_of_chains: List[str]) -> "Mcs":
        """
        Add chain to list of chains.

        Args:
            list_of_chains (List[str]): A list of chains to add.

        Returns:
            self (Aug): The updated Mcs instance with added chains.
            result, allowing for method chaining.
        """

        for chain in list_of_chains:
            self.chains.append(chain.strip())
        return self

    def copy(self) -> "Aug":
        """Deep Copy of the Mcs object."""

        return copy.deepcopy(self)

    def eval(self) -> list["Mcs"]:
        """
        Evaluate list of chains.

        Args:
            none

        Returns:
            self (Aug): The updated Mcs instance with added chains.
            result, allowing for method chaining.
        """

        res = []
        _ = self.copy()
        print("_sample_rate:", _.get().sample_rate)
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
            self (Aug): The updated Mcs instance itself, allowing for method
            chaining.
        """

        self.signal.read(path)
        res = self.eval()
        return res

    # Alias Method Names
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
