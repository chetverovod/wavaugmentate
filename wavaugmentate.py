#!/usr/bin/env python3

import numpy as np
import copy
import argparse
from scipy.io import wavfile
import os
from pathlib import Path
import random

"""
 Default sampling frequency, Hz.
"""
def_fs = 44100

random_noise_gen = np.random.default_rng()


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


def generate(frequency_list: list[100, 200, 300, 400], duration: float,
             sample_rate=def_fs, mode="sine", seed: int = -1):
    """
    Generate a multichannel sound based on the given frequency list, duration,
    sample rate, and mode. The mode can be 'sine' or 'speech'. In 'sine' mode,
    output multichannel sound will be a list of sine waves. In 'speech' mode,
    output will be a list of speech like signals. In this mode input
    frequencies list will be used as basic tone frequency for corresponding
    channel, it should be in interval 600..300.

    Parameters:
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

    if mode == 'speech':

        if seed != -1:
            random.seed(seed)
        for f in frequency_list:
            if f > 300 or f < 60: 
                print(error_mark + 'Use basic tone from interval 600..300 Hz')
                exit(1)
            # Formants:
            FBT = random.randint(f, 300)    # 60–300 Гц
            F1 = random.randint(2*FBT, 850)    # 150–850 Гц
            F2 = random.randint(3*FBT, 2500)   # 500–2500 Гц
            F3 = random.randint(4*FBT, 3500)  # 1500–3500 Гц
            F4 = random.randint(5*FBT, 4500)  # 2500–4500 Гц
            F = [FBT, F1, F2, F3, F4]
            signal = 0
            amp = 1
            for frm in F:
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

    return {"path": path, "channels_count": buf.shape[1],
            "sample_rate": sample_rate, "length_s": length}


# Audio augmentation functions


def amplitude_ctrl(mcs_data: np.ndarray,
                   amplitude_list: list[float]) -> np.ndarray:
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


def delay_ctrl(mcs_data: np.ndarray, delay_us_list: list[int],
               sampling_rate: int = def_fs) -> np.ndarray:
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
    max_samples_delay = int(max(delay_us_list) * 1.E-6 * sampling_rate)  # In samples.

    for signal, delay in zip(mcs_data, delay_us_list):
        samples_delay = int(delay * 1.E-6 * sampling_rate)  # In samples.
        res = np.zeros(samples_delay)
        res = np.append(res, signal)
        if samples_delay < max_samples_delay:
            res = np.append(res, np.zeros(max_samples_delay - samples_delay))
        channels.append(res)
        multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def echo_ctrl(mcs_data, delay_us_list: list[int], amplitude_list: list[float],
              sampling_rate: int = def_fs) -> np.ndarray:
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
    e = delay_ctrl(a, delay_us_list)
    channels = []
    for d in mcs_data:
        zl = e.shape[1] - d.shape[0]
        channels.append(np.append(d, np.zeros(zl)))
    multichannel_sound = np.array(channels).copy() + e

    return multichannel_sound


def noise_ctrl(mcs_data: np.ndarray, noise_level_list: list[float],
               sampling_rate: int = def_fs, seed: int = -1) -> np.ndarray:
    """
    Apply noise to a multichannel sound.

    Parameters:
        mcs_data (np.ndarray): The multichannel sound data.
        noise_level_list (list[float]): The list of noise levels to apply to
        each channel.
        sampling_rate (int): The sampling rate of the sound data. Defaults to
        def_fs.
        seed (int): The seed for random number generation. Defaults to -1.

    Returns:
        np.ndarray: The noise-controlled multichannel sound.
    """

    channels = []
    for signal, level in zip(mcs_data, noise_level_list):
        if seed != -1:
            random.seed(seed)
            n_noise = random_noise_gen.standard_normal(mcs_data.shape[1])
        else:
            # TODO seed should be fixed for repeatable results
            n_noise = random_noise_gen.standard_normal(mcs_data.shape[1])
        noise = n_noise
        # print('noise.shape', noise.shape)
        # print('noise=', noise[0:10])
        res = signal + level * noise
        channels.append(res)
    multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def pause_detect(mcs_data: np.ndarray, relative_level: list[float]):
    """Detect pauses in multichannel sound.

    Args:
    mcs_data - array of shape [channels, samples].
    relative_level - list of relative levels of pause for each channel.

    Returns:
    mask - array of shape [channels, samples], containing zeros and ones.
    0 - pause, 1 - not a pause.
    """

    r = rms(mcs_data)
    a = abs(mcs_data)
    mask = np.zeros(mcs_data.shape)

    for i in range(0, mcs_data.shape[0]):
        ll = r[i]*relative_level[i]
        mask[i] = np.clip(a[i], a_min=ll, a_max=1.1*ll)
        mask[i] -= ll
        mask[i] /= 0.09*ll 
        mask[i] = np.clip(mask[i], a_min=0, a_max=1).astype(int)
    return mask


def pause_shrink(mcs_data: np.ndarray, mask: np.ndarray, min_pause: list[int]):
    """Shrink pauses in multichannel sound."""

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
    """Measure pauses in multichannel sound."""

    chans = mask.shape[0]
    pause_list = []
    out_list = []
    index = 0
    for i in range(0, chans):
        k = 0
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


def pause_set(mcs_data: np.ndarray, pause_map: list, pause_sz: list[int]):
    """Shrink pauses in multichannel sound."""

    chans = mcs_data.shape[0]
    out_list = []
    for i in range(0, chans):
        prev_index = 0
        local_list = []
        for p in pause_map[i]:
            index = p[0] + p[1]
            delta = index - prev_index
            if delta > 0:
                local_list.append(mcs_data[i][prev_index:prev_index + delta])
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


def split(mcs_data, channels_count: int):
    """Split mono signal to several identical channels.

    Returns:
        Output data containing channels_count identical channels.
    """
    out_data = np.zeros((channels_count, mcs_data.shape[1]), dtype=np.float32)
    for i in range(0, channels_count):
        out_data[i] = mcs_data.copy()

    return out_data


def merge(mcs_data):
    """Mix mcs_data channels a to single signal.

    Returns:
        Output data containing 1 channel of mono signal.
    """
    out_data = np.zeros(mcs_data.shape[1], dtype=np.float32)
    print('outdata.shape:', out_data.shape)
    channels_count = mcs_data.shape[0]
    for i in range(0, channels_count):
        out_data += mcs_data[i]

    return out_data


def sum(mcs_data1, mcs_data2):
    """Sum mcs_data1 and mcs_data2 signals.

    Returns:
        Output data containing sum of mcs_data1 and mcs_data2 signals.
    """
    out_data = mcs_data1 + mcs_data2

    return out_data


def side_by_side(mcs_data1, mcs_data2):
    """Join mcs_data1 and mcs_data2 signals side by side.

    Returns:
        Output data containing mcs_data1 and mcs_data2 signals.
    """
    out_data = np.zeros((mcs_data1.shape[0] + mcs_data2.shape[0],
                         mcs_data1.shape[1]), dtype=np.float32)
    out_data[0:mcs_data1.shape[0], :] = mcs_data1
    out_data[mcs_data1.shape[0]:, :] = mcs_data2
    return out_data


def stratch_ctrl(mcs_data, ratio_list, sampling_rate=def_fs):
    """Add pink noise to channels of multichannel sound."""

    stretch_audio("input.wav", "output.wav", ratio=1.1)
    channels = []
    for signal, ratio in zip(mcs_data, ratio_list):
        pknoise = pyplnoise.PinkNoise(sampling_rate, 1e-2, 50.)
        noise = pknoise.get_series(signal.shape[0])
        res = signal + ratio * np.array(noise)
        channels.append(res)
    multichannel_sound = np.array(channels).copy()
    return multichannel_sound


# Chaining class

class WaChain:
    def __init__(self, _data=None, fs=-1):
        self.data = _data
        self.path = ''
        self.sample_rate = fs

    def copy(self):
        return copy.deepcopy(self)

    def put(self, data, fs=-1):
        self.data = data.copy()
        self.sample_rate = fs
        return self

    def get(self):
        return self.data

    def gen(self, f_list, t, fs=def_fs):
        self.data = generate(f_list, t, fs)
        self.sample_rate = fs
        return self

    def rd(self, path):
        self.sample_rate, self.data = read(path)
        return self

    def wr(self, path):
        write(path, self.data, self.sample_rate)
        return self

    def amp(self, amplitude_list):
        self.data = amplitude_ctrl(self.data, amplitude_list)
        return self

    def dly(self, delay_list):
        self.data = delay_ctrl(self.data, delay_list)
        return self

    def ns(self, noise_level_list, sampling_rate=def_fs, seed=-1):
        self.data = noise_ctrl(self.data, noise_level_list,
                                      sampling_rate, seed)
        return self

    def echo(self, delay_us_list, amplitude_list, sampling_rate=def_fs):
        self.data = echo_ctrl(self.data, delay_us_list, amplitude_list,
                                     sampling_rate)
        return self

    def rms(self, last_index=-1, decimals=-1):
        return rms(self.data, last_index, decimals)

    def info(self):
        res = {"path": self.path, "channels_count": -1,
               "sample_rate": self.sample_rate, "length_s": -1}
        if self.data is not None:
            length = self.data.shape[1] / self.sample_rate
            res["channels_count"] = self.data.shape[0]
            res["length_s"] = length
        return res

    def sum(self, mcs_data):    
        self.data = sum(self.data, mcs_data)
        return self

    def mrg(self):
        self.data = merge(self.data)
        return self

    def splt(self, channels_count):
        self.data = split(self.data, channels_count)
        return self
   
    def sbs(self, mcs_data):
        self.data = side_by_side(self.data, mcs_data)
        return self
    
    def pdt(self, relative_level):
        self.data = pause_detect(self.data, relative_level)
        return self

# CLI interface functions
error_mark = "Error: "
prog_name = os.path.basename(__file__).split('.')[0]

application_info = f"{prog_name} application provides functions for \
multichannel WAV audio data augmentation."


def check_amp_list(ls):
    for n in ls:
        try:
            float(n)
        except ValueError:
            print(f"{error_mark}Amplitude list"
                  f" contains non number element: <{n}>.")
            exit(3)


def check_delay_list(ls):
    for n in ls:
        try:
            int(n)
        except ValueError:
            print(f"{error_mark}Delays list"
                  f" contains non integer element: <{n}>.")
            exit(1)


def print_help_and_info():
    """Function prints info about application"""

    print(application_info)
    exit(0)


def chain_hdr(args):
    if args.chain_code is None:
        return
    c = args.chain_code.strip()
    print('chain:', c)
    w = WaChain()
    cmd_prefix = "w."
    str(eval(cmd_prefix + c.strip()))
    print('Done.')
    exit(0)


def input_path_hdr(args):
    """Function checks presence of input file"""
    if args.in_path is None:
        print_help_and_info()
    if not os.path.exists(args.in_path) or not os.path.isfile(args.in_path):
        print(f"{error_mark}Input file <{args.in_path}> not found.")
        exit(1)


def is_file_creatable(fullpath: str) -> bool:
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
        exit(1)

    return True


def output_path_hdr(args):
    """Function checks of output file name and path."""

    if not is_file_creatable(args.out_path):
        print(f"{error_mark}Can't create file <{args.out_path}>.")
        exit(1)


def file_info_hdr(args):
    """Function prints info about input audio file."""

    print()
    if args.info:
        for key, value in file_info(args.path).items():
            print(f"{key}: {value}")
        exit(0)


def amplitude_hdr(args):
    """Function makes amplitude augmentation."""

    if args.amplitude_list is None:
        return

    amplitude_list = args.amplitude_list.split(',')
    check_amp_list(amplitude_list)

    float_list = [float(i) for i in amplitude_list]
    print(f"amplitudes: {float_list}")
    info = file_info(args.in_path)
    if info['channels_count'] != len(float_list):
        print(f"{error_mark}Amplitude list length <{len(float_list)}>"
              " does not match number of channels. It should have"
              f" <{info['channels_count']}> elements.")
        exit(2)
    _, mcs_data = read(args.in_path)
    res_data = amplitude_ctrl(mcs_data, float_list)
    write(args.out_path, res_data, info['sample_rate'])
    print('Done.')
    exit(0)


def noise_hdr(args):
    """Function makes noise augmentation."""

    if args.noise_list is None:
        return

    noise_list = args.noise_list.split(',')
    check_amp_list(noise_list)

    float_list = [float(i) for i in noise_list]
    print(f"noise levels: {float_list}")
    info = file_info(args.in_path)
    if info['channels_count'] != len(float_list):
        print(f"{error_mark}Noise list length <{len(float_list)}>"
              " does not match number of channels. It should have"
              f" <{info['channels_count']}> elements.")
        exit(2)
    _, mcs_data = read(args.in_path)
    res_data = noise_ctrl(mcs_data, float_list)
    write(args.out_path, res_data, info['sample_rate'])
    print('Done.')
    exit(0)

def echo_hdr(args):
    """Function makes echo augmentation."""

    if args.echo_list is None:
        return

    lists = args.echo_list.split('/')
    if len(lists) != 2:
        print(f"{error_mark}Can't distinguish delay and amplitude lists <{args.echo_list}>.")
        exit(1)
    
    delay_list = lists[0].split(',')
    amplitude_list = lists[1].split(',')
    if len(amplitude_list) != len(delay_list):
        print(f"{error_mark}Can't delay and amplitude lists length differ <{args.echo_list}>.")
        exit(2)

    check_delay_list(delay_list)
    check_amp_list(amplitude_list)

    int_list = [int(i) for i in delay_list]
    print(f"delays: {int_list}")
    info = file_info(args.in_path)
    if info['channels_count'] != len(int_list):
        print(f"{error_mark}Delay list length <{len(int_list)}>"
              " does not match number of channels. It should have"
              f" <{info['channels_count']}> elements.")
        exit(2)
    
    float_list = [float(i) for i in amplitude_list]
    print(f"amplitudes: {float_list}")

    _, mcs_data = read(args.in_path)
    res_data = echo_ctrl(mcs_data, int_list, float_list)

    write(args.out_path, res_data, info['sample_rate'])
    print('Done.')
    exit(0)


def delay_hdr(args):
    """Function makes delay augmentation."""

    if args.delay_list is None:
        return

    delay_list = args.delay_list.split(',')
    check_delay_list(delay_list)

    int_list = [int(i) for i in delay_list]
    print(f"delays: {int_list}")
    info = file_info(args.in_path)
    if info['channels_count'] != len(int_list):
        print(f"{error_mark}Delays list length <{len(int_list)}>"
              " does not match number of channels. It should have"
              f" <{info['channels_count']}> elements.")
        exit(2)
    _, mcs_data = read(args.in_path)
    res_data = delay_ctrl(mcs_data, int_list)
    write(args.out_path, res_data, info['sample_rate'])
    print('Done.')
    exit(0)


def parse_args():
    """CLI options parsing."""

    parser = argparse.ArgumentParser(
        prog=prog_name,
        description='WAV audio files augmentation utility.',
        epilog='Text at the bottom of help')

    parser.add_argument('-i', dest='in_path', help='Input audio'
                        ' file path.')
    parser.add_argument('-o', dest='out_path', help='Output audio file path.')
    parser.add_argument('--info', dest='info', action='store_true',
                        help='Print info about input audio file.')
    parser.add_argument('--amp', '-a', dest='amplitude_list',
                        help='Change amplitude (volume)'
                        ' of channels in audio file. Provide coefficients for'
                        ' every channel, example:\n\t -a "0.1, 0.2, 0.3, -1"')
    parser.add_argument('--echo', '-e', dest='echo_list',
                        help='Add echo to channels in audio file.'
                        ' of channels in audio file. Provide coefficients'
                        '  and delays (in microseconds) of '
                        ' reflected signal for every channel, example:\n\t'
                        '      -e "0.1, 0.2, 0.3, -1 / 100, 200, 0, 300"')
    parser.add_argument('--dly', '-d', dest='delay_list', type=str,
                        help='Add time delays'
                        ' to channels in audio file. Provide delay for'
                        ' every channel in microseconds, example:\n\t \
                            -d "100, 200, 300, 0"')
    parser.add_argument('--ns', '-n', dest='noise_list',
                        help='Add normal noise'
                        ' to channels in audio file. Provide coefficients for'
                        ' every channel, example:\n\t -n "0.1, 0.2, 0.3, -1"')                            
    parser.add_argument('--chain', '-c', dest='chain_code', type=str,
                        help='Execute chain of transformations.'
                        ' example:\n\t'
                        '-c "gen([100,250,100], 3, 44100).amp([0.1, 0.2, 0.3])'
                        '.wr("./sines.wav")"')

    return parser.parse_args()


def main():
    """Argument parsing."""

    args = parse_args()
    chain_hdr(args)
    input_path_hdr(args)
    file_info_hdr(args)
    output_path_hdr(args)
    amplitude_hdr(args)
    noise_hdr(args)
    delay_hdr(args)
    echo_hdr(args)


if __name__ == '__main__':
    main()
