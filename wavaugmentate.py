#!/usr/bin/env python3

import numpy as np
import argparse
from scipy.io import wavfile
import os
from pathlib import Path

def_fs = 44100  # Hz

random_noise_gen = np.random.default_rng()


def mcs_rms(mcs_data, last_index=-1):
    """Return RMS of multichannel sound."""

    res = []
    for signal in mcs_data:
        res.append(np.sqrt(np.mean(signal[0:last_index]**2)))
    return res


def mcs_generate(frequency_list, duration, sample_rate=def_fs):
    """Function generates multichannel sound as a set of sin-waves.

    return numpy array of shape [channels, samples]
    """

    samples = np.arange(duration * sample_rate) / sample_rate
    channels = []
    for f in frequency_list:
        signal = np.sin(2 * np.pi * f * samples)
        signal = np.float32(signal)
        channels.append(signal)
    multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def mcs_write(path, mcs_data, sample_rate=44100):
    """ Save multichannel sound to wav-file.

    path : string or open file handle
    Output wav file.

    sample_rate : int
    The sample rate (in samples/sec).
    """
    buf = mcs_data.T.copy()
    wavfile.write(path, sample_rate, buf)


def mcs_read(path):
    """ Read multichannel sound from wav-file.

    return sample_rate, mcs_data.
    """
    sample_rate, buf = wavfile.read(path)
    mcs_data = buf.T.copy()
    return sample_rate, mcs_data


def mcs_file_info(path):
    """ Return information about multichannel sound from wav-file.

    return path, channels_count, sample_rate, length (in seconds) of file.
    """
    sample_rate, buf = wavfile.read(path)
    length = buf.shape[0] / sample_rate

    return {"path": path, "channels_count": buf.shape[1],
            "sample_rate": sample_rate, "length_s": length}


# Audio augmentation functions


def mcs_amplitude_control(mcs_data, amplitude_list):
    """ Change amplitude of multichannel sound."""
    channels = [] 
    for signal, amplitude in zip(mcs_data, amplitude_list):
        channels.append(signal * amplitude)
        multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def mcs_delay_control(mcs_data, delay_us_list, sampling_rate=def_fs):
    """Add delays of channels of multichannel sound. Output data become longer."""

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


def mcs_echo_control(mcs_data, delay_us_list, amplitude_list, sampling_rate=def_fs):
    """Add echo to multichannel sound.

    Returns:
        Output data become longer.
    """
    a = mcs_amplitude_control(mcs_data, amplitude_list)
    e = mcs_delay_control(a, delay_us_list)
    channels = []
    for d in mcs_data:
        zl = e.shape[1] - d.shape[0]
        channels.append(np.append(d, np.zeros(zl)))
    multichannel_sound = np.array(channels).copy() + e

    return multichannel_sound


def mcs_noise_control(mcs_data, noise_level_list, sampling_rate=def_fs, seed=-1):
    """ Add pink noise to channels of multichannel sound."""

    channels = []
    for signal, level in zip(mcs_data, noise_level_list):
        if seed != -1:
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


def mcs_stratch_control(mcs_data, ratio_list, sampling_rate=def_fs):
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

    def put(self, data, fs=-1):
        self.data = data.copy()
        self.sample_rate = fs
        return self

    def get(self):
        return self.data

    def gen(self, f_list, t, fs=def_fs):
        self.data = mcs_generate(f_list, t, fs)
        self.sample_rate = fs
        return self

    def rd(self, path):
        self.sample_rate, self.data = mcs_read(path)
        return self

    def wr(self, path):
        mcs_write(path, self.data, self.sample_rate)
        return self

    def amp(self, amplitude_list):
        self.data = mcs_amplitude_control(self.data, amplitude_list)
        return self

    def dly(self, delay_list):
        self.data = mcs_delay_control(self.data, delay_list)
        return self

    def ns(self, noise_level_list, sampling_rate=def_fs, seed=-1):
        self.data = mcs_noise_control(self.data, noise_level_list,
                                      sampling_rate, seed)
        return self

    def echo(self, delay_us_list, amplitude_list, sampling_rate=def_fs):
        self.data = mcs_echo_control(self.data, delay_us_list, amplitude_list,
                                     sampling_rate)
        return self

    def rms(self, last_index=-1):
        return mcs_rms(self.data, last_index)

    def info(self):
        res = {"path": self.path, "channels_count": -1,
               "sample_rate": self.sample_rate, "length_s": -1}
        if self.data is not None:
            length = self.data.shape[1] / self.sample_rate
            res["channels_count"] = self.data.shape[0]
            res["length_s"] = length
        return res


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
        for key, value in mcs_file_info(args.path).items():
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
    info = mcs_file_info(args.in_path)
    if info['channels_count'] != len(float_list):
        print(f"{error_mark}Amplitude list length <{len(float_list)}>"
              " does not match number of channels. It should have"
              f" <{info['channels_count']}> elements.")
        exit(2)
    _, mcs_data = mcs_read(args.in_path)
    res_data = mcs_amplitude_control(mcs_data, float_list)
    mcs_write(args.out_path, res_data, info['sample_rate'])
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
    info = mcs_file_info(args.in_path)
    if info['channels_count'] != len(float_list):
        print(f"{error_mark}Noise list length <{len(float_list)}>"
              " does not match number of channels. It should have"
              f" <{info['channels_count']}> elements.")
        exit(2)
    _, mcs_data = mcs_read(args.in_path)
    res_data = mcs_noise_control(mcs_data, float_list)
    mcs_write(args.out_path, res_data, info['sample_rate'])
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
    info = mcs_file_info(args.in_path)
    if info['channels_count'] != len(int_list):
        print(f"{error_mark}Delay list length <{len(int_list)}>"
              " does not match number of channels. It should have"
              f" <{info['channels_count']}> elements.")
        exit(2)
    
    float_list = [float(i) for i in amplitude_list]
    print(f"amplitudes: {float_list}")

    _, mcs_data = mcs_read(args.in_path)
    res_data = mcs_echo_control(mcs_data, int_list, float_list)

    mcs_write(args.out_path, res_data, info['sample_rate'])
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
    info = mcs_file_info(args.in_path)
    if info['channels_count'] != len(int_list):
        print(f"{error_mark}Delays list length <{len(int_list)}>"
              " does not match number of channels. It should have"
              f" <{info['channels_count']}> elements.")
        exit(2)
    _, mcs_data = mcs_read(args.in_path)
    res_data = mcs_delay_control(mcs_data, int_list)
    mcs_write(args.out_path, res_data, info['sample_rate'])
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
