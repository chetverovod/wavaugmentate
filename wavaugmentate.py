import mcsaugment as ma
import argparse
import os
from pathlib import Path


def print_help_and_info(args):
    """Function prints ifo about application"""

    print('Wavaugmentate application provides functions for multichannel WAVaudio \
          data augmentation.')
    exit(0)


def input_path_hdr(args):
    """Function checks presence of input file"""

    if not os.path.exists(args.in_path) or not os.path.isfile(args.in_path):
        print(f"Input file <{args.in_path}> not found.")
        exit(1)


def is_file_creatable(fullpath: str) -> bool:
    
    # Split the path
    path, _ = os.path.split(fullpath)
    isdir = os.path.isdir(path)
    if isdir:
        try:
            Path(fullpath).touch(mode=0o777, exist_ok=True)
        except Exception:
            print(f"Can't create file <{fullpath}>.")
            raise
    else:
        print(f"Path <{path}> is not exists.")
        exit(1)

    return True


def output_path_hdr(args):
    """Function checks of output file name and path."""

    if not is_file_creatable(args.out_path):
        print(f"Can't create file <{args.out_path}>.")
        exit(1)


def file_info_hdr(args):
    """Function prints info about input audio file."""

    print()
    if args.info:
        for key, value in ma.mcs_file_info(args.path).items():
            print(f"{key}: {value}")
        exit(0)


def amplitude_hdr(args):
    """Function makes amplitude augmentation."""
    if args.amplitude_list == '':
        return

    amplitude_list = args.amplitude_list.split(',')
    float_amplitude_list = [float(i) for i in amplitude_list]
    print(f"amplitudes: {float_amplitude_list}")
    info = ma.mcs_file_info(args.in_path)
    if info['channels_count'] != len(float_amplitude_list):
        print(f"Amplitude list length <{len(float_amplitude_list)}>"
              " does not match number of channels. It should have"
              f" <{info['channels_count']}> elements.")
        exit(1)
    fs, mcs_data = ma.mcs_read(args.in_path)
    res_data = ma.mcs_amplitude_control(mcs_data, float_amplitude_list)
    ma.mcs_write(args.out_path, res_data, info['sample_rate'])
    print('Done.')
    exit(0)


def parse_args():
    """Настройка argparse"""

    parser = argparse.ArgumentParser(
        prog='WavAugmenator',
        description='Wav audio files augmentation utility.',
        epilog='Text at the bottom of help')

    parser.add_argument('-i', required=True, dest='in_path', help='Input audio file path.')
    parser.add_argument('-o', dest='out_path', help='Output audio file path.')
    parser.add_argument('--info', dest='info', action='store_true', help='Print info about input audio file.')
    parser.add_argument('--amplitude', '-a', dest='amplitude_list',  help='Change amplitude (volume) \
                         of channels in audio file. Provide coefficients for every \
                         channel, example: -a "0.1, 0.2, 0.3, -1"')

    return parser.parse_args()


def main():
    """Argument parsing."""

    args = parse_args()
    input_path_hdr(args)
    file_info_hdr(args)
    output_path_hdr(args)
    amplitude_hdr(args)


main()
