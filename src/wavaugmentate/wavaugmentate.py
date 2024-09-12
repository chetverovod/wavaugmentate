#!/usr/bin/env python3

"""
This module does multichannel audio flies augmentation.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from scipy.io import wavfile
import mcs as ms
from mcs import Mcs
from aug import Aug


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


# CLI interface functions

prog_name = os.path.basename(__file__).split(".")[0]

application_info = f"{prog_name.capitalize()} application provides functions for \
multichannel WAV audio data augmentation."


def validate_amp_list(amplitude_list: List[str]) -> None:
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
    for amplitude_value in amplitude_list:
        try:
            float(amplitude_value)
        except ValueError:
            print(
                f"{ms.ERROR_MARK}Amplitude list"
                f" contains non number element: <{amplitude_value}>."
            )
            sys.exit(3)


def validate_delay_list(delays_list: List[str]) -> None:
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
    for delay_value in delays_list:
        try:
            int(delay_value)
        except ValueError:
            print(
                f"{ms.ERROR_MARK}Delays list"
                f" contains non integer element: <{delay_value}>."
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
    chain = args.chain_code.strip()
    print("chain:", chain)
    aug_obj = Aug()
    cmd_prefix = "aug_obj."
    str(eval(cmd_prefix + chain.strip()))  # It is need for chain commands.
    print(ms.SUCCESS_MARK)
    aug_obj.info()
    sys.exit(0)


def input_path_hdr(args):
    """Function checks presence of input file"""
    if args.in_path is None:
        print_help_and_info()
    if not os.path.exists(args.in_path) or not os.path.isfile(args.in_path):
        print(f"{ms.ERROR_MARK}Input file <{args.in_path}> not found.")
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
            print(f"{ms.ERROR_MARK}Can't create file <{fullpath}>.")
            raise
    else:
        print(f"{ms.ERROR_MARK}Path <{path}> is not exists.")
        sys.exit(1)

    return True


def output_path_hdr(args):
    """Function checks of output file name and path."""

    if not is_file_creatable(args.out_path):
        print(f"{ms.ERROR_MARK}Can't create file <{args.out_path}>.")
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
    validate_amp_list(amplitude_list)

    float_list = [float(i) for i in amplitude_list]
    print(f"amplitudes: {float_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(float_list):
        print(
            f"{ms.ERROR_MARK}Amplitude list length <{len(float_list)}>"
            " does not match number of channels. It should have"
            f" <{info['channels_count']}> elements."
        )
        sys.exit(2)

    mcs = Mcs().read(args.in_path)
    aug_obj = Aug(mcs)
    aug_obj.amplitude_ctrl(float_list)
    aug_obj.get().write(args.out_path)
    print(ms.SUCCESS_MARK)
    sys.exit(0)


def noise_hdr(args):
    """Function makes CLI noise augmentation."""

    if args.noise_list is None:
        return

    noise_list = args.noise_list.split(",")
    validate_amp_list(noise_list)

    float_list = [float(i) for i in noise_list]
    print(f"noise levels: {float_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(float_list):
        print(
            f"{ms.ERROR_MARK}Noise list length <{len(float_list)}>"
            " does not match number of channels. It should have"
            f" <{info['channels_count']}> elements."
        )
        sys.exit(2)

    mcs = ms.Mcs().read(args.in_path)
    mcs.read(args.in_path)
    aug_obj = Aug(mcs)
    aug_obj.noise_ctrl(float_list)
    aug_obj.get().write(args.out_path)
    print(ms.SUCCESS_MARK)
    sys.exit(0)


def echo_hdr(args):
    """Function makes CLI echo augmentation."""

    if args.echo_list is None:
        return

    lists = args.echo_list.split("/")
    if len(lists) != 2:
        print(
            f"{ms.ERROR_MARK}Can't distinguish delay and amplitude"
            f"lists <{args.echo_list}>."
        )
        sys.exit(1)

    delay_list = lists[0].split(",")
    amplitude_list = lists[1].split(",")
    if len(amplitude_list) != len(delay_list):
        print(
            f"{ms.ERROR_MARK}Can't delay and amplitude lists lengths"
            f"differ <{args.echo_list}>."
        )
        sys.exit(2)

    validate_delay_list(delay_list)
    validate_amp_list(amplitude_list)

    int_list = [int(i) for i in delay_list]
    print(f"delays: {int_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(int_list):
        print(
            f"{ms.ERROR_MARK}Delay list length <{len(int_list)}>"
            " does not match number of channels. It should have"
            f" <{info['channels_count']}> elements."
        )
        sys.exit(2)

    float_list = [float(i) for i in amplitude_list]
    print(f"amplitudes: {float_list}")

    mcs = ms.Mcs().read(args.in_path)
    aug_obj = Aug(mcs)
    aug_obj.echo_ctrl(int_list, float_list)
    aug_obj.get().write(args.out_path)
    print(ms.SUCCESS_MARK)
    sys.exit(0)


def delay_hdr(args):
    """Function makes CLI delay augmentation."""

    if args.delay_list is None:
        return

    delay_list = args.delay_list.split(",")
    validate_delay_list(delay_list)

    int_list = [int(i) for i in delay_list]
    print(f"delays: {int_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(int_list):
        print(
            f"{ms.ERROR_MARK}Delays list length <{len(int_list)}>"
            " does not match number of channels. It should have"
            f" <{info['channels_count']}> elements."
        )
        sys.exit(2)

    mcs = ms.Mcs().read(args.in_path)
    aug_obj = Aug(mcs)
    aug_obj.delay_ctrl(int_list)
    aug_obj.get().write(args.out_path)
    print(ms.SUCCESS_MARK)
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
