#!/usr/bin/env python3

"""
This module does multichannel audio flies augmentation.
"""

import argparse
import os
import sys
import logging as log
from pathlib import Path
from typing import List
from scipy.io import wavfile
import mcs as ms
from mcs import MultiChannelSignal
from aug import AudioAugmentation


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

APPLICATION_INFO = f"{prog_name.capitalize()} application provides functions \
Gfor multichannel WAV audio data augmentation."


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
        except Exception as exc:
            msg = "Amplitude list contains non number element:" \
                  f"<{amplitude_value}>."
            print(f"{ms.ERROR_MARK}{msg}")
            log.error(msg)
            raise ValueError(msg) from exc


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
        except Exception as exc:
            msg = f"Delays list contains non integer element: <{delay_value}>."
            print(f"{ms.ERROR_MARK}{msg}")
            log.error(msg)
            raise ValueError(msg) from exc


def print_help_and_info():
    """Function prints info about application"""

    print(APPLICATION_INFO)
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
    aug_obj = AudioAugmentation()
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
        msg = f"Input file <{args.in_path}> not found."
        log.error(msg)
        print(msg)
        raise ValueError(msg)


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
        except Exception as exc:
            msg = f"Can't create file <{fullpath}>."
            log.error(msg)
            print(f"{ms.ERROR_MARK}{msg}")
            raise ValueError(msg) from exc
    else:
        msg = f"Path <{path}> is not exists."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)

    return True


def output_path_hdr(args):
    """Function checks of output file name and path."""

    if not is_file_creatable(args.out_path):
        msg = f"Can't create file <{args.out_path}>."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)


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
        msg = f"Amplitude list length <{len(float_list)}>" \
             " does not match number of channels. It should have" \
             f" <{info['channels_count']}> elements."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)

    mcs = MultiChannelSignal().read(args.in_path)
    aug_obj = AudioAugmentation(mcs)
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
        msg = f"Noise list length <{len(float_list)}>" \
            " does not match number of channels. It should have" \
            f" <{info['channels_count']}> elements."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)

    mcs = ms.MultiChannelSignal().read(args.in_path)
    mcs.read(args.in_path)
    aug_obj = AudioAugmentation(mcs)
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
        msg = "Can't distinguish delay and amplitude" \
             "lists <{args.echo_list}>."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)

    delay_list = lists[0].split(",")
    amplitude_list = lists[1].split(",")
    if len(amplitude_list) != len(delay_list):
        msg = "Can't delay and amplitude lists lengths" \
              f" differ <{args.echo_list}>."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)

    validate_delay_list(delay_list)
    validate_amp_list(amplitude_list)

    int_list = [int(i) for i in delay_list]
    print(f"delays: {int_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(int_list):
        msg = f"Delay list length <{len(int_list)}>" \
              " does not match number of channels. It should have" \
              f" <{info['channels_count']}> elements."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)

    float_list = [float(i) for i in amplitude_list]
    print(f"amplitudes: {float_list}")

    mcs = ms.MultiChannelSignal().read(args.in_path)
    aug_obj = AudioAugmentation(mcs)
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
        msg = f"Delays list length <{len(int_list)}>" \
            " does not match number of channels. It should have" \
            f" <{info['channels_count']}> elements."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)

    mcs = ms.MultiChannelSignal().read(args.in_path)
    aug_obj = AudioAugmentation(mcs)
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
        '-c \'gen([100,250,100], 3, 44100).amp([0.1, 0.2, 0.3])'
        '.wr("./sines.wav")"\'',
    )

    return parser.parse_args()


def augmentate(args):
    """
    Augmentates the input audio file based on the provided arguments.

    Args:
        args (argparse.Namespace): The command line arguments.

    Returns:
        None

    Raises:
        None

    This function performs the following steps:

    1. Calls the `chain_hdr` function to process the chain code from the
       arguments and execute the corresponding WaChain commands.
    2. Calls the `input_path_hdr` function to validate the input path.
    3. Calls the `file_info_hdr` function to retrieve information about the
       input file.
    4. Calls the `output_path_hdr` function to validate the output path.
    5. Calls the `amplitude_hdr` function to perform amplitude augmentation on
       the input file.
    6. Calls the `noise_hdr` function to perform noise augmentation on the
       input file.
    7. Calls the `delay_hdr` function to perform time delay augmentation on the
       input file.
    8. Calls the `echo_hdr` function to perform echo augmentation on the input
       file.

    Note: This function does not return any value. It is expected to be called
    from the main function of the program.
    """

    chain_hdr(args)
    input_path_hdr(args)
    file_info_hdr(args)
    output_path_hdr(args)
    amplitude_hdr(args)
    noise_hdr(args)
    delay_hdr(args)
    echo_hdr(args)


def main():
    """CLI arguments parsing."""

    args = parse_args()
    augmentate(args)


if __name__ == "__main__":
    main()
