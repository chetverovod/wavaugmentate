#!/usr/bin/env python3

"""
This module does multichannel audio flies augmentation.
"""

__author__ = "Igor Plastov"
__version__ = '0.2.5'

import argparse
import os
import ast
import logging as log
from pathlib import Path
from typing import List
from scipy.io import wavfile
import mcs as ms
from mcs import MultiChannelSignal
from aug import SignalAugmentation


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
for multichannel WAV audio data augmentation."


def validate_amp_list(amplitude_list: List[str]) -> None:
    """
    Checks if all elements in the given amplitudes list are valid numbers.

    Args:
        amplitude_list (list[str]): The list of elements to check.

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
        delays_list (list[str]): The list of elements to check.

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

    chain = args.chain_code.strip()
    print(f'chain:\n{chain}')
    chunks = chain.split(").")
    chunks = [f'{e})' for e in chunks]
    prog = []
    for element in chunks:
        cmd, brackets = element.split('(')
        brackets = brackets.strip(')')
        prog.append([cmd, brackets])
    aug_obj = SignalAugmentation()
    i = 1
    # print(f'steps:{len(prog)}')
    for step in prog:
        input_string = step[1]
        # print(f'input string {i}: {input_string}')
        i += 1
        if len(input_string) > 0:
            arguments = ast.literal_eval(input_string)
        else:
            arguments = ()

        if isinstance(arguments, tuple):
            unpacked = list(arguments)
        else:
            unpacked = arguments

        # print(f'arg type: ({type(unpacked)})')
        # print(f'cmd {step[0]}({unpacked})')

        if isinstance(unpacked, (tuple, list)):
            aug_obj = getattr(aug_obj, step[0])(*unpacked)
        elif isinstance(unpacked, str):
            aug_obj = getattr(aug_obj, step[0])(unpacked)
        else:
            raise ValueError(f"Unsupported object type: {type(unpacked)}")
    print(ms.SUCCESS_MARK)
    aug_obj.info()


def input_path_validation(in_path) -> str:
    """Function checks presence of input file"""

    if in_path is None:
        print_help_and_info()
    if not os.path.exists(in_path) or not os.path.isfile(in_path):
        msg = f"Input file <{in_path}> not found."
        log.error(msg)
        print(msg)
        raise ValueError(msg)
    return in_path


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


def output_path_validation(out_path):
    """Function checks of output file name and path."""

    if not is_file_creatable(out_path):
        msg = f"Can't create file <{out_path}>."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)
    return out_path


def file_info_hdr(args):
    """Function prints info about input audio file."""

    print()
    print(args.info_path)
    for key, value in file_info(args.info_path).items():
        print(f"{key}: {value}")


def amplitude_hdr(args):
    """Function makes CLI amplitude augmentation."""

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
    aug_obj = SignalAugmentation(mcs)
    aug_obj.amplitude_ctrl(float_list)
    aug_obj.get().write(args.out_path)
    print(ms.SUCCESS_MARK)


def noise_hdr(args):
    """Function makes CLI noise augmentation."""

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
    aug_obj = SignalAugmentation(mcs)
    aug_obj.noise_ctrl(float_list)
    aug_obj.get().write(args.out_path)
    print(ms.SUCCESS_MARK)


def echo_hdr(args):
    """Function makes CLI echo augmentation."""

    lists = args.echo_list.split("/")
    delay_list = lists[0].split(",")
    amplitude_list = lists[1].split(",")
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
    aug_obj = SignalAugmentation(mcs)
    aug_obj.echo_ctrl(int_list, float_list)
    aug_obj.get().write(args.out_path)
    print(ms.SUCCESS_MARK)


def delay_hdr(args):
    """Function makes CLI delay augmentation."""

    delay_list = args.delay_list.split(",")
    validate_delay_list(delay_list)

    int_list = [int(i) for i in delay_list]
    print(f"\ndelays: {int_list}")
    info = file_info(args.in_path)
    if info["channels_count"] != len(int_list):
        msg = f"Delays list length <{len(int_list)}>" \
            " does not match number of channels. It should have" \
            f" <{info['channels_count']}> elements."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)

    mcs = ms.MultiChannelSignal().read(args.in_path)
    aug_obj = SignalAugmentation(mcs)
    aug_obj.delay_ctrl(int_list)
    aug_obj.get().write(args.out_path)
    print(ms.SUCCESS_MARK)


def echo_args_validation(echo_list) -> str:
    """Function make external check of args for option echo."""

    if echo_list is None:
        msg = "echo_list is None"
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise argparse.ArgumentTypeError(msg)

    try:
        arg_string = str(echo_list)
        if len(arg_string) == 0:
            raise ValueError(f"{arg_string} must be a not empty string")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))

    lists = echo_list.split("/")
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
              f" differ <{echo_list}>."
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise ValueError(msg)

    delay_args_validation(lists[0])
    amp_args_validation(lists[1])
    return echo_list


def delay_args_validation(delay_list_str) -> str:
    """Function make external check of args for option dly."""

    if delay_list_str is None:
        msg = "delay_list is None"
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise argparse.ArgumentTypeError(msg)

    try:
        arg_string = str(delay_list_str)
        if len(arg_string) == 0:
            raise ValueError(f"{arg_string} must be a not empty string")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))

    delay_list = delay_list_str.split(",")
    validate_delay_list(delay_list)
    return delay_list_str


def amp_args_validation(amplitude_list_str) -> str:
    """Function make external check of args for option dly."""

    if amplitude_list_str is None:
        msg = "amplitudes_list is None"
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise argparse.ArgumentTypeError(msg)

    try:
        arg_string = str(amplitude_list_str)
        if len(arg_string) == 0:
            raise ValueError(f"{arg_string} must be a not empty string")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))

    amplitude_list = amplitude_list_str.split(",")
    validate_amp_list(amplitude_list)
    return amplitude_list_str


def noise_args_validation(noise_list_str) -> str:
    """Function make external check of args for option noise."""

    if noise_list_str is None:
        msg = "noise_list is None"
        print(f"{ms.ERROR_MARK}{msg}")
        log.error(msg)
        raise argparse.ArgumentTypeError(msg)

    try:
        arg_string = str(noise_list_str)
        if len(arg_string) == 0:
            raise ValueError(f"{arg_string} must be a not empty string")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))

    noise_list = noise_list_str.split(",")
    validate_amp_list(noise_list)
    return noise_list_str


def parse_args():
    """CLI options parsing."""

    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=("WAV audio files augmentation utility. Developed by"
                     f" {__author__}, chetverovod@gmail.com."),
        epilog="",  # "Text at the bottom of help"
    )
    parser.add_argument("-v", "--version", action="store_true", help="Version "
                        "information.")
    parser.add_argument(
        "-i",
        type=input_path_validation,
        dest="in_path",
        help="Input audio file path."
        )
    parser.add_argument(
        "-o",
        type=output_path_validation,
        dest="out_path", help="Output audio file path."
        )
    parser.add_argument(
        "--info",
        type=input_path_validation,
        dest="info_path",
        help="Print info about input audio file.",
    )
    parser.add_argument(
        "--amp",
        "-a",
        dest="amplitude_list",
        type=amp_args_validation,
        help="Change amplitude (volume)"
        " of channels in audio file. Provide coefficients for"
        ' every channel, example:\n\t -a "0.1, 0.2, 0.3, -1"',
    )
    parser.add_argument(
        "--echo",
        "-e",
        dest="echo_list",
        type=echo_args_validation,
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
        type=delay_args_validation,
        help="Add time delays"
        " to channels in audio file. Provide delay for"
        ' every channel in microseconds, example:\n\t \
                            -d "100, 200, 300, 0"',
    )
    parser.add_argument(
        "--ns",
        "-n",
        dest="noise_list",
        type=noise_args_validation,
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
        '-c \'gen([100,250,100], 3, 44100).amp([0.1, 0.2, 0.3], None)'
        '.wr("./sines.wav")"\'',
    )

    # Check presence of known args.
    known_args, unknown_args = parser.parse_known_args()
    if not known_args.__dict__:
        print_help_and_info()
        return None

    # Check presence of unknown args.
    if unknown_args:
        msg = f'Unknown arguments: {unknown_args}'
        print(msg)
        return None

    #return parser.parse_args()
    return known_args


def augmentate(args):
    """
    Augments the input audio file based on the provided arguments.

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

    if args.chain_code is not None:
        chain_hdr(args)
    elif args.info_path is not None:
        file_info_hdr(args)
    elif args.in_path is None:
        print_help_and_info()
    elif args.amplitude_list is not None:
        amplitude_hdr(args)
    elif args.noise_list is not None:
        noise_hdr(args)
    elif args.delay_list is not None:
        delay_hdr(args)
    elif args.echo_list is not None:
        echo_hdr(args)


def main():
    """CLI arguments parsing."""

    args = parse_args()
    if args is None:
        return
    if args.version is True:
        print(__version__)
        return
    augmentate(args)


if __name__ == "__main__":
    main()
