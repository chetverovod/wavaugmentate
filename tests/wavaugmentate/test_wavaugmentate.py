"""Module providing test functions for wavaugmentate.py  module."""

import os
import subprocess as sp
import common_test_functions as ctf
import mcs as ms
from mcs import MultiChannelSignal as Mcs
import wavaug as wau


def test_echo_ctrl_option():
    """
    Test function to verify the functionality of the `echo_ctrl` option in the
    command line interface.

    This function generates a multichannel sound using the `generate` function
    from the `ma` module with the given `f_list`, `t`, and `fs` parameters.
    It then writes the generated sound to a file using the `write` function
    from the `ma` module with the given file path and `fs` parameters.

    The function then constructs a command to apply echo control to the
    generated sound using the `echo_ctrl` option in the command line interface.
    It runs the command and captures the output.

    Finally, it compares the captured output to the expected output and
    verifies that the output file exists and has the correct shape and RMS
    values.

    Args:
        None

    Returns:
        None
    """

    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)

    test_sound_1 = Mcs(samp_rt=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)

    cmd = [
        ctf.PROG_NAME,
        "-i",
        ctf.TEST_SOUND_1_FILE,
        "-o",
        ctf.OUTPUT_FILE,
        "-e",
        "100, 300, 400, 500 / 0.5, 0.6, 0.7, 0.1 ",
    ]
    print("\n", " ".join(cmd))
    if os.path.exists(ctf.OUTPUT_FILE):
        os.remove(ctf.OUTPUT_FILE)
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    responce_string = str(res.stdout)
    out = ctf.shrink(responce_string)
    print("out:", out)
    full_ref = (
        "\ndelays: [100, 300, 400, 500]\n"
        + f"\namplitudes: [0.5, 0.6, 0.7, 0.1]\n{ms.SUCCESS_MARK}\n"
    )
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(ctf.OUTPUT_FILE)

    written = Mcs()
    written.read(ctf.OUTPUT_FILE)
    for channel in written.data:
        assert channel.shape[0] == 220522
    rms_list = written.rms(decimals=3)
    print("rms_list:", rms_list)
    reference_list = [1.054, 0.716, 1.144, 0.749]
    for rms_value, ref in zip(rms_list, reference_list):
        assert abs(rms_value - ref) < ctf.ABS_ERR


def test_wavaugmentate_noise_option():
    """
    Test function to verify the functionality of the `noise` option in the
    command line interface.

    This function generates a multichannel sound using the `generate` function
    from the `wau` module with the given `f_list`, `t`, and `fs` parameters.
    It then writes the generated sound to a file using the `write` function
    from the `wau` module with the given file path and `fs` parameters.

    The function then constructs a command to apply noise to the generated
    sound using the `noise` option in the command line interface. It runs the
    command and captures the output.

    Finally, it compares the captured output to the expected output and
    verifies that the output file exists and has the correct shape and RMS
    values.

    Args:
        None

    Returns:
        None
    """
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    test_sound_1 = Mcs(samp_rt=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)

    cmd = [
        ctf.PROG_NAME,
        "-i",
        ctf.TEST_SOUND_1_FILE,
        "-o",
        ctf.OUTPUT_FILE,
        "-n",
        "0.5, 0.6, 0.7, 0.1",
    ]
    print("\n", " ".join(cmd))
    if os.path.exists(ctf.OUTPUT_FILE):
        os.remove(ctf.OUTPUT_FILE)
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    responce_value = str(res.stdout)
    out = ctf.shrink(responce_value)
    print("out:", out)
    full_ref = f"\nnoise levels: [0.5, 0.6, 0.7, 0.1]\n{ms.SUCCESS_MARK}\n"
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(ctf.OUTPUT_FILE)
    written = Mcs()
    written.read(ctf.OUTPUT_FILE)
    for channel in written.data:
        assert channel.shape[0] == 220500
    rms_list = written.rms(decimals=3)
    print("rms_list:", rms_list)
    reference_list = [0.866, 0.927, 0.996, 0.714]

    for rms_value, ref in zip(rms_list, reference_list):
        assert abs(rms_value - ref) < 0.01


def test_wavaugmentate_greeting():
    """
    Test function to verify the functionality of the greeting option in the
    command line interface.

    This function runs the command with the greeting option and asserts that
    the output matches the application info.

    Args:
        None

    Returns:
        None
    """

    cmd = [ctf.PROG_NAME]
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    assert res.stdout == wau.application_info + "\n"


def test_wavaugmentate_info_option():
    """
    Test function to verify the functionality of the `info` option in the
    command line interface.

    This function runs the command with the `info` option and asserts that
    the output matches the application info.

    Args:
        None

    Returns:
        None
    """
    cmd = [ctf.PROG_NAME]
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    assert res.stdout == wau.application_info + "\n"


def test_wavaugmentate_amplitude_option():
    """
    Test function to verify the functionality of the `amplitude` option in the
    command line interface.

    This function runs the command with the `amplitude` option and asserts that
    the output matches the expected output. It also checks that the output file
    exists and has the correct shape and RMS values.

    Args:
        None

    Returns:
        None
    """

    cmd = [
        ctf.PROG_NAME,
        "-i",
        ctf.TEST_SOUND_1_FILE,
        "-o",
        ctf.OUTPUT_FILE,
        "-a",
        "0.5, 0.6, 0.7, 0.1",
    ]
    print("\n", " ".join(cmd))
    if os.path.exists(ctf.OUTPUT_FILE):
        os.remove(ctf.OUTPUT_FILE)
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    responce_string = str(res.stdout)
    out = ctf.shrink(responce_string)
    print("out:", out)
    full_ref = f"\namplitudes: [0.5, 0.6, 0.7, 0.1]\n{ms.SUCCESS_MARK}\n"
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(ctf.OUTPUT_FILE)
    written = Mcs()
    written.read(ctf.OUTPUT_FILE)
    for channel in written.data:
        assert channel.shape[0] == 220500
    rms_list = written.rms(decimals=3)
    print("rms_list:", rms_list)
    reference_list = [0.354, 0.424, 0.495, 0.071]
    for rms_value, ref in zip(rms_list, reference_list):
        assert abs(rms_value - ref) < ctf.ABS_ERR


def test_wavaugmentate_amplitude_option_fail_case1():
    """
    Test function to verify the functionality of the `amplitude` option in the
    command line interface when a non-numeric value is provided in the
    amplitude list.

    This function runs the command with the `amplitude` option and asserts that
    the output matches the expected error message. It checks that the output
    file does not exist.

    Args:
        None

    Returns:
        None
    """

    cmd = [
        ctf.PROG_NAME,
        "-i",
        ctf.TEST_SOUND_1_FILE,
        "-o",
        ctf.OUTPUT_FILE,
        "-a",
        "0.1, abc, 0.3, 0.4",
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    responce_string = str(res.stdout)
    out = ctf.shrink(responce_string)
    full_ref = f"{ms.ERROR_MARK}Amplitude list contains non number element:"
    full_ref += " < abc>."
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    print("out:", out)
    assert out == ref


def test_wavaugmentate_amplitude_option_fail_case2():
    """
    Test function to verify the functionality of the `amplitude` option in the
    command line interface when the amplitude list length does not match the
    number of channels.

    This function runs the command with the `amplitude` option and asserts that
    the output matches the expected error message.

    Args:
        None

    Returns:
        None
    """

    cmd = [
        ctf.PROG_NAME,
        "-i",
        ctf.TEST_SOUND_1_FILE,
        "-o",
        ctf.OUTPUT_FILE,
        "-a",
        "0.1, 0.3, 0.4",
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    responce_string = str(res.stdout)
    out = ctf.shrink(responce_string)
    print("out:", out)
    full_ref = f"\namplitudes: [0.1, 0.3, 0.4]\n\
    {ms.ERROR_MARK}Amplitude list length <3> does not match number of\n\
      channels. It should have <4> elements.\n"
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref


def test_wavaugmentate_delay_option():
    """
    Test function to verify the functionality of the `delay` option in the
    command line interface.

    This function runs the command with the `delay` option and asserts that
    the output matches the expected output. It also checks that the output
    file exists and has the correct shape and RMS values.

    Args:
        None

    Returns:
        None
    """

    cmd = [
        ctf.PROG_NAME,
        "-i",
        ctf.TEST_SOUND_1_FILE,
        "-o",
        ctf.OUTPUT_FILE,
        "-d",
        "100, 200, 300, 0",
    ]
    print("\n", " ".join(cmd))
    if os.path.exists(ctf.OUTPUT_FILE):
        os.remove(ctf.OUTPUT_FILE)
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    responce_string = str(res.stdout)
    out = ctf.shrink(responce_string)
    print("out:", out)
    full_ref = f"\ndelays: [100, 200, 300, 0]\n{ms.SUCCESS_MARK}\n"
    assert res.stdout == full_ref
    assert os.path.exists(ctf.OUTPUT_FILE)
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(ctf.OUTPUT_FILE)
    written = Mcs()
    written.read(ctf.OUTPUT_FILE)
    for channel in written.data:
        assert channel.shape[0] == 220513
    rms_list = written.rms(decimals=3)
    print("rms_list:", rms_list)
    reference_list = [0.707, 0.707, 0.707, 0.707]
    for rms_value, ref in zip(rms_list, reference_list):
        assert abs(rms_value - ref) < ctf.ABS_ERR


def test_wavaugmentate_delay_option_fail_case1():
    """
    Test function to verify the functionality of the `delay` option in the
    command line interface when a non-integer value is provided in the
    delays list.

    This function runs the command with the `delay` option and asserts that
    the output matches the expected error message.

    Args:
        None

    Returns:
        None
    """

    cmd = [
        ctf.PROG_NAME,
        "-i",
        ctf.TEST_SOUND_1_FILE,
        "-o",
        ctf.OUTPUT_FILE,
        "-d",
        "100, 389.1, 999, 456",
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    responce_string = str(res.stdout)
    out = ctf.shrink(responce_string)
    print("out:", out)
    full_ref = f"{ms.ERROR_MARK}Delays list contains non integer element:"
    full_ref += " <389.1>.\n"
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref


def test_wavaugmentate_delay_option_fail_case2():
    """
    Test function to verify the functionality of the `delay` option in the
    command line interface when the delays list length does not match the
    number of channels.

    This function runs the command with the `delay` option and asserts that
    the output matches the expected error message.

    Args:
        None

    Returns:
        None
    """

    cmd = [
        ctf.PROG_NAME,
        "-i",
        ctf.TEST_SOUND_1_FILE,
        "-o",
        ctf.OUTPUT_FILE,
        "-d",
        "100, 200, 300",
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    responce_string = str(res.stdout)
    out = ctf.shrink(responce_string)
    print("out:", out)
    full_ref = f"\ndelays: [100, 200, 300]\n\
{ms.ERROR_MARK}Delays list length <3> does not match number of\
 channels. It should have <4> elements.\n"
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref


def test_chain_option():
    """
    Test function to verify the functionality of the `-c` option in the command
    line interface.

    This function generates a multichannel sound using the `gen` function from
    the `wavaugmentate` module with the given frequency list, number of
    repetitions, and sample rate. It then applies amplitude control to the
    generated sound using the `amp` function from the `wavaugmentate` module
    with the given amplitude list. The generated sound is written to a file
    using the `wr` function from the `wavaugmentate` module with the given file
    path and sample rate.

    This function runs the command with the `-c` option and asserts that the
    output matches the expected output. It also checks that the output file
    exists and has the correct shape and RMS values.

    Args:
        None

    Returns:
        None
    """

    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    cmd = [
        ctf.PROG_NAME,
        "-c",
        'gen([100,250,100], 3, 44100).amp([0.1, 0.2, 0.3]).get().wr("'
        + ctf.TEST_SOUND_1_FILE
        + '")',
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    responce_string = str(res.stdout)
    print('responce_string:', responce_string)
    out = ctf.shrink(responce_string)
    full_ref = (
        'chain:gen([100,250,100],3,44100).amp([0.1,0.2,0.3]).get().wr("'
        + ctf.TEST_SOUND_1_FILE
        + '")\n'
        + f"{ms.SUCCESS_MARK}\n"
    )
    ref = ctf.shrink(full_ref)
    print("out:", out)
    print("ref:", ref)
    assert out == ref
    exists = os.path.exists(ctf.TEST_SOUND_1_FILE)
    assert exists is True
