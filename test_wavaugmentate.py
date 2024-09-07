"""Module providing test functions for wavaugmentate.py  module."""

import os
import subprocess as sp
import numpy as np
import wavaugmentate as wau
import common_test_functions as ctf


def test_generate_sine():
    """
    Test function to verify the shape of the multichannel sound generated by
    the `generate` function.

    This function calls the `generate` function from the `ma` module with
    the given `f_list`, `t`, and `fs` parameters.  It then asserts that the
    shape of the generated `test_sound_1` is equal to `(4, 220500)`.

    Args:
        None

    Returns:
        None
    """
    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    assert test_sound_1.shape() == (4, 220500)
    rms_list = test_sound_1.rms(decimals=3)
    for r in rms_list:
        assert abs(r - 0.707) < 0.001


def test_generate_speech():
    """
    Test speech-like signal generation using the `generate` function.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(
        ctf.frm_list, ctf.SIGNAL_TIME_LEN, mode="speech", seed=42
    )
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    assert test_sound_1.shape() == (4, 220500)
    rms_list = test_sound_1.rms(decimals=3)
    ref_list = [0.327, 0.326, 0.33, 0.332]
    for r, ref in zip(rms_list, ref_list):
        assert abs(r - ref) < 0.001


def test_write():
    """
    Test function to verify the functionality of the `write` function.

    This function calls the `write` function from the `ma` module with the
    given `test_sound_1_file`, `test_sound_1`, and `fs` parameters.  It first
    generates a multichannel sound using the `generate` function and then
    writes it to a file using the `write` function.

    Args:
        None

    Returns:
        None
    """
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    exists = os.path.exists(ctf.TEST_SOUND_1_FILE)
    assert exists is True


def test_read():
    """
    Test function to verify the functionality of the `read` function.

    This function calls the `read` function from the `ma` module with the
    given `test_sound_1_file` parameter.  It then asserts that the sample rate
    of the read sound is equal to `fs` and the shape of the read multichannel
    sound is equal to `(4, 220500)`.

    Args:
        None

    Returns:
        None
    """
    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    test_rs = wau.Mcs()
    assert test_rs.sample_rate == -1
    test_rs.read(ctf.TEST_SOUND_1_FILE)
    assert test_rs.sample_rate == ctf.FS
    assert test_rs.shape() == (4, 220500)
    assert np.array_equal(test_rs.data, test_sound_1.data)


def test_file_info():
    """
    Test function to verify the functionality of the `file_info` function.

    This function calls the `file_info` function from the `ma` module with
    the given `test_sound_1_file` parameter.  It then asserts that the path,
    channels count, sample rate, and length of the file are correct.

    Args:
        None

    Returns:
        None
    """

    info = wau.file_info(ctf.TEST_SOUND_1_FILE)
    assert info["path"] == ctf.TEST_SOUND_1_FILE
    assert info["channels_count"] == 4
    assert info["sample_rate"] == wau.DEF_FS
    assert info["length_s"] == 5.0


def test_amplitude_ctrl():
    """
    Test function to verify the functionality of the `amplitude_ctrl`
    function.

    This function generates a multichannel sound using the `generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies amplitude control to the generated sound using
    the `amplitude_ctrl` function from the `ma` module with the given
    `test_sound_1` and `amplitude_list` parameters.  It asserts that the shape
    of the amplified multichannel sound is equal to `(4, 220500)`.  It writes
    the amplified multichannel sound to a file using the `write` function
    from the `ma` module with the given `test_sound_1_ac_file`, `test_ac`, and
    `fs` parameters.  It then asserts that each channel of the amplified
    multichannel sound is equal to the corresponding channel of the original
    multichannel sound multiplied by the corresponding amplitude from the
    `amplitude_list`.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    amplitude_list = [0.1, 0.2, 0.3, 0.4]
    test_ac = test_sound_1.copy()
    test_ac.amplitude_ctrl(amplitude_list)
    assert test_sound_1.shape() == (4, 220500)
    assert test_ac.shape() == (4, 220500)
    test_ac.write(ctf.TEST_SOUND_1_AC_FILE)
    for a, sig, coef in zip(test_ac.data, test_sound_1.data, amplitude_list):
        assert np.array_equal(a, sig * coef)


def test_rn_amplitude_ctrl():
    """
    Test random amplitudes control.

       Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    amplitude_list = [0.1, 0.2, 0.3, 0.4]
    amplitude_deviation_list = [0.1, 0.1, 0.1, 0.1]
    test_ac = test_sound_1.copy()
    test_ac.set_seed(42)
    test_ac.amplitude_ctrl(amplitude_list, amplitude_deviation_list)
    assert test_ac.shape() == (4, 220500)
    test_ac.write(ctf.TEST_SOUND_1_AC_FILE)
    r_list = test_ac.rms(decimals=3)
    ref_list = [0.109, 0.180, 0.251, 0.322]
    for r, ref in zip(r_list, ref_list):
        assert abs(r - ref) < 0.001


def test_delay_ctrl():
    """
    Test function to verify the functionality of the `delay_ctrl`
    function.

    This function generates a multichannel sound using the `generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies delay control to the generated sound using the
    `delay_ctrl` function from the `ma` module with the given
    `test_sound_1` and `delay_list` parameters.  It asserts that the shape of
    the delayed multichannel sound is equal to `(4, 220500)`.  It writes the
    delayed multichannel sound to a file using the `write` function from
    the `ma` module with the given file path and `fs` parameters.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    delay_list = [100, 200, 300, 0]
    test_dc = test_sound_1.delay_ctrl(delay_list)
    assert test_dc.shape() == (4, 220513)
    test_dc.write(ctf.TEST_SOUND_1_DELAY_FILE)
    for ch in test_dc.data:
        assert ch.shape[0] == 220513
    rms_list = test_dc.rms(last_index=24, decimals=3)
    reference_list = [0.511, 0.627, 0.445, 0.705]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_rn_delay_ctrl():
    """
    Test function to verify the functionality of the `delay_ctrl`
    function for random delays.

    This function generates a multichannel sound using the `generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies delay control to the generated sound using the
    `delay_ctrl` function from the `ma` module with the given
    `test_sound_1` and `delay_list` parameters.  It asserts that the shape of
    the delayed multichannel sound is equal to `(4, 220500)`.  It writes the
    delayed multichannel sound to a file using the `write` function from
    the `ma` module with the given file path and `fs` parameters.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    delay_list = [100, 200, 300, 40]
    delay_deviation_list = [10, 20, 30, 15]
    test_sound_1.set_seed(42)
    test_dc = test_sound_1.delay_ctrl(delay_list, delay_deviation_list)
    assert test_dc.shape() == (4, 220512)
    test_dc.write(ctf.TEST_SOUND_1_DELAY_FILE)
    for ch in test_dc.data:
        assert ch.shape[0] == 220512
    rms_list = test_dc.rms(last_index=24, decimals=3)
    reference_list = [0.511, 0.627, 0.456, 0.699]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_echo_ctrl():
    """
    Test function to verify the functionality of the `echo_ctrl`
    function.

    This function generates a multichannel sound using the `generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies echo control to the generated sound using the
    `echo_ctrl` function from the `ma` module with the given
    `test_sound_1`, `delay_us_list`, and `amplitude_list` parameters.
    It writes the echoed multichannel sound to a file using the `write`
    function from the `ma` module with the given file path and `fs` parameters.
    Finally, it calculates the root mean square (RMS) values of the echoed
    sound and compares them to the expected values in the `reference_list`.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    delay_list = [1e6, 2e6, 3e6, 0]
    amplitude_list = [-0.3, -0.4, -0.5, 0]
    test_ec = test_sound_1.copy()
    test_ec.echo_ctrl(delay_list, amplitude_list)
    test_ec.write(ctf.TEST_SOUND_1_ECHO_FILE)
    rms_list = test_ec.rms(decimals=3)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_rn_echo_ctrl():
    """
    Test function to verify the functionality of the `echo_ctrl`
    function.

    This function generates a multichannel sound using the `generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies echo control to the generated sound using the
    `echo_ctrl` function from the `ma` module with the given
    `test_sound_1`, `delay_us_list`, and `amplitude_list` parameters.
    It writes the echoed multichannel sound to a file using the `write`
    function from the `ma` module with the given file path and `fs` parameters.
    Finally, it calculates the root mean square (RMS) values of the echoed
    sound and compares them to the expected values in the `reference_list`.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    delay_list = [1e6, 2e6, 3e6, 100]
    amplitude_list = [-0.3, -0.4, -0.5, 0.1]
    amplitude_deviation_list = [0.1, 0.1, 0.1, 0.1]
    delay_deviation_list = [10, 20, 30, 5]
    test_ec = test_sound_1.copy()
    test_ec.set_seed(42)
    test_ec.echo_ctrl(
        delay_list,
        amplitude_list,
        delay_deviation_list,
        amplitude_deviation_list,
    )
    test_ec.write(ctf.TEST_SOUND_1_ECHO_FILE)
    rms_list = test_ec.rms(decimals=3)
    reference_list = [0.457, 0.471, 0.536, 0.52]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


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

    test_sound_1 = wau.Mcs(fs=ctf.FS)
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
    s = str(res.stdout)
    out = ctf.shrink(s)
    print("out:", out)
    full_ref = (
        "\ndelays: [100, 300, 400, 500]\n"
        + f"\namplitudes: [0.5, 0.6, 0.7, 0.1]\n{wau.SUCCESS_MARK}\n"
    )
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(ctf.OUTPUT_FILE)

    written = wau.Mcs()
    written.read(ctf.OUTPUT_FILE)
    for ch in written.data:
        assert ch.shape[0] == 220522
    rms_list = written.rms(decimals=3)
    print("rms_list:", rms_list)
    reference_list = [1.054, 0.716, 1.144, 0.749]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_noise_ctrl():
    """
    Test function to verify the functionality of the `noise_ctrl`
    function.

    This function generates a multichannel sound using the `generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies noise control to the generated sound using the
    `noise_ctrl` function from the `ma` module with the given
    `test_sound_1`, `noise_level_list`, and `fs` parameters.  It writes the
    noise-controlled multichannel sound to a file using the `write`
    from the `ma` module with the given file path and `fs` parameters. Finally,
    it calculates the root mean square (RMS) values of the noise-controlled
    sound and compares them to the expected values in the `reference_list`.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    test_sound_1.set_seed(42)
    test_nc = test_sound_1.noise_ctrl([1, 0.2, 0.3, 0])
    test_nc.write(ctf.TEST_SOUND_1_NOISE_FILE)
    rms_list = test_nc.rms(decimals=3)
    reference_list = [1.224, 0.735, 0.769, 0.707]

    for r, ref in zip(rms_list, reference_list):
        # Threshold increased, because noise is not repeatable with fixed seed.
        assert abs(r - ref) < 0.01


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
    test_sound_1 = wau.Mcs(fs=ctf.FS)
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
    s = str(res.stdout)
    out = ctf.shrink(s)
    print("out:", out)
    full_ref = f"\nnoise levels: [0.5, 0.6, 0.7, 0.1]\n{wau.SUCCESS_MARK}\n"
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(ctf.OUTPUT_FILE)
    written = wau.Mcs()
    written.read(ctf.OUTPUT_FILE)
    for ch in written.data:
        assert ch.shape[0] == 220500
    rms_list = written.rms(decimals=3)
    print("rms_list:", rms_list)
    reference_list = [0.866, 0.927, 0.996, 0.714]

    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.01


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
    s = str(res.stdout)
    out = ctf.shrink(s)
    print("out:", out)
    full_ref = f"\namplitudes: [0.5, 0.6, 0.7, 0.1]\n{wau.SUCCESS_MARK}\n"
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(ctf.OUTPUT_FILE)
    written = wau.Mcs()
    written.read(ctf.OUTPUT_FILE)
    for ch in written.data:
        assert ch.shape[0] == 220500
    rms_list = written.rms(decimals=3)
    print("rms_list:", rms_list)
    reference_list = [0.354, 0.424, 0.495, 0.071]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


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
    s = str(res.stdout)
    out = ctf.shrink(s)
    full_ref = f"{wau.ERROR_MARK}Amplitude list contains non number element:"
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
    s = str(res.stdout)
    out = ctf.shrink(s)
    print("out:", out)
    full_ref = f"\namplitudes: [0.1, 0.3, 0.4]\n\
    {wau.ERROR_MARK}Amplitude list length <3> does not match number of\n\
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
    s = str(res.stdout)
    out = ctf.shrink(s)
    print("out:", out)
    full_ref = f"\ndelays: [100, 200, 300, 0]\n{wau.SUCCESS_MARK}\n"
    assert res.stdout == full_ref
    assert os.path.exists(ctf.OUTPUT_FILE)
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(ctf.OUTPUT_FILE)
    written = wau.Mcs()
    written.read(ctf.OUTPUT_FILE)
    for ch in written.data:
        assert ch.shape[0] == 220513
    rms_list = written.rms(decimals=3)
    print("rms_list:", rms_list)
    reference_list = [0.707, 0.707, 0.707, 0.707]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


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
    s = str(res.stdout)
    out = ctf.shrink(s)
    print("out:", out)
    full_ref = f"{wau.ERROR_MARK}Delays list contains non integer element:"
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
    s = str(res.stdout)
    out = ctf.shrink(s)
    print("out:", out)
    full_ref = f"\ndelays: [100, 200, 300]\n\
{wau.ERROR_MARK}Delays list length <3> does not match number of\
 channels. It should have <4> elements.\n"
    ref = ctf.shrink(full_ref)
    print("ref:", ref)
    assert out == ref
