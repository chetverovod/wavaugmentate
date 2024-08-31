import numpy as np
import os
import subprocess as sp
import wavaugmentate as wau

fs = wau.def_fs
t = 5
f_list = [400, 1000, 2333, 3700]  # Frequencies list.
frm_list = [60, 140, 230, 300]  # Speech formants list.

# Output files names.
test_sound_1_file = "./test_sounds/test_sound_1.wav"
test_sound_1_ac_file = "./test_sounds/test_sound_1_ac.wav"
test_sound_1_delay_file = "./test_sounds/test_sound_1_delay.wav"
test_sound_1_echo_file = "./test_sounds/test_sound_1_echo.wav"
test_sound_1_noise_file = "./test_sounds/test_sound_1_noise.wav"

output_file = "./outputwav/out.wav"
prog = "./" + wau.prog_name + ".py"


def shrink(s: str):
    """Drops white spaces, newlines, and tabs from a string."""

    subst_table = str.maketrans(
        {" ": None, "\n": None, "\t": None, "\r": None}
    )
    return s.translate(subst_table)


def test_generate_sine():
    """
    Test function to verify the shape of the multichannel sound generated by
    the `generate` function.

    This function calls the `generate` function from the `ma` module with
    the given `f_list`, `t`, and `fs` parameters.  It then asserts that the
    shape of the generated `test_sound_1` is equal to `(4, 220500)`.

    Parameters:
        None

    Returns:
        None
    """
    test_sound_1 = wau.generate(f_list, t, fs)
    wau.write(test_sound_1_file, test_sound_1, fs)
    assert test_sound_1.shape == (4, 220500)
    rms_list = wau.rms(test_sound_1, decimals=3)
    for r in rms_list:
        assert abs(r - 0.707) < 0.001


def test_generate_speech():
    """

    Parameters:
        None

    Returns:
        None
    """
    test_sound_1 = wau.generate(frm_list, t, fs, mode="speech", seed=42)
    wau.write(test_sound_1_file, test_sound_1, fs)
    assert test_sound_1.shape == (4, 220500)
    rms_list = wau.rms(test_sound_1, decimals=3)
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

    Parameters:
        None

    Returns:
        None
    """
    if os.path.exists(test_sound_1_file):
        os.remove(test_sound_1_file)
    test_sound_1 = wau.generate(f_list, t, fs)
    wau.write(test_sound_1_file, test_sound_1, fs)
    exists = os.path.exists(test_sound_1_file)
    assert exists is True


def test_read():
    """
    Test function to verify the functionality of the `read` function.

    This function calls the `read` function from the `ma` module with the
    given `test_sound_1_file` parameter.  It then asserts that the sample rate
    of the read sound is equal to `fs` and the shape of the read multichannel
    sound is equal to `(4, 220500)`.

    Parameters:
        None

    Returns:
        None
    """
    test_sound_1 = wau.generate(f_list, t, fs)
    test_rs, test_mcs = wau.read(test_sound_1_file)
    assert test_rs == fs
    assert test_mcs.shape == (4, 220500)
    assert np.array_equal(test_mcs, test_sound_1)


def test_file_info():
    """
    Test function to verify the functionality of the `file_info` function.

    This function calls the `file_info` function from the `ma` module with
    the given `test_sound_1_file` parameter.  It then asserts that the path,
    channels count, sample rate, and length of the file are correct.

    Parameters:
        None

    Returns:
        None
    """

    info = wau.file_info(test_sound_1_file)
    assert info["path"] == test_sound_1_file
    assert info["channels_count"] == 4
    assert info["sample_rate"] == wau.def_fs
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

    Parameters:
        None

    Returns:
        None
    """

    test_sound_1 = wau.generate(f_list, t, fs)
    amplitude_list = [0.1, 0.2, 0.3, 0.4]
    test_ac = wau.amplitude_ctrl(test_sound_1, amplitude_list)
    assert test_ac.shape == (4, 220500)
    wau.write(test_sound_1_ac_file, test_ac, fs)

    for a, sig, coef in zip(test_ac, test_sound_1, amplitude_list):
        assert np.array_equal(a, sig * coef)


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

    Parameters:
        None

    Returns:
        None
    """

    test_sound_1 = wau.generate(f_list, t, fs)
    delay_list = [100, 200, 300, 0]
    test_dc = wau.delay_ctrl(test_sound_1, delay_list)
    assert test_dc.shape == (4, 220513)
    wau.write(test_sound_1_delay_file, test_dc, fs)
    for ch in test_dc:
        assert ch.shape[0] == 220513
    rms_list = wau.rms(test_dc, last_index=24, decimals=3)
    reference_list = [0.511, 0.627, 0.445, 0.705]
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

    test_sound_1 = wau.generate(f_list, t, fs)
    delay_list = [1e6, 2e6, 3e6, 0]
    amplitude_list = [-0.3, -0.4, -0.5, 0]
    test_ec = wau.echo_ctrl(test_sound_1, delay_list, amplitude_list, fs)
    wau.write(test_sound_1_echo_file, test_ec, fs)
    rms_list = wau.rms(test_ec, decimals=3)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_echo_ctrl_option():
    if os.path.exists(test_sound_1_file):
        os.remove(test_sound_1_file)
    test_sound_1 = wau.generate(f_list, t, fs)
    wau.write(test_sound_1_file, test_sound_1, fs)

    cmd = [
        prog,
        "-i",
        test_sound_1_file,
        "-o",
        output_file,
        "-e",
        "100, 300, 400, 500 / 0.5, 0.6, 0.7, 0.1 ",
    ]
    print("\n", " ".join(cmd))
    if os.path.exists(output_file):
        os.remove(output_file)
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = shrink(s)
    print("out:", out)
    full_ref = (
        "\ndelays: [100, 300, 400, 500]\n"
        + f"\namplitudes: [0.5, 0.6, 0.7, 0.1]\n{wau.success_mark}\n"
    )
    ref = shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(output_file)
    _, written_data = wau.read(output_file)
    for ch in written_data:
        assert ch.shape[0] == 220522
    rms_list = wau.rms(written_data, decimals=3)
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

    Parameters:
        None

    Returns:
        None
    """

    test_sound_1 = wau.generate(f_list, t, fs)
    test_nc = wau.noise_ctrl(test_sound_1, [1, 0.2, 0.3, 0], fs, seed=42)
    wau.write(test_sound_1_noise_file, test_nc, fs)
    rms_list = wau.rms(test_nc, decimals=3)
    reference_list = [1.224, 0.735, 0.769, 0.707]

    for r, ref in zip(rms_list, reference_list):
        # Threshold increased, because noise is not repeatable with fixed seed.
        assert abs(r - ref) < 0.01


def test_wavaugmentate_noise_option():
    if os.path.exists(test_sound_1_file):
        os.remove(test_sound_1_file)
    test_sound_1 = wau.generate(f_list, t, fs)
    wau.write(test_sound_1_file, test_sound_1, fs)

    cmd = [
        prog,
        "-i",
        test_sound_1_file,
        "-o",
        output_file,
        "-n",
        "0.5, 0.6, 0.7, 0.1",
    ]
    print("\n", " ".join(cmd))
    if os.path.exists(output_file):
        os.remove(output_file)
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = shrink(s)
    print("out:", out)
    full_ref = f"\nnoise levels: [0.5, 0.6, 0.7, 0.1]\n{wau.success_mark}\n"
    ref = shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(output_file)
    _, written_data = wau.read(output_file)
    for ch in written_data:
        assert ch.shape[0] == 220500
    rms_list = wau.rms(written_data, decimals=3)
    print("rms_list:", rms_list)
    reference_list = [0.866, 0.927, 0.996, 0.714]

    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.01


def test_wavaugmentate_greeting():
    cmd = [prog]
    res = sp.run(cmd, capture_output=True, text=True)
    assert res.stdout == wau.application_info + "\n"


def test_wavaugmentate_info_option():
    cmd = [prog]
    res = sp.run(cmd, capture_output=True, text=True)
    assert res.stdout == wau.application_info + "\n"


def test_wavaugmentate_amplitude_option():
    cmd = [
        prog,
        "-i",
        test_sound_1_file,
        "-o",
        output_file,
        "-a",
        "0.5, 0.6, 0.7, 0.1",
    ]
    print("\n", " ".join(cmd))
    if os.path.exists(output_file):
        os.remove(output_file)
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = shrink(s)
    print("out:", out)
    full_ref = "\namplitudes: [0.5, 0.6, 0.7, 0.1]\nDone.\n"
    ref = shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(output_file)
    _, written_data = wau.read(output_file)
    for ch in written_data:
        assert ch.shape[0] == 220500
    rms_list = wau.rms(written_data, decimals=3)
    print("rms_list:", rms_list)
    reference_list = [0.354, 0.424, 0.495, 0.071]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_wavaugmentate_amplitude_option_fail_case1():
    cmd = [
        prog,
        "-i",
        test_sound_1_file,
        "-o",
        output_file,
        "-a",
        "0.1, abc, 0.3, 0.4",
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = shrink(s)
    print("out:", out)
    full_ref = f"{wau.error_mark}Amplitude list contains non number element: < abc>."
    ref = shrink(full_ref)
    print("ref:", ref)
    assert out == ref


def test_wavaugmentate_amplitude_option_fail_case2():
    cmd = [
        prog,
        "-i",
        test_sound_1_file,
        "-o",
        output_file,
        "-a",
        "0.1, 0.3, 0.4",
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = shrink(s)
    print("out:", out)
    full_ref = f"\namplitudes: [0.1, 0.3, 0.4]\n\
    {wau.error_mark}Amplitude list length <3> does not match number of\n\
      channels. It should have <4> elements.\n"
    ref = shrink(full_ref)
    print("ref:", ref)
    assert out == ref


def test_wavaugmentate_delay_option():
    cmd = [
        prog,
        "-i",
        test_sound_1_file,
        "-o",
        output_file,
        "-d",
        "100, 200, 300, 0",
    ]
    print("\n", " ".join(cmd))
    if os.path.exists(output_file):
        os.remove(output_file)
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = shrink(s)
    print("out:", out)
    full_ref = "\ndelays: [100, 200, 300, 0]\nDone.\n"
    assert res.stdout == full_ref
    assert os.path.exists(output_file)
    ref = shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    assert os.path.exists(output_file)
    _, written_data = wau.read(output_file)
    for ch in written_data:
        assert ch.shape[0] == 220513
    rms_list = wau.rms(written_data, decimals=3)
    print("rms_list:", rms_list)
    reference_list = [0.707, 0.707, 0.707, 0.707]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_wavaugmentate_delay_option_fail_case1():
    cmd = [
        prog,
        "-i",
        test_sound_1_file,
        "-o",
        output_file,
        "-d",
        "100, 389.1, 999, 456",
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = shrink(s)
    print("out:", out)
    full_ref = f"{wau.error_mark}Delays list contains non integer element: <389.1>.\n"
    ref = shrink(full_ref)
    print("ref:", ref)
    assert out == ref


def test_wavaugmentate_delay_option_fail_case2():
    cmd = [
        prog,
        "-i",
        test_sound_1_file,
        "-o",
        output_file,
        "-d",
        "100, 200, 300",
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = shrink(s)
    print("out:", out)
    full_ref = f"\ndelays: [100, 200, 300]\n\
{wau.error_mark}Delays list length <3> does not match number of\
 channels. It should have <4> elements.\n"
    ref = shrink(full_ref)
    print("ref:", ref)
    assert out == ref


def test_WaChain_controls():
    test_sound_1 = wau.generate(f_list, t, fs)
    w = wau.WaChain(test_sound_1)
    print(w.data)

    w.amp([0.1, 0.3, 0.4, 1]).dly([100, 200, 300, 0])
    res1 = w.data

    w.put(np.zeros(1))
    res2 = (
        w.put(test_sound_1)
        .amp([0.1, 0.3, 0.4, 1])
        .dly([100, 200, 300, 0])
        .get()
    )
    print(res2)
    assert np.array_equal(res1, res2)


def test_WaChain_wr_rd():
    w = wau.WaChain()
    if os.path.exists(test_sound_1_file):
        os.remove(test_sound_1_file)
    w.gen(f_list, t, fs).wr(test_sound_1_file)

    r = wau.WaChain()
    r.rd(test_sound_1_file)

    assert np.array_equal(w.data, r.data)


def test_WaChain_echo():
    d_list = [1e6, 2e6, 3e6, 0]
    a_list = [-0.3, -0.4, -0.5, 0]
    w = wau.WaChain()
    w.gen(f_list, t, fs).echo(d_list, a_list)
    rms_list = w.rms(decimals=3)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_WaChain_noise():
    n_list = [1, 0.2, 0.3, 0]

    w = wau.WaChain()
    w.gen(f_list, t, fs).ns(n_list, seed=42)
    rms_list = w.rms(decimals=3)
    reference_list = [1.224, 0.735, 0.769, 0.707]
    for r, ref in zip(rms_list, reference_list):
        # Threshold increased, because noise is not repeatable with fixed seed.
        assert abs(r - ref) < 0.01


def test_WaChain_info():
    w = wau.WaChain()
    if os.path.exists(test_sound_1_file):
        os.remove(test_sound_1_file)
    w.gen(f_list, t, fs).wr(test_sound_1_file)
    print(w.info())

    ref = {
        "path": "",
        "channels_count": 4,
        "sample_rate": 44100,
        "length_s": 5.0,
    }
    assert w.info() == ref


def test_WaChain_chain_class():
    w = wau.WaChain()
    cmd_prefix = "w."
    cmd = "gen(f_list, t, fs).rms()"
    s = str(eval(cmd_prefix + cmd.strip()))
    out = shrink(s)
    ref = "[0.70710844,0.7071083,0.707108,0.70710754]"

    print(out)
    assert out == ref


def test_chain_option():
    if os.path.exists(test_sound_1_file):
        os.remove(test_sound_1_file)
    cmd = [
        prog,
        "-c",
        'gen([100,250,100], 3, 44100).amp([0.1, 0.2, 0.3]).wr("'
        + test_sound_1_file
        + '")',
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = shrink(s)
    print("out:", out)
    full_ref = (
        'chain:gen([100,250,100],3,44100).amp([0.1,0.2,0.3]).wr("'
        + test_sound_1_file
        + '")\nDone.\n'
    )
    ref = shrink(full_ref)
    print("ref:", ref)
    assert out == ref
    exists = os.path.exists(test_sound_1_file)
    assert exists is True


def test_README_examples():
    # preparations
    fn = "./sound.wav"
    if os.path.exists(fn):
        os.remove(fn)

    test_sound_1 = wau.generate(f_list, t, fs)
    wau.write(fn, test_sound_1, fs)

    # examples code for  README.md

    # Read WAV-file to array.
    fsmp, mcs = wau.read("./sound.wav")

    # Apply delays.
    delay_list = [100, 200, 300, 400]  # Corresponds to channels quantity.
    d = wau.delay_ctrl(mcs, delay_list)

    # Apply amplitude changes.
    amplitude_list = [0.1, 0.2, 0.3, 0.4]  # Corresponds to channels quantity.
    res = wau.amplitude_ctrl(d, amplitude_list)

    # Augmentation result saving.
    wau.write("./sound_delayed.wav", res, fsmp)

    # The same code in OOP approach:

    w = wau.WaChain()
    w.rd("./sound.wav").dly([100, 200, 300, 400]).amp([0.1, 0.2, 0.3, 0.4]).wr(
        "./sound_delayed.wav"
    )


def test_sum():
    test_sound_1 = wau.generate([100], t, fs)
    test_sound_2 = wau.generate([300], t, fs)
    res = wau.sum(test_sound_1, test_sound_2)
    wau.write(
        test_sound_1_file,
        res,
        fs,
    )
    ref = [0.707, 0.707, 1.0]
    for s, ref_value in zip([test_sound_1, test_sound_2, res], ref):
        r = wau.rms(s, decimals=3)
        print(r)
        assert abs(r[0] - ref_value) < 0.001


def test_merge():
    test_sound_1 = wau.generate([100, 300], t, fs)
    res = wau.merge(test_sound_1)
    wau.write(
        test_sound_1_file,
        res,
        fs,
    )
    ref_value = 1.0
    r = wau.rms(res, decimals=3)
    print(r)
    assert abs(r[0] - ref_value) < 0.001


def test_split():
    test_sound_1 = wau.generate([300], t, fs)
    res = wau.split(test_sound_1, 5)
    wau.write(test_sound_1_file, res, fs)
    ref_value = 0.707
    # for i in range(0, test_sound_1.shape[0]):
    r = wau.rms(test_sound_1, decimals=3)
    print(r)
    for i in range(0, test_sound_1.shape[0]):
        assert abs(r[i] - ref_value) < 0.001


def test_chain_sum():
    w = wau.WaChain()
    res = wau.WaChain()
    w.gen([100], t, fs)
    res = w.copy()
    test_sound_2 = wau.generate([300], t, fs)
    res.sum(test_sound_2).wr(test_sound_1_file)
    ref = [0.707, 0.707, 1.0]
    for s, ref_value in zip([w.data, test_sound_2, res.data], ref):
        r = wau.rms(s, decimals=3)
        print(r)
        assert abs(r[0] - ref_value) < 0.001


def test_chain_merge():
    w = wau.WaChain()
    r = w.gen([100, 300], t, fs).mrg().wr(test_sound_1_file).rms(decimals=3)
    print(r)
    ref_value = 1.0
    assert abs(r[0] - ref_value) < 0.001


def test_chain_split():
    w = wau.WaChain()
    w.gen([300], t, fs).splt(5).wr(test_sound_1_file)
    c = w.data.shape[0]
    assert c == 5
    ref_value = 0.707
    r = w.rms(decimals=3)
    print(r)
    for i in range(0, c):
        assert abs(r[0] - ref_value) < 0.001


def test_chain_side_by_side():
    test_sound_1 = wau.generate([300], t, fs)
    w = wau.WaChain()
    r = (
        w.gen([1000], t, fs)
        .amp([0.3])
        .sbs(test_sound_1)
        .wr(test_sound_1_file)
        .rms(decimals=3)
    )
    print(r)
    ref_value = [0.212, 0.707]
    for r, ref in zip(r, ref_value):
        print(r)
        assert abs(r - ref) < 0.001


def test_side_by_side():
    test_sound_1 = wau.generate([100], t, fs)
    test_sound_1 = wau.amplitude_ctrl(test_sound_1, [0.3])
    test_sound_2 = wau.generate([300], t, fs)
    res = wau.side_by_side(test_sound_1, test_sound_2)
    wau.write(test_sound_1_file, res, fs)
    ref_list = [0.212, 0.707]
    r = wau.rms(res, decimals=3)
    for r, ref in zip(r, ref_list):
        print(r)
        assert abs(r - ref) < 0.001


def test_pause_detect():
    test_sound_1 = wau.generate([100, 400], t, fs)
    mask = wau.pause_detect(test_sound_1, [0.5, 0.3])
    res = wau.side_by_side(test_sound_1, mask)
    print(res)
    wau.write(test_sound_1_file, res, fs)
    r = wau.rms(res, decimals=3)
    ref_list = [0.707, 0.707, 0.865, 0.923]
    for r, ref in zip(r, ref_list):
        print(r)
        assert abs(r - ref) < 0.001


def test_chain_pause_detect():
    w = wau.WaChain()
    w1 = wau.WaChain()
    w.gen([100, 400], t, fs)
    w1 = w.copy()
    w.pdt([0.5, 0.3])
    w1.sbs(w.data).wr(test_sound_1_file)
    r = w1.rms(decimals=3)
    ref = [0.707, 0.707, 0.865, 0.923]
    for i in range(0, len(r)):
        print(r[i])
        assert abs(r[i] - ref[i]) < 0.001


def test_pause_shrink_sine():
    test_sound_1 = wau.generate([100, 400], t, fs)
    mask = wau.pause_detect(test_sound_1, [0.5, 0.3])
    res = wau.side_by_side(test_sound_1, mask)
    print(res)
    res = wau.pause_shrink(test_sound_1, mask, [20, 4])
    wau.write(test_sound_1_file, res, fs)
    r = wau.rms(res, decimals=3)
    ref = [0.702, 0.706, 0.865, 0.923]
    for r, ref in zip(r, ref):
        print(r)
        assert abs(r - ref) < 0.001


def test_pause_shrink_speech():
    test_sound_1 = wau.generate([100, 300], t, fs, mode="speech", seed=42)
    mask = wau.pause_detect(test_sound_1, [0.5, 0.3])
    res = wau.side_by_side(test_sound_1, mask)
    res = wau.pause_shrink(test_sound_1, mask, [20, 4])
    wau.write(test_sound_1_file, res, fs)
    r = wau.rms(res, decimals=3)
    ref = [0.331, 0.324]
    for r, ref in zip(r, ref):
        print(r)
        assert abs(r - ref) < 0.001


def test_pause_measure():
    test_sound_1 = wau.generate([100, 300], 0.003, fs, mode="speech", seed=42)
    mask = wau.pause_detect(test_sound_1, [0.5, 0.3])
    res_list = wau.pause_measure(mask)
    print(res_list)

    # res = wau.pause_shrink(test_sound_1, mask, [20, 4])
    # wau.write(test_sound_1_file, res, fs)
    ref_list = [
        [
            (0, 2),
            (31, 4),
            (37, 5),
            (47, 5),
            (56, 4),
            (70, 10),
            (86, 5),
            (97, 15),
            (117, 7),
        ],
        [
            (0, 1),
            (16, 3),
            (45, 2),
            (53, 2),
            (66, 2),
            (73, 2),
            (79, 4),
            (88, 5),
            (98, 1),
            (114, 4),
        ],
    ]

    for res, ref in zip(res_list, ref_list):
        print(res)
        assert res == ref


def test_pause_set():
    test_sound_1 = wau.generate([100, 300], 0.003, fs, mode="speech", seed=42)
    mask = wau.pause_detect(test_sound_1, [0.5, 0.3])
    pause_list = wau.pause_measure(mask)
    res = wau.pause_set(test_sound_1, pause_list, [10, 150])
    assert res.shape == (2, 1618)
    print("res shape:", res.shape)
    print("res:", type(res[0, 1]))
    wau.write(test_sound_1_file, res, fs)
    r = wau.rms(res, decimals=3)
    ref = [0.105, 0.113]
    for r, ref in zip(r, ref):
        print(r)
        assert abs(r - ref) < 0.001
