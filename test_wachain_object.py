"""Module providing test functions for wavaugmentate.py  module."""

import os
import subprocess as sp
import numpy as np
import wavaugmentate as wau
import common_test_functions as ctf


def test_mcs_put():
    """
    Test function to verify the functionality of the WaChain class's put method.

    This function generates a multichannel sound using the generate function from
    the wau module with the given frequency list, time duration, and sample rate.
    It then applies the put method of the WaChain class to the generated sound
    and asserts that the shape and data of the original sound are equal to the
    shape and data of the sound after applying the put method.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)

    w = wau.Mcs()
    w.put(test_sound_1)

    assert np.array_equal(test_sound_1.data, w.data)
    assert np.array_equal(test_sound_1.data, w.get())

    w2 = wau.Mcs(test_sound_1.data)
    assert np.array_equal(w2.data, w.get())


def test_wachain_amp_control():
    """
    Test function to verify the functionality of the WaChain class.

    This function tests the amp method of the WaChain class by
    applying amplitude control to a generated multichannel sound.

    Args:
        None

    Returns:
        None
    """

    a_list = [0.1, 0.3, 0.4, 1]
    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    w = wau.Mcs(test_sound_1.data)

    w.amp(a_list)
    res1 = w
    print("res1 =", res1.data)

    d = wau.Mcs()

    res2 = d.put(test_sound_1).amp(a_list).get()

    print("res2 =", res2.data)
    assert np.array_equal(res1.get(), res2)


def test_wachain_dly_controls():
    """
    Test function to verify the functionality of the WaChain class.

    This function tests the dly method of the WaChain class by
    applying delay controls to a generated multichannel sound.

    Args:
        None

    Returns:
        None
    """

    d_list = [100, 200, 300, 0]
    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    w = test_sound_1.copy()

    w.dly(d_list)
    res1 = w
    print("res1 shape =", res1.data.shape)
    print("res1 =", res1.data)

    d = wau.Mcs()
    d.put(test_sound_1)
    res2 = d.dly(d_list)

    print("res2 shape =", res2.data.shape)
    print("res2 =", res2.data)
    assert res1.data.shape == res2.data.shape
    assert np.array_equal(res1.get(), res2.get())


def test_wachain_wr_rd():
    """
    Test function to verify the functionality of the WaChain class.

    This function tests the wr and rd methods of the WaChain class by
    generating a multichannel sound, writing it to a file, reading it back,
    and comparing the original and read sound data.

    Args:
        None

    Returns:
        None
    """

    w = wau.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    w.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)

    r = wau.Mcs()
    r.rd(ctf.TEST_SOUND_1_FILE)

    assert np.array_equal(w.data, r.data)


def test_wachain_echo():
    """
    Test function to verify the functionality of the `echo` method in the
    WaChain class.

    This function generates a multichannel sound using the `gen` method of the
    WaChain class with the given frequency list, duration, and sample rate. It
    then applies the `echo` method to the generated sound with the given delay
    list and amplitude list. Finally, it calculates the root mean square (RMS)
    values of the echoed sound and compares them to the expected values in the
    reference list.

    Args:
        None

    Returns:
        None
    """
    d_list = [1e6, 2e6, 3e6, 0]
    a_list = [-0.3, -0.4, -0.5, 0]
    w = wau.Mcs()
    w.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).echo(d_list, a_list)
    rms_list = w.rms(decimals=3)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001
    d_list = [1e6, 2e6, 3e6, 0]
    a_list = [-0.3, -0.4, -0.5, 0]
    w = wau.Mcs()
    w.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).echo(d_list, a_list)
    rms_list = w.rms(decimals=3)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_wachain_noise():
    """
    Test function to verify the functionality of the `ns` method in the
    `WaChain` class.

    This function generates a multichannel sound using the `gen` method of the
    `WaChain` class with the given frequency list, duration, and sample rate. It
    then applies the `ns` method to the generated sound with the given noise
    level list. Finally, it calculates the root mean square (RMS) values of the
    noise-controlled sound and compares them to the expected values in the
    reference list.

    Args:
        None

    Returns:
        None
    """

    n_list = [1, 0.2, 0.3, 0]

    w = wau.Mcs()
    w.set_seed(42)
    w.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).ns(n_list)
    rms_list = w.rms(decimals=3)
    reference_list = [1.224, 0.735, 0.769, 0.707]
    for r, ref in zip(rms_list, reference_list):
        # Threshold increased, because noise is not repeatable with fixed seed.
        assert abs(r - ref) < 0.01


def test_wachain_info():
    """
    Test the `info` method of the `WaChain` class.

    This function creates a `WaChain` object, generates a sound file with the
    given frequency list, duration, and sample rate, and writes it to a file.
    It then calls the `info` method of the `WaChain` object and prints the
    result.  Finally, it asserts that the returned dictionary matches the
    expected reference dictionary.

    Args:
        None

    Returns:
        None
    """

    w = wau.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    w.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)
    print(w.info())

    ref = {
        "path": "",
        "channels_count": 4,
        "sample_rate": 44100,
        "length_s": 5.0,
    }
    assert w.info() == ref


#  Test not finished.
def test_wachain_rn_rd():
    """Test augmentation on the fly."""

    w = wau.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    w.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)

    a = wau.Mcs()
    a.rd(ctf.TEST_SOUND_1_FILE)

    b = wau.Mcs()
    b.rd(ctf.TEST_SOUND_1_FILE)

    assert np.array_equal(w.data, a.data)
    assert np.array_equal(w.data, b.data)

    b.achn(["amp([1, 0.7, 0.5, 0.3])"])
    res = b.rdac(ctf.TEST_SOUND_1_FILE)
    print("res=", res.data[0])

    # !!! assert np.array_equal(w.data, b.data)


def test_wachain_rn_aug_rd():
    """Test augmentation on the fly."""

    w = wau.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    w.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)

    a = wau.Mcs()
    a.rd(ctf.TEST_SOUND_1_FILE)

    b = wau.Mcs()
    b.rd(ctf.TEST_SOUND_1_FILE)

    assert np.array_equal(w.data, a.data)
    assert np.array_equal(w.data, b.data)

    a.amp([1, 0.7, 0.5, 0.3])
    b.amp([1, 0.7, 0.5, 0.3])
    assert np.array_equal(a.data, b.data)

    a.set_seed(42)
    b.set_seed(42)
    a.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
    b.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
    assert np.array_equal(a.data, b.data)

    a.set_seed(-1)
    b.set_seed(-1)
    for _ in range(10):
        a.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
        b.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
        assert not np.array_equal(a.data, b.data)


def test_wachain_chain_class():
    """
    Tests the functionality of the WaChain class by generating a multichannel
    sound, computing its RMS values, and comparing them to the expected values.

    Args:
        None

    Returns:
        None
    """

    w = wau.Mcs()
    cmd_prefix = "w."
    cmd = "gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).rms()"
    s = str(eval(cmd_prefix + cmd.strip()))
    out = ctf.shrink(s)
    ref = "[0.70710844,0.7071083,0.707108,0.70710754]"

    print(out)
    w.info()
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
        'gen([100,250,100], 3, 44100).amp([0.1, 0.2, 0.3]).wr("'
        + ctf.TEST_SOUND_1_FILE
        + '")',
    ]
    print("\n", " ".join(cmd))
    res = sp.run(cmd, capture_output=True, text=True, check=False)
    s = str(res.stdout)
    out = ctf.shrink(s)
    full_ref = (
        'chain:gen([100,250,100],3,44100).amp([0.1,0.2,0.3]).wr("'
        + ctf.TEST_SOUND_1_FILE
        + '")\n'
        + f"{wau.SUCCESS_MARK}\n"
    )
    ref = ctf.shrink(full_ref)
    print("out:", out)
    print("ref:", ref)
    assert out == ref
    exists = os.path.exists(ctf.TEST_SOUND_1_FILE)
    assert exists is True


def test_readme_examples():
    """
    This function tests the functionality of examples for README file of the
    wavaugmentate module by generating a multichannel sound, applying various
    augmentations, and saving the results to WAV files. It also demonstrates
    the usage of the WaChain class for object-oriented augmentation.

    Args:
        None

    Returns:
        None
    """

    # Preparations
    file_name = "./sound.wav"
    if os.path.exists(file_name):
        os.remove(file_name)

    # Frequencies list, corresponds to channels quantity.
    freq_list = [400, 1000, 2333, 3700]

    fs = 441100  # Select sampling frequency, Hz.
    time_len = 5  # Length of signal in seconds.

    # Create Mcs-object and generate sine waves in 4 channels.
    mcs = wau.Mcs().generate(freq_list, time_len, fs)
    mcs.write(file_name)

    # Examples code for  README.md

    # Read WAV-file to Mcs-object.
    mcs.read(file_name)

    # Apply delays.
    delay_list = [100, 200, 300, 400]  # Corresponds to channels quantity.
    mcs.delay_ctrl(delay_list)

    # Apply amplitude changes.
    amplitude_list = [0.1, 0.2, 0.3, 0.4]  # Corresponds to channels quantity.
    mcs.amplitude_ctrl(amplitude_list)

    # Augmentation result saving.
    mcs.write("./sound_augmented.wav")

    # The same code in OOP approach:

    mcs = wau.Mcs().generate(freq_list, time_len, fs)
    w = wau.Mcs(mcs)
    w.rd("./sound.wav").dly([100, 200, 300, 400]).amp([0.1, 0.2, 0.3, 0.4]).wr(
        "./sound_augmented.wav"
    )

    # How to make 100 augmented files (amplitude and delay) from 1 sound file.
    v = wau.Mcs()
    v.rd(file_name)  # Read original file.
    result = []
    for _ in range(5):
        b = v.copy()
        b.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3]).dly(
            [100, 200, 300, 400], [30, 40, 50, 60]
        )
        result.append(b.get())


def test_sum():
    """
    Test function to verify the functionality of the `sum` function.

    This function generates two multichannel sounds using the `generate`
    function from the `wau` module with the given frequency lists, time
    duration, and sample rate. It then applies the `sum` function to the
    generated sounds and writes the result to a file using the `write` function
    from the `wau` module with the given file path and sample rate. Finally, it
    calculates the root mean square (RMS) values of the original and summed
    sounds and compares them to the expected values.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate([100], ctf.SIGNAL_TIME_LEN)
    test_sound_2 = wau.Mcs(fs=ctf.FS)
    test_sound_2.generate([300], ctf.SIGNAL_TIME_LEN)
    res = test_sound_1.copy()
    res.sum(test_sound_2)
    res.write(ctf.TEST_SOUND_1_FILE)
    ref = [0.707, 0.707, 1.0]
    for s, ref_value in zip([test_sound_1, test_sound_2, res], ref):
        r = s.rms(decimals=3)
        print(r)
        assert abs(r[0] - ref_value) < 0.001


def test_merge():
    """
    Test function to verify the functionality of the `merge` function.

    This function generates a multichannel sound using the `generate` function
    from the `wau` module with the given frequency lists, time duration, and
    sample rate. It then applies the `merge` function to the generated sound
    and writes the result to a file using the `write` function from the `wau`
    module with the given file path and sample rate. Finally, it calculates
    the root mean square (RMS) value of the merged sound and compares it to the
    expected value.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate([100, 300], ctf.SIGNAL_TIME_LEN)
    res = test_sound_1.copy()
    res.merge()
    res.write(ctf.TEST_SOUND_1_FILE)
    print("res.shape =", res.shape())
    ref_value = 1.0
    r = res.rms(decimals=3)
    print(r)
    assert abs(r[0] - ref_value) < 0.001


def test_split():
    """
    Test function to verify the functionality of the `split` function.

    This function generates a multichannel sound using the `generate` function
    from the `wau` module with the given frequency list, time duration, and
    sample rate. It then applies the `split` function to the generated sound
    and writes the result to a file using the `write` function from the `wau`
    module with the given file path and sample rate. Finally, it calculates the
    root mean square (RMS) value of the split sound and compares it to the
    expected value.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(fs=ctf.FS)
    test_sound_1.generate([300], ctf.SIGNAL_TIME_LEN)
    test_sound_1.split(5)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    ref_value = 0.707
    r = test_sound_1.rms(decimals=3)
    print(r)
    for i in range(0, test_sound_1.shape()[0]):
        assert abs(r[i] - ref_value) < 0.001


def test_chain_sum():
    """
    Test the functionality of the `sum` method in the `WaChain` class.

    This function creates two instances of the `WaChain` class, `w` and `res`,
    and generates a multichannel sound using the `gen` method of the `WaChain`
    class. It then copies the data of `w` to `res` and generates another
    multichannel sound using the `generate` function. The `sum` method is used
    to add the two sounds together, and the result is written to a file using
    the `wr` method.

    The function then calculates the root mean square (RMS) values of the
    original and summed sounds and compares them to the expected values.

    Args:
        None

    Returns:
        None
    """

    w = wau.Mcs()
    res = wau.Mcs()
    w.gen([100], ctf.SIGNAL_TIME_LEN, ctf.FS)
    res = w.copy()
    test_sound_2 = wau.Mcs()
    test_sound_2.generate([300], ctf.SIGNAL_TIME_LEN, ctf.FS)
    res.sum(test_sound_2).wr(ctf.TEST_SOUND_1_FILE)
    ref = [0.707, 0.707, 1.0]
    for s, ref_value in zip([w, test_sound_2, res], ref):
        r = s.rms(decimals=3)
        print(r)
        assert abs(r[0] - ref_value) < 0.001


def test_chain_merge():
    """
    Tests the functionality of the `merge` method in the `WaChain` class.

    This function creates an instance of the `WaChain` class, generates a
    multichannel sound using the `gen` method, merges the channels using the
    `mrg` method, writes the result to a file using the `wr` method, and
    calculates the root mean square (RMS) value of the merged sound using the
    `rms` method.

    Args:
        None

    Returns:
        None
    """

    w = wau.Mcs()
    r = (
        w.gen([100, 300], ctf.SIGNAL_TIME_LEN, ctf.FS)
        .mrg()
        .wr(ctf.TEST_SOUND_1_FILE)
        .rms(decimals=3)
    )
    print(r)
    ref_value = 1.0
    assert abs(r[0] - ref_value) < 0.001


def test_chain_split():
    """
    Test the functionality of the `splt` and `wr` methods in the `WaChain`
    class.

    This function creates a `WaChain` instance and generates a multichannel
    sound using the `gen` method. It then splits the channels using the `splt`
    method and writes the result to a file using the `wr` method. The function
    then checks the shape of the `data` attribute of the `WaChain` instance and
    compares it to the expected value. It also calculates the root mean square
    (RMS) value of the generated sound using the `rms` method and compares it
    to the expected value.

    Args:
        None

    Returns:
        None
    """

    w = wau.Mcs()
    w.gen([300], ctf.SIGNAL_TIME_LEN, ctf.FS).splt(5).wr(ctf.TEST_SOUND_1_FILE)
    c = w.data.shape[0]
    assert c == 5
    ref_value = 0.707
    r = w.rms(decimals=3)
    print(r)
    for i in range(0, c):
        assert abs(r[i] - ref_value) < 0.001


def test_chain_side_by_side():
    """
    Tests the functionality of the `sbs` method in the `WaChain` class.

    This function generates two multichannel sounds using the `generate`
    function from the `wau` module with the given frequency lists, time
    duration, and sample rate. It then applies the `sbs` method to the
    generated sounds and writes the result to a file using the `wr` method.
    The function then calculates the root mean square (RMS) value of the
    side-by-side sound using the `rms` method and compares it to the expected
    values.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs().generate([300], ctf.SIGNAL_TIME_LEN, ctf.FS)

    w = wau.Mcs()
    r = (
        w.gen([1000], ctf.SIGNAL_TIME_LEN, ctf.FS)
        .amp([0.3])
        .sbs(test_sound_1)
        .wr(ctf.TEST_SOUND_1_FILE)
        .rms(decimals=3)
    )
    print(r)
    ref_value = [0.212, 0.707]
    for r, ref in zip(r, ref_value):
        print(r)
        assert abs(r - ref) < 0.001


def test_side_by_side():
    """
    Tests the functionality of the side_by_side function.

    This function generates two multichannel sounds using the generate function
    from the wau module with the given frequency lists, time duration, and
    sample rate.  It then applies the side_by_side function to the generated
    sounds and writes the result to a file using the write function. The
    function then calculates the root mean square (RMS) value of the
    side-by-side sound using the rms method and compares it to the expected
    values.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs().generate([100], ctf.SIGNAL_TIME_LEN, ctf.FS)
    test_sound_1.amplitude_ctrl([0.3])
    test_sound_2 = wau.Mcs().generate([300], ctf.SIGNAL_TIME_LEN, ctf.FS)
    test_sound_1.side_by_side(test_sound_2)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    ref_list = [0.212, 0.707]
    r = test_sound_1.rms(decimals=3)
    for r, ref in zip(r, ref_list):
        print(r)
        assert abs(r - ref) < 0.001


def test_pause_detect():
    """
    Tests the functionality of the pause_detect function.

    This function generates a multichannel sound using the generate function
    from the wau module with the given frequency lists, time duration, and
    sample rate. It then applies the pause_detect function to the generated
    sound and writes the result to a file using the write function. The
    function then calculates the root mean square (RMS) value of the sound
    using the rms method and compares it to the expected values.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs().generate([100, 400], ctf.SIGNAL_TIME_LEN, ctf.FS)
    mask = test_sound_1.pause_detect([0.5, 0.3])
    test_sound_1.side_by_side(mask)
    print(test_sound_1)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    r = test_sound_1.rms(decimals=3)
    ref_list = [0.707, 0.707, 0.865, 0.923]
    for r, ref in zip(r, ref_list):
        print(r)
        assert abs(r - ref) < 0.001


def test_chain_pause_detect():
    """
    Tests the functionality of the WaChain class by creating two instances,
    generating a multichannel sound, copying the sound, applying pause
    detection, and then asserting that the RMS values of the resulting sound
    are within a certain tolerance of the reference values.

    Args:
        None

    Returns:
        None
    """

    w = wau.Mcs()
    w1 = wau.Mcs()
    w.gen([100, 400], ctf.SIGNAL_TIME_LEN, ctf.FS)
    w1 = w.copy()
    mask = w.pdt([0.5, 0.3])
    w1.sbs(mask).wr(ctf.TEST_SOUND_1_FILE)
    r = w1.rms(decimals=3)
    ref = [0.707, 0.707, 0.865, 0.923]
    for i, ri in enumerate(r):
        print(ri)
        assert abs(ri - ref[i]) < 0.001


def test_pause_shrink_sine():
    """
    Tests the functionality of the pause_shrink function.

    This function generates a multichannel sound using the generate function
    from the wau module with the given frequency lists, time duration, and
    sample rate. It then applies the pause_detect function to the generated
    sound and writes the result to a file using the write function. The
    function then applies the pause_shrink function to the generated sound and
    writes the result to a file using the write function. Finally, it calculates
    the root mean square (RMS) value of the sound using the rms method and
    compares it to the expected values.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs().generate([100, 400], ctf.SIGNAL_TIME_LEN, ctf.FS)
    mask = test_sound_1.pause_detect([0.5, 0.3])
    res = test_sound_1.copy()
    res.side_by_side(mask)
    print(res)
    test_sound_1.pause_shrink(mask, [20, 4])
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    r = test_sound_1.rms(decimals=3)
    ref = [0.702, 0.706, 0.865, 0.923]
    for r, ref in zip(r, ref):
        print(r)
        assert abs(r - ref) < 0.001


def test_pause_shrink_speech():
    """
    Tests the functionality of the pause_shrink function with speech-like
    input.

    This function generates a speech-like multichannel sound using the generate
    function from the wau module with the given frequency lists, time duration,
    and sample rate. It then applies the pause_detect function to the generated
    sound and writes the result to a file using the write function. The
    function then applies the pause_shrink function to the generated sound and
    writes the result to a file using the write function. Finally, it
    calculates the root mean square (RMS) value of the sound using the rms
    method and compares it to the expected values.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(seed=42)
    test_sound_1.generate(
        [100, 300], ctf.SIGNAL_TIME_LEN, ctf.FS, mode="speech"
    )
    mask = test_sound_1.pause_detect([0.5, 0.3])
    res = test_sound_1.copy()
    res.side_by_side(mask)
    res.write(ctf.TEST_SOUND_1_FILE)
    test_sound_1.pause_shrink(mask, [20, 4])
    r = test_sound_1.rms(decimals=3)
    ref = [0.331, 0.324]
    for r, ref in zip(r, ref):
        print(r)
        assert abs(r - ref) < 0.001


def test_pause_measure():
    """
    Tests the functionality of the pause_measure function.

    This function generates a multichannel sound using the generate function
    from the wau module with the given frequency lists, time duration, and
    sample rate. It then applies the pause_detect function to the generated
    sound and writes the result to a file using the write function. The
    function then applies the pause_measure function to the generated sound
    and writes the result to a file using the write function. Finally, it
    calculates the root mean square (RMS) value of the sound using the rms
    method and compares it to the expected values.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(seed=42).generate(
        [100, 300], 0.003, ctf.FS, mode="speech"
    )
    mask = test_sound_1.pause_detect([0.5, 0.3])
    res_list = wau.pause_measure(mask)
    print(res_list)

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
    """
    Tests the functionality of the pause_set function.

    This function generates a multichannel sound using the generate function
    from the wau module with the given frequency lists, time duration, and
    sample rate. It then applies the pause_detect function to the generated
    sound and writes the result to a file using the write function. The
    function then applies the pause_measure function to the generated sound
    and writes the result to a file using the write function. Finally, it
    calculates the root mean square (RMS) value of the sound using the rms
    method and compares it to the expected values.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(seed=42).generate(
        [100, 300], 0.003, ctf.FS, mode="speech"
    )
    mask = test_sound_1.pause_detect([0.5, 0.3])
    pause_list = wau.pause_measure(mask)
    test_sound_1.pause_set(pause_list, [10, 150])
    res = test_sound_1.copy()
    assert res.shape() == (2, 1618)
    print("res shape:", res.shape())
    print("res:", type(res.data[0, 1]))
    res.write(ctf.TEST_SOUND_1_FILE)
    r = res.rms(decimals=3)
    ref = [0.105, 0.113]
    for r, ref in zip(r, ref):
        print(r)
        assert abs(r - ref) < 0.001


def test_chain_add_chain():
    """
    Test function to verify the functionality of the `add_chain` method in the
    `WaChain` class.

    This function creates a `WaChain` instance, defines two chain commands as
    strings, adds them to the `chains` list of the `WaChain` instance, evaluates
    the chains, and compares the result to the expected values.

    Args:
        None

    Returns:
        None
    """
    mcs = wau.Mcs(fs=ctf.FS)
    w = wau.Mcs(mcs.data, mcs.sample_rate)  # Create a Mcs instance

    # Define the first chain command
    c1 = "gen([1000, 300], 5).amp([0.3]).rms(decimals=3)"
    # Define the second chain command
    c2 = "gen([700, 100], 5).amp([0.15]).rms(decimals=3)"
    w.achn([c1, c2])  # Add the chain commands to the chains list
    print(c1)  # Print the first chain command
    print(c2)  # Print the second chain command
    r = w.eval()  # Evaluate the chains
    print("r", r)  # Print the result
    ref_value = [[0.212], [0.106]]  # Define the expected values
    # Compare the result to the expected values
    for r, ref in zip(r, ref_value):
        print(r)  # Print the result
        # Assert that the result is within the expected tolerance
        assert abs(r[0] - ref[0]) < 0.001
    w = wau.Mcs(mcs.data, mcs.sample_rate)
    c1 = "gen([1000, 300], 5).amp([0.3]).rms(decimals=3)"
    c2 = "gen([700, 100], 5).amp([0.15]).rms(decimals=3)"
    w.achn([c1, c2])
    print(c1)
    print(c2)
    r = w.eval()
    print("r", r)
    ref_value = [[0.212], [0.106]]
    for r, ref in zip(r, ref_value):
        print(r)
        assert abs(r[0] - ref[0]) < 0.001
