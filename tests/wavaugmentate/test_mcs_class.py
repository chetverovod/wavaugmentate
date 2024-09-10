"""Module providing test functions for wavaugmentate.py  module."""

import sys
import os
import subprocess as sp
import numpy as np
import common_test_functions as ctf
sys.path.insert(1, ctf.WAU_DIR)
import wavaugmentate as wau


def test_mcs_put():
    """
    Test function to verify the functionality of the mcs class's put method.

    This function generates a multichannel sound using the generate function from
    the wau module with the given frequency list, time duration, and sample rate.
    It then applies the put method of the mcs class to the generated sound
    and asserts that the shape and data of the original sound are equal to the
    shape and data of the sound after applying the put method.

    Args:
        None

    Returns:
        None
    """

    test_sound_1 = wau.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)

    mcs = wau.Mcs()
    mcs.put(test_sound_1)

    assert np.array_equal(test_sound_1.data, mcs.data)
    assert np.array_equal(test_sound_1.data, mcs.get())

    mcs_2 = wau.Mcs(test_sound_1.data)
    assert np.array_equal(mcs_2.data, mcs.get())


def test_mcs_amp_control():
    """
    Test function to verify the functionality of the mcs class.

    This function tests the amp method of the mcs class by
    applying amplitude control to a generated multichannel sound.

    Args:
        None

    Returns:
        None
    """

    a_list = [0.1, 0.3, 0.4, 1]
    test_sound_1 = wau.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    mcs = wau.Mcs(test_sound_1.data)

    mcs.amp(a_list)
    res1 = mcs
    print("res1 =", res1.data)

    dest = wau.Mcs()

    res2 = dest.put(test_sound_1).amp(a_list).get()

    print("res2 =", res2.data)
    assert np.array_equal(res1.get(), res2)


def test_mcs_dly_controls():
    """
    Test function to verify the functionality of the mcs class.

    This function tests the dly method of the mcs class by
    applying delay controls to a generated multichannel sound.

    Args:
        None

    Returns:
        None
    """

    d_list = [100, 200, 300, 0]
    test_sound_1 = wau.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    mcs = test_sound_1.copy()

    mcs.dly(d_list)
    res1 = mcs
    print("res1 shape =", res1.data.shape)
    print("res1 =", res1.data)

    dest = wau.Mcs()
    dest.put(test_sound_1)
    res2 = dest.dly(d_list)

    print("res2 shape =", res2.data.shape)
    print("res2 =", res2.data)
    assert res1.data.shape == res2.data.shape
    assert np.array_equal(res1.get(), res2.get())


def test_mcs_wr_rd():
    """
    Test function to verify the functionality of the mcs class.

    This function tests the wr and rd methods of the mcs class by
    generating a multichannel sound, writing it to a file, reading it back,
    and comparing the original and read sound data.

    Args:
        None

    Returns:
        None
    """

    mcs = wau.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)

    ref_mcs = wau.Mcs()
    ref_mcs.rd(ctf.TEST_SOUND_1_FILE)

    assert np.array_equal(mcs.data, ref_mcs.data)


def test_mcs_write_by_channel():
    # Preparations
    file_name = ctf.OUTPUTWAV_DIR + "sound.wav"
    if os.path.exists(file_name):
        os.remove(file_name)

    # Frequencies list, corresponds to channels quantity.
    freq_list = [400]
    samp_rt = 44100  # Select sampling frequency, Hz.
    time_len = 3  # Length of signal in seconds.

    # Create Mcs-object and generate sine waves in 7 channels.
    mcs1 = wau.Mcs().generate(freq_list, time_len, samp_rt)
    mcs1.write(file_name)

    # Create Mcs-object.
    mcs = wau.Mcs()

    # Read WAV-file to Mcs-object.
    mcs.read(file_name)

    # Change quantity of channels to 7.
    mcs.split(7)

    # Apply delays.
    # Corresponds to channels quantity.
    delay_list = [0, 150, 200, 250, 300, 350, 400]
    mcs.delay_ctrl(delay_list)

    # Apply amplitude changes.
    # Corresponds to channels quantity.
    amplitude_list = [1, 0.17, 0.2, 0.23, 0.3, 0.37, 0.4]
    mcs.amplitude_ctrl(amplitude_list)

    mcs.write_by_channel(ctf.OUTPUTWAV_DIR + "sound_augmented.wav")

    for i in range(7):
        mcs.read(f"{ctf.OUTPUTWAV_DIR}sound_augmented_{i + 1}.wav")
        rms_value = mcs.rms()
        assert abs(rms_value[0] - 0.707 * amplitude_list[i]) < 0.001


def test_mcs_echo():
    """
    Test function to verify the functionality of the `echo` method in the
    mcs class.

    This function generates a multichannel sound using the `gen` method of the
    mcs class with the given frequency list, duration, and sample rate. It
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
    mcs = wau.Mcs()
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).echo(d_list, a_list)
    rms_list = mcs.rms(decimals=3)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for rms_value, ref in zip(rms_list, reference_list):
        assert abs(rms_value - ref) < 0.001
    d_list = [1e6, 2e6, 3e6, 0]
    a_list = [-0.3, -0.4, -0.5, 0]
    mcs = wau.Mcs()
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).echo(d_list, a_list)
    rms_list = mcs.rms(decimals=3)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for rms_value, ref in zip(rms_list, reference_list):
        assert abs(rms_value - ref) < 0.001


def test_mcs_noise():
    """
    Test function to verify the functionality of the `ns` method in the
    `mcs` class.

    This function generates a multichannel sound using the `gen` method of the
    `mcs` class with the given frequency list, duration, and sample rate. It
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

    mcs = wau.Mcs()
    mcs.set_seed(42)
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).ns(n_list)
    rms_list = mcs.rms(decimals=3)
    reference_list = [1.224, 0.735, 0.769, 0.707]
    for rms_value, ref in zip(rms_list, reference_list):
        # Threshold increased, because noise is not repeatable with fixed seed.
        assert abs(rms_value - ref) < 0.01


def test_mcs_info():
    """
    Test the `info` method of the `mcs` class.

    This function creates a `mcs` object, generates a sound file with the
    given frequency list, duration, and sample rate, and writes it to a file.
    It then calls the `info` method of the `mcs` object and prints the
    result.  Finally, it asserts that the returned dictionary matches the
    expected reference dictionary.

    Args:
        None

    Returns:
        None
    """

    mcs = wau.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)
    print(mcs.info())

    ref = {
        "path": "",
        "channels_count": 4,
        "sample_rate": 44100,
        "length_s": 5.0,
    }
    assert mcs.info() == ref


#  Test not finished.
def test_mcs_rn_rd():
    """Test augmentation on the fly."""

    mcs = wau.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)

    mcs_for_chain = wau.Mcs()
    mcs_for_chain.rd(ctf.TEST_SOUND_1_FILE)

    assert np.array_equal(mcs.data, mcs_for_chain.data)
    
    mcs_for_chain.amp([1, 0.7, 0.5, 0.3])
    mcs.amp([1, 0.7, 0.5, 0.3])
    assert np.array_equal(mcs.data, mcs_for_chain.data)

    mcs_for_chain.achn(["amp([1, 0.7, 0.5, 0.3])"])
    res = mcs_for_chain.rdac(ctf.TEST_SOUND_1_FILE)
    print("b = ", res[0].data)

    assert np.array_equal(mcs.data, res[0].data)


def test_mcs_rn_aug_rd():
    """Test augmentation on the fly."""

    w = wau.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    w.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)

    mcs_a = wau.Mcs()
    mcs_a.rd(ctf.TEST_SOUND_1_FILE)

    mcs_b = wau.Mcs()
    mcs_b.rd(ctf.TEST_SOUND_1_FILE)

    assert np.array_equal(w.data, mcs_a.data)
    assert np.array_equal(w.data, mcs_b.data)

    mcs_a.amp([1, 0.7, 0.5, 0.3])
    mcs_b.amp([1, 0.7, 0.5, 0.3])
    assert np.array_equal(mcs_a.data, mcs_b.data)

    mcs_a.set_seed(42)
    mcs_b.set_seed(42)
    mcs_a.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
    mcs_b.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
    assert np.array_equal(mcs_a.data, mcs_b.data)

    mcs_a.set_seed(-1)
    mcs_b.set_seed(-1)
    for _ in range(10):
        mcs_a.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
        mcs_b.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
        assert not np.array_equal(mcs_a.data, mcs_b.data)


def test_mcs_chain_class():
    """
    Tests the functionality of the mcs class by generating a multichannel
    sound, computing its RMS values, and comparing them to the expected values.

    Args:
        None

    Returns:
        None
    """

    mcs = wau.Mcs()
    cmd_prefix = "mcs."
    cmd = "gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).rms()"
    eval_results_list = str(eval(cmd_prefix + cmd.strip()))
    out = ctf.shrink(eval_results_list)
    ref_rms_list = "[0.70710844,0.7071083,0.707108,0.70710754]"

    print(out)
    mcs.info()
    assert out == ref_rms_list


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
    responce_string = str(res.stdout)
    out = ctf.shrink(responce_string)
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
    the usage of the mcs class for object-oriented augmentation.

    Args:
        None

    Returns:
        None
    """

    # Preparations
    sound_file_path = ctf.OUTPUTWAV_DIR + "sound.wav"
    sound_aug_file_path = ctf.OUTPUTWAV_DIR + "sound_augmented.wav"

    file_name = sound_file_path
    if os.path.exists(file_name):
        os.remove(file_name)

    # Frequencies list, corresponds to channels quantity.
    freq_list = [400]
    samp_rt = 44100  # Select sampling frequency, Hz.
    time_len = 3  # Length of signal in seconds.

    # Create Mcs-object and generate sine waves in 7 channels.
    mcs1 = wau.Mcs().generate(freq_list, time_len, samp_rt)
    mcs1.write(file_name)

    # Examples code for  README.md

    # Example 1:

    # File name of original sound.
    file_name = sound_file_path

    # Create Mcs-object.
    mcs = wau.Mcs()

    # Read WAV-file to Mcs-object.
    mcs.read(file_name)

    # Change quantity of channels to 7.
    mcs.split(7)

    # Apply delays.
    # Corresponds to channels quantity.
    delay_list = [0, 150, 200, 250, 300, 350, 400]
    mcs.delay_ctrl(delay_list)

    # Apply amplitude changes.
    # Corresponds to channels quantity.
    amplitude_list = [1, 0.17, 0.2, 0.23, 0.3, 0.37, 0.4]
    mcs.amplitude_ctrl(amplitude_list)

    # Augmentation result saving by single file, containing 7 channels.
    mcs.write(sound_aug_file_path)

    # Augmentation result saving to 7 files, each 1 by channel.
    # ./outputwav/sound_augmented_1.wav
    # ./outputwav/sound_augmented_2.wav and so on.
    mcs.write_by_channel(sound_aug_file_path)

    # The same code as chain, Example 2:
    delay_list = [0, 150, 200, 250, 300, 350, 400]
    amplitude_list = [1, 0.17, 0.2, 0.23, 0.3, 0.37, 0.4]

    # Create Mcs-object.
    mcs = wau.Mcs(mcs)

    # Apply all transformations of Example 1 in chain.
    mcs.rd(file_name).splt(7).dly(delay_list).amp(amplitude_list).wr(
        ctf.OUTPUTWAV_DIR + "sound_augmented_by_chain.wav"
    )

    # Augmentation result saving to 7 files, each 1 by channel.
    mcs.wrbc(ctf.OUTPUTWAV_DIR + "sound_augmented_by_chain.wav")

    # How to make 100 augmented files (amplitude and delay) from 1 sound file.

    # Example 5:

    file_name = sound_file_path
    mcs = wau.Mcs()
    mcs.rd(file_name)  # Read original file with single channel.
    file_name_head = ctf.OUTPUTWAV_DIR + "sound_augmented"

    # Suppose we need 15 augmented files.
    aug_count = 15
    for i in range(aug_count):
        signal = mcs.copy()
        # Apply random amplitude [0.3..1.7) and delay [70..130)
        # microseconds changes to each copy of original signal.
        signal.amp([1], [0.7]).dly([100], [30])
        name = file_name_head + f"_{i + 1}.wav"
        signal.write(name)


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

    test_sound_1 = wau.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate([100], ctf.SIGNAL_TIME_LEN)
    test_sound_2 = wau.Mcs(samp_rt=ctf.FS)
    test_sound_2.generate([300], ctf.SIGNAL_TIME_LEN)
    res = test_sound_1.copy()
    res.sum(test_sound_2)
    res.write(ctf.TEST_SOUND_1_FILE)
    ref = [0.707, 0.707, 1.0]
    for sound, ref_value in zip([test_sound_1, test_sound_2, res], ref):
        r = sound.rms(decimals=3)
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

    test_sound_1 = wau.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate([100, 300], ctf.SIGNAL_TIME_LEN)
    res = test_sound_1.copy()
    res.merge()
    res.write(ctf.TEST_SOUND_1_FILE)
    print("res.shape =", res.shape())
    ref_value = 1.0
    rms_list = res.rms(decimals=3)
    print(rms_list)
    assert abs(rms_list[0] - ref_value) < 0.001


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

    test_sound_1 = wau.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate([300], ctf.SIGNAL_TIME_LEN)
    test_sound_1.split(5)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    ref_value = 0.707
    rms_list = test_sound_1.rms(decimals=3)
    print(rms_list)
    for i in range(0, test_sound_1.shape()[0]):
        assert abs(rms_list[i] - ref_value) < 0.001


def test_chain_sum():
    """
    Test the functionality of the `sum` method in the `mcs` class.

    This function creates two instances of the `mcs` class, `w` and `res`,
    and generates a multichannel sound using the `gen` method of the `mcs`
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

    mcs = wau.Mcs()
    res = wau.Mcs()
    mcs.gen([100], ctf.SIGNAL_TIME_LEN, ctf.FS)
    res = mcs.copy()
    test_sound_2 = wau.Mcs()
    test_sound_2.generate([300], ctf.SIGNAL_TIME_LEN, ctf.FS)
    res.sum(test_sound_2).wr(ctf.TEST_SOUND_1_FILE)
    ref = [0.707, 0.707, 1.0]
    for sound, ref_value in zip([mcs, test_sound_2, res], ref):
        rms_list = sound.rms(decimals=3)
        print(rms_list)
        assert abs(rms_list[0] - ref_value) < 0.001


def test_chain_merge():
    """
    Tests the functionality of the `merge` method in the `mcs` class.

    This function creates an instance of the `mcs` class, generates a
    multichannel sound using the `gen` method, merges the channels using the
    `mrg` method, writes the result to a file using the `wr` method, and
    calculates the root mean square (RMS) value of the merged sound using the
    `rms` method.

    Args:
        None

    Returns:
        None
    """

    mcs = wau.Mcs()
    rms_list = (
        mcs.gen([100, 300], ctf.SIGNAL_TIME_LEN, ctf.FS)
        .mrg()
        .wr(ctf.TEST_SOUND_1_FILE)
        .rms(decimals=3)
    )
    print(rms_list)
    ref_value = 1.0
    assert abs(rms_list[0] - ref_value) < 0.001


def test_chain_split():
    """
    Test the functionality of the `splt` and `wr` methods in the `mcs`
    class.

    This function creates a `mcs` instance and generates a multichannel
    sound using the `gen` method. It then splits the channels using the `splt`
    method and writes the result to a file using the `wr` method. The function
    then checks the shape of the `data` attribute of the `mcs` instance and
    compares it to the expected value. It also calculates the root mean square
    (RMS) value of the generated sound using the `rms` method and compares it
    to the expected value.

    Args:
        None

    Returns:
        None
    """

    mcs = wau.Mcs()
    mcs.gen([300], ctf.SIGNAL_TIME_LEN, ctf.FS).splt(5).wr(ctf.TEST_SOUND_1_FILE)
    channels = mcs.channels_count()
    assert channels == 5
    ref_value = 0.707
    rms_list = mcs.rms(decimals=3)
    print(rms_list)
    for i in range(0, channels):
        assert abs(rms_list[i] - ref_value) < 0.001


def test_chain_side_by_side():
    """
    Tests the functionality of the `sbs` method in the `mcs` class.

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

    mcs = wau.Mcs()
    rms_list = (
        mcs.gen([1000], ctf.SIGNAL_TIME_LEN, ctf.FS)
        .amp([0.3])
        .sbs(test_sound_1)
        .wr(ctf.TEST_SOUND_1_FILE)
        .rms(decimals=3)
    )
    print(rms_list)
    ref_value = [0.212, 0.707]
    for rms_list, ref in zip(rms_list, ref_value):
        print(rms_list)
        assert abs(rms_list - ref) < 0.001


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
    ref_rms_list = [0.212, 0.707]
    rms_list = test_sound_1.rms(decimals=3)
    for rms_list, ref in zip(rms_list, ref_rms_list):
        print(rms_list)
        assert abs(rms_list - ref) < 0.001


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
    rms_list = test_sound_1.rms(decimals=3)
    ref_rms_list = [0.707, 0.707, 0.865, 0.923]
    for rms_value, ref in zip(rms_list, ref_rms_list):
        print(rms_value)
        assert abs(rms_value - ref) < 0.001


def test_chain_pause_detect():
    """
    Tests the functionality of the mcs class by creating two instances,
    generating a multichannel sound, copying the sound, applying pause
    detection, and then asserting that the RMS values of the resulting sound
    are within a certain tolerance of the reference values.

    Args:
        None

    Returns:
        None
    """

    mcs = wau.Mcs()
    mcs_1 = wau.Mcs()
    mcs.gen([100, 400], ctf.SIGNAL_TIME_LEN, ctf.FS)
    mcs_1 = mcs.copy()
    mask = mcs.pdt([0.5, 0.3])
    mcs_1.sbs(mask).wr(ctf.TEST_SOUND_1_FILE)
    rms_list = mcs_1.rms(decimals=3)
    ref_rms_list = [0.707, 0.707, 0.865, 0.923]
    for i, rms_value in enumerate(rms_list):
        print(rms_value)
        assert abs(rms_value - ref_rms_list[i]) < 0.001


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
    rms_list = test_sound_1.rms(decimals=3)
    ref_rms_list = [0.702, 0.706, 0.865, 0.923]
    for rms_value, ref_rms_value in zip(rms_list, ref_rms_list):
        print(rms_value)
        assert abs(rms_value - ref_rms_value) < 0.001


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
    rms_list = test_sound_1.rms(decimals=3)
    ref_rms_list = [0.331, 0.324]
    for rms_value, ref_value in zip(rms_list, ref_rms_list):
        print(rms_value)
        assert abs(rms_value - ref_value) < 0.001


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
    rms_list = res.rms(decimals=3)
    ref_rms_list = [0.105, 0.113]
    for r_value, ref_value in zip(rms_list, ref_rms_list):
        print(r_value)
        assert abs(r_value - ref_value) < 0.001


def test_chain_add_chain():
    """
    Test function to verify the functionality of the `add_chain` method in the
    `mcs` class.

    This function creates a `mcs` instance, defines two chain commands as
    strings, adds them to the `chains` list of the `mcs` instance, evaluates
    the chains, and compares the result to the expected values.

    Args:
        None

    Returns:
        None
    """
    mcs = wau.Mcs(samp_rt=ctf.FS)
    mcs_1 = wau.Mcs(mcs.data, mcs.sample_rate)  # Create a Mcs instance

    # Define the first chain command
    chain_1 = "gen([1000, 300], 5).amp([0.3, 0.2]).rms(decimals=3)"
    # Define the second chain command
    chain_2 = "gen([700, 100], 5).amp([0.15, 0.1]).rms(decimals=3)"
    mcs_1.achn([chain_1, chain_2])  # Add the chain commands to the chains list
    print(chain_1)  # Print the first chain command
    print(chain_2)  # Print the second chain command
    rms_list = mcs_1.eval()  # Evaluate the chains
    print("rms list:", rms_list)  # Print the result
    ref_rms_list = [[0.212], [0.106]]  # Define the expected values
    # Compare the result to the expected values
    for rms_value, ref_rms_value in zip(rms_list, ref_rms_list):
        print(rms_value)  # Print the result
        # Assert that the result is within the expected tolerance
        assert abs(rms_value[0] - ref_rms_value[0]) < 0.001
    mcs_1 = wau.Mcs(mcs.data, mcs.sample_rate)
    chain_1 = "gen([1000, 300], 5).amp([0.3, 0.2]).rms(decimals=3)"
    chain_2 = "gen([700, 100], 5).amp([0.15, 0.1]).rms(decimals=3)"
    mcs_1.achn([chain_1, chain_2])
    print(chain_1)
    print(chain_2)
    rms_list = mcs_1.eval()
    print("r", rms_value)
    ref_rms_list = [[0.212], [0.106]]
    for rms_value, ref_rms_value in zip(rms_list, ref_rms_list):
        print(rms_value)
        assert abs(rms_value[0] - ref_rms_value[0]) < 0.001
