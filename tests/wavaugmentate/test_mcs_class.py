"""Module provides test functions for mcs.py  module."""

import os
import common_test_functions as ctf
import numpy as np
import wavaugmentate.mcs as ms
from wavaugmentate.mcs import Mcs
from wavaugmentate.aug import Aug


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

    test_sound_1 = Mcs(samp_rt=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)

    mcs = Mcs()
    mcs.put(test_sound_1)

    assert np.array_equal(test_sound_1.data, mcs.data)
    assert np.array_equal(test_sound_1.data, mcs.get())

    mcs_2 = Mcs(test_sound_1.data)
    assert np.array_equal(mcs_2.data, mcs.get())


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

    mcs = Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)

    ref_mcs = Mcs()
    ref_mcs.rd(ctf.TEST_SOUND_1_FILE)

    assert np.array_equal(mcs.data, ref_mcs.data)


def test_mcs_write_by_channel():
    """
    Test function to verify the functionality of the mcs class's
    write_by_channel method.

    This function generates a multichannel sound using the generate method of
    the mcs class with the given frequency list, time duration, and sample rate.
    It then writes the generated sound to a file using the write method of the
    mcs class.  The function reads the written sound back into the mcs object
    using the read method and changes the quantity of channels to 7 using the
    split method.  It applies delays and amplitude changes to the sound using
    the delay_ctrl and amplitude_ctrl methods, respectively.  Finally, it writes
    the sound to separate WAV files for each channel using the write_by_channel
    method and verifies the RMS values of the written sounds.

    Args:
        None

    Returns:
        None
    """

    # Preparations
    file_name = ctf.OUTPUTWAV_DIR + "sound.wav"
    if os.path.exists(file_name):
        os.remove(file_name)

    # Frequencies list, corresponds to channels quantity.
    freq_list = [400]
    samp_rt = 44100  # Select sampling frequency, Hz.
    time_len = 3  # Length of signal in seconds.

    # Create Mcs-object and generate sine waves in 7 channels.
    mcs1 = Mcs().generate(freq_list, time_len, samp_rt)
    mcs1.write(file_name)

    # Create Mcs-object.
    mcs = Mcs()

    # Read WAV-file to Mcs-object.
    mcs.read(file_name)

    # Change quantity of channels to 7.
    mcs.split(7)

    # Apply delays.
    # Corresponds to channels quantity.
    delay_list = [0, 150, 200, 250, 300, 350, 400]
    aug_obj = Aug(mcs)
    aug_obj.delay_ctrl(delay_list)

    # Apply amplitude changes.
    # Corresponds to channels quantity.
    amplitude_list = [1, 0.17, 0.2, 0.23, 0.3, 0.37, 0.4]
    aug_obj.amplitude_ctrl(amplitude_list)

    aug_obj.get().write_by_channel(ctf.OUTPUTWAV_DIR + "sound_augmented.wav")

    for i in range(7):
        mcs.read(f"{ctf.OUTPUTWAV_DIR}sound_augmented_{i + 1}.wav")
        rms_value = mcs.rms()
        assert abs(rms_value[0] - 0.707 * amplitude_list[i]) < ctf.ABS_ERR


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

    mcs = Mcs()
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

    test_sound_1 = Mcs(samp_rt=ctf.FS)
    test_sound_1.generate([100], ctf.SIGNAL_TIME_LEN)
    test_sound_2 = Mcs(samp_rt=ctf.FS)
    test_sound_2.generate([300], ctf.SIGNAL_TIME_LEN)
    res = test_sound_1.copy()
    res.sum(test_sound_2)
    res.write(ctf.TEST_SOUND_1_FILE)
    ref = [0.707, 0.707, 1.0]
    for sound, ref_value in zip([test_sound_1, test_sound_2, res], ref):
        rms_value = sound.rms(decimals=3)
        print(rms_value)
        assert abs(rms_value[0] - ref_value) < ctf.ABS_ERR


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

    test_sound_1 = Mcs(samp_rt=ctf.FS)
    test_sound_1.generate([100, 300], ctf.SIGNAL_TIME_LEN)
    res = test_sound_1.copy()
    res.merge()
    res.write(ctf.TEST_SOUND_1_FILE)
    print("res.shape =", res.shape())
    ref_value = 1.0
    rms_list = res.rms(decimals=3)
    print(rms_list)
    assert abs(rms_list[0] - ref_value) < ctf.ABS_ERR


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

    test_sound_1 = Mcs(samp_rt=ctf.FS)
    test_sound_1.generate([300], ctf.SIGNAL_TIME_LEN)
    test_sound_1.split(5)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    ref_value = 0.707
    rms_list = test_sound_1.rms(decimals=3)
    print(rms_list)
    for i in range(0, test_sound_1.shape()[0]):
        assert abs(rms_list[i] - ref_value) < ctf.ABS_ERR


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

    mcs = Mcs()
    res = Mcs()
    mcs.gen([100], ctf.SIGNAL_TIME_LEN, ctf.FS)
    res = mcs.copy()
    test_sound_2 = Mcs()
    test_sound_2.generate([300], ctf.SIGNAL_TIME_LEN, ctf.FS)
    res.sum(test_sound_2).wr(ctf.TEST_SOUND_1_FILE)
    ref = [0.707, 0.707, 1.0]
    for sound, ref_value in zip([mcs, test_sound_2, res], ref):
        rms_list = sound.rms(decimals=3)
        print(rms_list)
        assert abs(rms_list[0] - ref_value) < ctf.ABS_ERR


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

    mcs = Mcs()
    rms_list = (
        mcs.gen([100, 300], ctf.SIGNAL_TIME_LEN, ctf.FS)
        .mrg()
        .wr(ctf.TEST_SOUND_1_FILE)
        .rms(decimals=3)
    )
    print(rms_list)
    ref_value = 1.0
    assert abs(rms_list[0] - ref_value) < ctf.ABS_ERR


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

    mcs = Mcs()
    mcs.gen([300], ctf.SIGNAL_TIME_LEN, ctf.FS).splt(5).wr(ctf.TEST_SOUND_1_FILE)
    channels = mcs.info()['channels_count']
    assert channels == 5
    ref_value = 0.707
    rms_list = mcs.rms(decimals=3)
    print(rms_list)
    for i in range(0, channels):
        assert abs(rms_list[i] - ref_value) < ctf.ABS_ERR


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

    test_sound_1 = Mcs().generate([100], ctf.SIGNAL_TIME_LEN, ctf.FS)
    aug_obj = Aug(test_sound_1)
    test_sound_1 = aug_obj.amplitude_ctrl([0.3]).get()
    test_sound_2 = Mcs().generate([300], ctf.SIGNAL_TIME_LEN, ctf.FS)
    test_sound_1.side_by_side(test_sound_2)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    ref_rms_list = [0.212, 0.707]
    rms_list = test_sound_1.rms(decimals=3)
    for rms_list, ref in zip(rms_list, ref_rms_list):
        print(rms_list)
        assert abs(rms_list - ref) < ctf.ABS_ERR


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

    test_sound_1 = Mcs().generate([100, 400], ctf.SIGNAL_TIME_LEN, ctf.FS)
    mask = test_sound_1.pause_detect([0.5, 0.3])
    test_sound_1.side_by_side(mask)
    print(test_sound_1)
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    rms_list = test_sound_1.rms(decimals=3)
    ref_rms_list = [0.707, 0.707, 0.865, 0.923]
    for rms_value, ref in zip(rms_list, ref_rms_list):
        print(rms_value)
        assert abs(rms_value - ref) < ctf.ABS_ERR


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

    mcs = Mcs()
    mcs_1 = Mcs()
    mcs.gen([100, 400], ctf.SIGNAL_TIME_LEN, ctf.FS)
    mcs_1 = mcs.copy()
    mask = mcs.pdt([0.5, 0.3])
    mcs_1.sbs(mask).wr(ctf.TEST_SOUND_1_FILE)
    rms_list = mcs_1.rms(decimals=3)
    ref_rms_list = [0.707, 0.707, 0.865, 0.923]
    for i, rms_value in enumerate(rms_list):
        print(rms_value)
        assert abs(rms_value - ref_rms_list[i]) < ctf.ABS_ERR


def test_pause_shrink_sine():
    """
    Tests the functionality of the pause_shrink function.

    This function generates a multichannel sound using the generate function
    from the wau module with the given frequency lists, time duration, and
    sample rate. It then applies the pause_detect function to the generated
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

    test_sound_1 = Mcs().generate([100, 400], ctf.SIGNAL_TIME_LEN, ctf.FS)
    mask = test_sound_1.pause_detect([0.5, 0.3])
    res = test_sound_1.copy()
    res.side_by_side(mask)
    print(res)
    test_sound_1.pause_shrink(mask, [20, 4])
    test_sound_1.write(ctf.TEST_SOUND_1_FILE)
    _rms_list = test_sound_1.rms(decimals=3)
    _ref_rms_list = [0.702, 0.706, 0.865, 0.923]
    for _rms_value, ref_rms_value in zip(_rms_list, _ref_rms_list):
        print(_rms_value)
        assert abs(_rms_value - ref_rms_value) < ctf.ABS_ERR


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

    test_sound_1 = Mcs(seed=42)
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
        assert abs(rms_value - ref_value) < ctf.ABS_ERR


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

    test_sound_1 = Mcs(seed=42).generate(
        [100, 300], 0.003, ctf.FS, mode="speech"
    )
    mask = test_sound_1.pause_detect([0.5, 0.3])
    res_list = ms.pause_measure(mask)

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
