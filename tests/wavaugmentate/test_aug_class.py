"""Module provides test functions for mcs.py  module."""

import os
import sys

import common_test_functions as ctf
import numpy as np

sys.path.insert(1, ctf.WAU_DIR)
import mcs as ms
import aug as am
from mcs import Mcs
from aug import Aug




def test_aug_amp_control():
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
    test_sound_1 = ms.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    ao = am.Aug(test_sound_1)
    
    ao.amp(a_list)
    res1 = ao.signal
    print("res1 =", res1.data)

    dest = am.Aug()

    res2 = dest.put(test_sound_1).amp(a_list).get()

    print("res2 =", res2.data)
    assert np.array_equal(res1.get(), res2.get())


def test_aug_dly_controls():
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
    test_sound_1 = ms.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    #mcs = test_sound_1.copy()
    ao = Aug(test_sound_1)

    ao.dly(d_list)
    res1 = ao.get()
    print("res1 shape =", res1.data.shape)
    print("res1 =", res1.data)

    dest = Aug()
    dest.put(test_sound_1)
    res2 = dest.dly(d_list).get()

    print("res2 shape =", res2.data.shape)
    print("res2 =", res2.data)
    assert res1.data.shape == res2.data.shape
    assert np.array_equal(res1.get(), res2.get())


def test_aug_echo():
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
    mcs = ms.Mcs()
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS)
    ao = am.Aug(mcs)
    ao.echo(d_list, a_list)
    rms_list = ao.get().rms(decimals=3)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for rms_value, ref in zip(rms_list, reference_list):
        assert abs(rms_value - ref) < ctf.ABS_ERR

    d_list = [1e6, 2e6, 3e6, 0]
    a_list = [-0.3, -0.4, -0.5, 0]
    mcs = ms.Mcs()
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS)
    ao = am.Aug(mcs)
    ao.echo(d_list, a_list)
    rms_list = ao.get().rms(decimals=3)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for rms_value, ref in zip(rms_list, reference_list):
        assert abs(rms_value - ref) < ctf.ABS_ERR


def test_aug_noise():
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

    mcs = ms.Mcs()
    mcs.set_seed(42)
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS)
    ao = Aug(mcs)
    ao.ns(n_list)
    rms_list = ao.get().rms(decimals=3)
    reference_list = [1.224, 0.735, 0.769, 0.707]
    for rms_value, ref in zip(rms_list, reference_list):
        # Threshold increased, because noise is not repeatable with fixed seed.
        assert abs(rms_value - ref) < 0.01




#  Test not finished.
def test_aug_rn_rd():
    """Test augmentation on the fly."""

    mcs = ms.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)
    ao = Aug(mcs)

    mcs_for_chain = ms.Mcs()
    mcs_for_chain.rd(ctf.TEST_SOUND_1_FILE)
    ao1 = Aug(mcs_for_chain)

    assert np.array_equal(ao.get().data, ao1.get().data)

    ao1.amp([1, 0.7, 0.5, 0.3])
    ao.amp([1, 0.7, 0.5, 0.3])
    assert np.array_equal(ao.get().data, ao1.get().data)

    """
    ao1.achn(["amp([1, 0.7, 0.5, 0.3])"])
    res = ao1.rdac(ctf.TEST_SOUND_1_FILE)
    print("b = ", res[0].data)

    assert np.array_equal(mcs.data, res[0].data)
    """


def test_aug_rn_aug_rd():
    """Test augmentation on the fly."""

    mcs = ms.Mcs()
    if os.path.exists(ctf.TEST_SOUND_1_FILE):
        os.remove(ctf.TEST_SOUND_1_FILE)
    mcs.gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).wr(ctf.TEST_SOUND_1_FILE)

    ao_a = Aug(Mcs().rd(ctf.TEST_SOUND_1_FILE))

    ao_b = Aug(Mcs().rd(ctf.TEST_SOUND_1_FILE))

    assert np.array_equal(mcs.data, ao_a.get().data)
    assert np.array_equal(mcs.data, ao_b.get().data)

    ao_a.amp([1, 0.7, 0.5, 0.3])
    ao_b.amp([1, 0.7, 0.5, 0.3])
    assert np.array_equal(ao_a.get().data, ao_b.get().data)

    ao_a.set_seed(42)
    ao_b.set_seed(42)
    ao_a.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
    ao_b.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
    assert np.array_equal(ao_a.get().data, ao_b.get().data)

    ao_a.set_seed(-1)
    ao_b.set_seed(-1)
    for _ in range(10):
        ao_a.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
        ao_b.amp([1, 0.7, 0.5, 0.3], [1, 0.7, 0.5, 0.3])
        assert not np.array_equal(ao_a.get().data, ao_b.get().data)


def test_aug_chaininig():
    """
    Tests the functionality of the mcs class by generating a multichannel
    sound, computing its RMS values, and comparing them to the expected values.

    Args:
        None

    Returns:
        None
    """

    mcs = ms.Mcs()
    cmd_prefix = "mcs."
    cmd = "gen(ctf.f_list, ctf.SIGNAL_TIME_LEN, ctf.FS).rms()"
    eval_results_list = str(eval(cmd_prefix + cmd.strip()))
    out = ctf.shrink(eval_results_list)
    ref_rms_list = "[0.70710844,0.7071083,0.707108,0.70710754]"

    print(out)
    mcs.info()
    assert out == ref_rms_list


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
    mcs1 = Mcs().generate(freq_list, time_len, samp_rt)
    mcs1.write(file_name)

    # Examples code for  README.md

    # Example 1:

    # File name of original sound.
    file_name = sound_file_path

    # Create Mcs-object.
    mcs = Mcs()

    # Read WAV-file to Mcs-object.
    mcs.read(file_name)

    # Change quantity of channels to 7.
    mcs.split(7)
   
    # Create augmentation object.
    aug = Aug(mcs)

    # Apply delays.
    # Corresponds to channels quantity.
    delay_list = [0, 150, 200, 250, 300, 350, 400]
    aug.delay_ctrl(delay_list)

    # Apply amplitude changes.
    # Corresponds to channels quantity.
    amplitude_list = [1, 0.17, 0.2, 0.23, 0.3, 0.37, 0.4]
    aug.amplitude_ctrl(amplitude_list)

    # Augmentation result saving by single file, containing 7 channels.
    aug.get().write(sound_aug_file_path)

    # Augmentation result saving to 7 files, each 1 by channel.
    # ./outputwav/sound_augmented_1.wav
    # ./outputwav/sound_augmented_2.wav and so on.
    aug.get().write_by_channel(sound_aug_file_path)

    # The same code as chain, Example 2:
    delay_list = [0, 150, 200, 250, 300, 350, 400]
    amplitude_list = [1, 0.17, 0.2, 0.23, 0.3, 0.37, 0.4]

    # Apply all transformations of Example 1 in chain.
    Aug(Mcs().rd(file_name)).splt(7).dly(delay_list).amp(amplitude_list).get().wr(
        ctf.OUTPUTWAV_DIR + "sound_augmented_by_chain.wav"
    )

    # Augmentation result saving to 7 files, each 1 by channel.
    mcs.wrbc(ctf.OUTPUTWAV_DIR + "sound_augmented_by_chain.wav")

    # How to make 100 augmented files (amplitude and delay) from 1 sound file.

    # Example 5:

    file_name = sound_file_path
    mcs = ms.Mcs()
    mcs.rd(file_name)  # Read original file with single channel.
    file_name_head = ctf.OUTPUTWAV_DIR + "sound_augmented"

    # Suppose we need 15 augmented files.
    aug_count = 15
    for i in range(aug_count):
        signal = Aug(mcs.copy())
        # Apply random amplitude [0.3..1.7) and delay [70..130)
        # microseconds changes to each copy of original signal.
        signal.amp([1], [0.7]).dly([100], [30])
        name = file_name_head + f"_{i + 1}.wav"
        signal.get().write(name)


def test_aug_noise_ctrl():
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

    test_sound_1 = ms.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate(ctf.f_list, ctf.SIGNAL_TIME_LEN)
    test_sound_1.set_seed(42)
    test_nc = Aug(test_sound_1).noise_ctrl([1, 0.2, 0.3, 0]).get()
    test_nc.write(ctf.TEST_SOUND_1_NOISE_FILE)
    rms_list = test_nc.rms(decimals=3)
    reference_list = [1.224, 0.735, 0.769, 0.707]

    for rms_value, ref in zip(rms_list, reference_list):
        # Threshold increased, because noise is not repeatable with fixed seed.
        assert abs(rms_value - ref) < 0.01



def test_aug_sum():
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

    test_sound_1 = ms.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate([100], ctf.SIGNAL_TIME_LEN)
    test_sound_2 = ms.Mcs(samp_rt=ctf.FS)
    test_sound_2.generate([300], ctf.SIGNAL_TIME_LEN)
    
    res = Aug(test_sound_1)
    res.sum(test_sound_2)
    res.get().write(ctf.TEST_SOUND_1_FILE)
    ref = [0.707, 0.707, 1.0]
    for sound, ref_value in zip([test_sound_1, test_sound_2, res.get()], ref):
        rms_value = sound.rms(decimals=3)
        print(rms_value)
        assert abs(rms_value[0] - ref_value) < ctf.ABS_ERR


def test_aug_merge():
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

    test_sound_1 = ms.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate([100, 300], ctf.SIGNAL_TIME_LEN)
    res = Aug(test_sound_1.copy())
    res.merge()
    res.get().write(ctf.TEST_SOUND_1_FILE)
    print("res.get().shape =", res.get().shape())
    ref_value = 1.0
    rms_list = res.get().rms(decimals=3)
    print(rms_list)
    assert abs(rms_list[0] - ref_value) < ctf.ABS_ERR


def test_aug_split():
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

    test_sound_1 = ms.Mcs(samp_rt=ctf.FS)
    test_sound_1.generate([300], ctf.SIGNAL_TIME_LEN)
    ao = Aug(test_sound_1)
    ao.split(5)
    ao.get().write(ctf.TEST_SOUND_1_FILE)
    ref_value = 0.707
    rms_list = ao.get().rms(decimals=3)
    print(rms_list)
    for i in range(0, ao.get().shape()[0]):
        assert abs(rms_list[i] - ref_value) < ctf.ABS_ERR


def test_aug_chain_sum():
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

    mcs = ms.Mcs()
    res = ms.Mcs()
    mcs.gen([100], ctf.SIGNAL_TIME_LEN, ctf.FS)
    res = Aug(mcs.copy())
    test_sound_2 = ms.Mcs()
    test_sound_2.generate([300], ctf.SIGNAL_TIME_LEN, ctf.FS)
    res.sum(test_sound_2).get().wr(ctf.TEST_SOUND_1_FILE)
    ref = [0.707, 0.707, 1.0]
    for sound, ref_value in zip([mcs, test_sound_2, res.get()], ref):
        rms_list = sound.rms(decimals=3)
        print(rms_list)
        assert abs(rms_list[0] - ref_value) < ctf.ABS_ERR


def test_aug_chain_merge():
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

    ao = Aug()
    rms_list = (
        ao.get().gen([100, 300], ctf.SIGNAL_TIME_LEN, ctf.FS)
        .mrg()
        .wr(ctf.TEST_SOUND_1_FILE)
        .rms(decimals=3)
    )
    print(rms_list)
    ref_value = 1.0
    assert abs(rms_list[0] - ref_value) < ctf.ABS_ERR


def test_aug_chain_split():
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

    mcs = Mcs().gen([300], ctf.SIGNAL_TIME_LEN, ctf.FS)
    ao = Aug(mcs)
    ao.splt(5).get().wr(ctf.TEST_SOUND_1_FILE)
    channels = ao.get().channels_count()
    assert channels == 5
    ref_value = 0.707
    rms_list = ao.get().rms(decimals=3)
    print(rms_list)
    for i in range(0, channels):
        assert abs(rms_list[i] - ref_value) < ctf.ABS_ERR


def test_aug_chain_side_by_side():
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

    test_sound_1 = Mcs().generate([300], ctf.SIGNAL_TIME_LEN, ctf.FS)
    mcs = Mcs().gen([1000], ctf.SIGNAL_TIME_LEN, ctf.FS)

    ao = Aug(mcs)
    rms_list = (
        ao.amp([0.3])
        .sbs(test_sound_1)
        .get().wr(ctf.TEST_SOUND_1_FILE)
        .rms(decimals=3)
    )
    print(rms_list)
    ref_value = [0.212, 0.707]
    for rms_list, ref in zip(rms_list, ref_value):
        print(rms_list)
        assert abs(rms_list - ref) < ctf.ABS_ERR

def test_aug_pause_detect():
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
    ao = Aug(test_sound_1)
    mask = ao.pause_detect([0.5, 0.3])
    ao.side_by_side(mask)
    print(test_sound_1)
    ao.get().write(ctf.TEST_SOUND_1_FILE)
    rms_list = ao.get().rms(decimals=3)
    ref_rms_list = [0.707, 0.707, 0.865, 0.923]
    for rms_value, ref in zip(rms_list, ref_rms_list):
        print(rms_value)
        assert abs(rms_value - ref) < ctf.ABS_ERR


def test_aug_chain_pause_detect():
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
    mcs.gen([100, 400], ctf.SIGNAL_TIME_LEN, ctf.FS)
    ao = Aug(mcs)
    mask = ao.pdt([0.5, 0.3])
    ao.sbs(mask).get().wr(ctf.TEST_SOUND_1_FILE)
    rms_list = ao.get().rms(decimals=3)
    ref_rms_list = [0.707, 0.707, 0.865, 0.923]
    for i, rms_value in enumerate(rms_list):
        print(rms_value)
        assert abs(rms_value - ref_rms_list[i]) < ctf.ABS_ERR


def test_aug_pause_shrink_sine():
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

    test_sound_1 = ms.Mcs().generate([100, 400], ctf.SIGNAL_TIME_LEN, ctf.FS)
    ao = Aug(test_sound_1)
    mask = test_sound_1.pause_detect([0.5, 0.3])
    res = test_sound_1.copy()
    res.side_by_side(mask)
    print(res)
    ao.pause_shrink(mask, [20, 4])
    ao.get().write(ctf.TEST_SOUND_1_FILE)
    rms_list = ao.get().rms(decimals=3)
    ref_rms_list = [0.702, 0.706, 0.865, 0.923]
    for rms_value, ref_rms_value in zip(rms_list, ref_rms_list):
        print(rms_value)
        assert abs(rms_value - ref_rms_value) < ctf.ABS_ERR


def test_aug_pause_shrink_speech():
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

    test_sound_1 = ms.Mcs(seed=42)
    test_sound_1.generate(
        [100, 300], ctf.SIGNAL_TIME_LEN, ctf.FS, mode="speech"
    )
    ao = Aug(test_sound_1)
    mask = test_sound_1.pause_detect([0.5, 0.3])
    res = test_sound_1.copy()
    res.side_by_side(mask)
    res.write(ctf.TEST_SOUND_1_FILE)
    ao.pause_shrink(mask, [20, 4])
    rms_list = ao.get().rms(decimals=3)
    ref_rms_list = [0.331, 0.324]
    for rms_value, ref_value in zip(rms_list, ref_rms_list):
        print(rms_value)
        assert abs(rms_value - ref_value) < ctf.ABS_ERR


def test_aug_pause_set():
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

    test_sound_1 = ms.Mcs(seed=42).generate(
        [100, 300], 0.003, ctf.FS, mode="speech"
    )

    ao = Aug(test_sound_1)
    mask = ao.pause_detect([0.5, 0.3])
    pause_list = ms.pause_measure(mask)
    ao.pause_set(pause_list, [10, 150])
    res = ao.get().copy()
    assert res.shape() == (2, 1618)
    res.write(ctf.TEST_SOUND_1_FILE)
    rms_list = res.rms(decimals=3)
    ref_rms_list = [0.105, 0.113]
    for r_value, ref_value in zip(rms_list, ref_rms_list):
        print(r_value)
        assert abs(r_value - ref_value) < ctf.ABS_ERR


def test_aug_chain_add_chain():
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
    mcs = ms.Mcs(samp_rt=ctf.FS).gen([1000, 300], 5)
    mcs_1 = ms.Mcs(mcs.data, mcs.sample_rate).gen([700, 100], 5)  # Create a Mcs instance
     
    ao = Aug(mcs)
    ao_1 = Aug(mcs_1)
    
    # Define the first chain command
    chain_1 = "amp([0.3, 0.2]).get().rms(decimals=3)"
    # Define the second chain command
    chain_2 = "amp([0.15, 0.1]).get().rms(decimals=3)"
    ao_1.achn([chain_1, chain_2])  # Add the chain commands to the chains list
    print(chain_1)  # Print the first chain command
    print(chain_2)  # Print the second chain command
    rms_list = ao_1.eval()  # Evaluate the chains
    print("rms list:", rms_list)  # Print the result
    #ref_rms_list = [[0.212], [0.106]]  # Define the expected values
    ref_rms_list = [[0.212], [0.032]]  # Define the expected values
    # Compare the result to the expected values
    for rms_value, ref_rms_value in zip(rms_list, ref_rms_list):
        print(rms_value)  # Print the result
        # Assert that the result is within the expected tolerance
        assert abs(rms_value[0] - ref_rms_value[0]) < ctf.ABS_ERR

    
def test_aug_pause_set():
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

    test_sound_1 = Mcs(seed=42).generate(
        [100, 300], 0.003, ctf.FS, mode="speech"
    )
    mask = test_sound_1.pause_detect([0.5, 0.3])
    pause_list = ms.pause_measure(mask)
    aug = Aug(test_sound_1)  
    aug.pause_set(pause_list, [10, 150])
    # res = test_sound_1.copy()
    res = aug.get().copy()
    assert res.shape() == (2, 1618)
    res.write(ctf.TEST_SOUND_1_FILE)
    rms_list = res.rms(decimals=3)
    ref_rms_list = [0.105, 0.113]
    for r_value, ref_value in zip(rms_list, ref_rms_list):
        print(r_value)
        assert abs(r_value - ref_value) < ctf.ABS_ERR
