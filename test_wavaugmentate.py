import numpy as np
import os
import  string
import subprocess as sp
import wavaugmentate as wau

fs = 44100
t = 5
f_list = [400, 1000, 2333, 3700]  # Frequencies list.

# Output files names.
test_sound_1_file = "./test_sounds/test_sound_1.wav"
test_sound_1_ac_file = "./test_sounds/test_sound_1_ac.wav"
test_sound_1_delay_file = "./test_sounds/test_sound_1_delay.wav"
test_sound_1_echo_file = "./test_sounds/test_sound_1_echo.wav"
test_sound_1_noise_file = "./test_sounds/test_sound_1_noise.wav"

output_file = './outputwav/out.wav'
prog = './' + wau.prog_name + '.py'
subst_table = str.maketrans({' ': None, '\n': None, '\t': None, '\r': None})


def test_mcs_generate():
    """
    Test function to verify the shape of the multichannel sound generated by
    the `mcs_generate` function.

    This function calls the `mcs_generate` function from the `ma` module with
    the given `f_list`, `t`, and `fs` parameters.  It then asserts that the
    shape of the generated `test_sound_1` is equal to `(4, 220500)`.

    Parameters:
        None

    Returns:
        None
    """
    test_sound_1 = wau.mcs_generate(f_list, t, fs)
    assert test_sound_1.shape == (4, 220500)
    rms_list = np.round(wau.mcs_rms(test_sound_1), decimals=3, out=None)
    for r in rms_list:
        assert abs(r - 0.707) < 0.001


def test_mcs_write():
    """
    Test function to verify the functionality of the `mcs_write` function.

    This function calls the `mcs_write` function from the `ma` module with the
    given `test_sound_1_file`, `test_sound_1`, and `fs` parameters.  It first
    generates a multichannel sound using the `mcs_generate` function and then
    writes it to a file using the `mcs_write` function.
    
    Parameters:
        None

    Returns:
        None
    """
    if os.path.exists(test_sound_1_file):
        os.remove(test_sound_1_file)
    test_sound_1 = wau.mcs_generate(f_list, t, fs)
    wau.mcs_write(test_sound_1_file, test_sound_1, fs)


def test_mcs_read():
    """
    Test function to verify the functionality of the `mcs_read` function.

    This function calls the `mcs_read` function from the `ma` module with the
    given `test_sound_1_file` parameter.  It then asserts that the sample rate
    of the read sound is equal to `fs` and the shape of the read multichannel
    sound is equal to `(4, 220500)`.

    Parameters:
        None

    Returns:
        None
    """
    test_sound_1 = wau.mcs_generate(f_list, t, fs)
    test_rs, test_mcs = wau.mcs_read(test_sound_1_file)
    assert test_rs == fs
    assert test_mcs.shape == (4, 220500)
    assert np.array_equal(test_mcs, test_sound_1)


def test_mcs_file_info():
    """
    Test function to verify the functionality of the `mcs_file_info` function.

    This function calls the `mcs_file_info` function from the `ma` module with
    the given `test_sound_1_file` parameter.  It then asserts that the path,
    channels count, sample rate, and length of the file are correct.

    Parameters:
        None

    Returns:
        None
    """

    info = wau.mcs_file_info(test_sound_1_file)
    assert info['path'] == test_sound_1_file
    assert info['channels_count'] == 4
    assert info['sample_rate'] == 44100
    assert info['length_s'] == 5.0


def test_mcs_amplitude_control():
    """
    Test function to verify the functionality of the `mcs_amplitude_control`
    function.

    This function generates a multichannel sound using the `mcs_generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies amplitude control to the generated sound using
    the `mcs_amplitude_control` function from the `ma` module with the given
    `test_sound_1` and `amplitude_list` parameters.  It asserts that the shape
    of the amplified multichannel sound is equal to `(4, 220500)`.  It writes
    the amplified multichannel sound to a file using the `mcs_write` function
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

    test_sound_1 = wau.mcs_generate(f_list, t, fs)
    amplitude_list = [0.1, 0.2, 0.3, 0.4]
    test_ac = wau.mcs_amplitude_control(test_sound_1, amplitude_list)
    assert test_ac.shape == (4, 220500)
    wau.mcs_write(test_sound_1_ac_file, test_ac, fs)

    for a, sig, coef in zip(test_ac, test_sound_1, amplitude_list):
        assert np.array_equal(a, sig * coef)


def test_mcs_delay_control():
    """
    Test function to verify the functionality of the `mcs_delay_control`
    function.

    This function generates a multichannel sound using the `mcs_generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies delay control to the generated sound using the
    `mcs_delay_control` function from the `ma` module with the given
    `test_sound_1` and `delay_list` parameters.  It asserts that the shape of
    the delayed multichannel sound is equal to `(4, 220500)`.  It writes the
    delayed multichannel sound to a file using the `mcs_write` function from 
    the `ma` module with the given file path and `fs` parameters.

    Parameters:
        None

    Returns:
        None
    """

    test_sound_1 = wau.mcs_generate(f_list, t, fs)
    delay_list = [100, 200, 300, 0]
    test_dc = wau.mcs_delay_control(test_sound_1, delay_list)
    assert test_dc.shape == (4, 220513)
    wau.mcs_write(test_sound_1_delay_file, test_dc, fs)
    for ch in test_dc:
        assert ch.shape[0] == 220513
    rms_list = np.round(wau.mcs_rms(test_dc, last_index=24), decimals=3, out=None)
    reference_list = [0.511, 0.627, 0.445, 0.705]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_mcs_echo_control():
    """
    Test function to verify the functionality of the `mcs_echo_control`
    function.

    This function generates a multichannel sound using the `mcs_generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies echo control to the generated sound using the
    `mcs_echo_control` function from the `ma` module with the given
    `test_sound_1`, `delay_us_list`, and `amplitude_list` parameters.  It writes
    the echoed multichannel sound to a file using the `mcs_write` function from
    the `ma` module with the given file path and `fs` parameters.  Finally, it
    calculates the root mean square (RMS) values of the echoed sound and
    compares them to the expected values in the `reference_list`.

    Parameters:
        None

    Returns:
        None
    """

    test_sound_1 = wau.mcs_generate(f_list, t, fs)
    delay_list = [1E6, 2E6, 3E6, 0]
    amplitude_list = [-0.3, -0.4, -0.5, 0]
    test_ec = wau.mcs_echo_control(test_sound_1, delay_list, amplitude_list, fs)
    wau.mcs_write(test_sound_1_echo_file, test_ec, fs)
    rms_list = np.round(wau.mcs_rms(test_ec), decimals=3, out=None)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001

   


def test_mcs_noise_control():
    """
    Test function to verify the functionality of the `mcs_noise_control`
    function.

    This function generates a multichannel sound using the `mcs_generate`
    function from the `ma` module with the given `f_list`, `t`, and `fs`
    parameters.  It then applies noise control to the generated sound using the
    `mcs_noise_control` function from the `ma` module with the given
    `test_sound_1`, `noise_level_list`, and `fs` parameters.  It writes the
    noise-controlled multichannel sound to a file using the `mcs_write` 
    from the `ma` module with the given file path and `fs` parameters. Finally,
    it calculates the root mean square (RMS) values of the noise-controlled
    sound and compares them to the expected values in the `reference_list`.

    Parameters:
        None

    Returns:
        None
    """

    test_sound_1 = wau.mcs_generate(f_list, t, fs)
    test_nc = wau.mcs_noise_control(test_sound_1, [0.01, 0.02, 0.03, 0],
                                    fs, seed=42)
    wau.mcs_write(test_sound_1_noise_file, test_nc, fs)
    rms_list = np.round(wau.mcs_rms(test_nc), decimals=3, out=None)
    reference_list = [0.776, 0.952, 1.192, 0.707]
    for r, ref in zip(rms_list, reference_list):
        # Threshold increased, because noise is not repeatable with fixed seed.
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
    cmd = [prog, '-i', test_sound_1_file, '-o', output_file, '-a',
           '0.5, 0.6, 0.7, 0.1']
    print('\n', ' '.join(cmd))
    if os.path.exists(output_file):
        os.remove(output_file)
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = s.translate(subst_table)
    print('out:', out)
    full_ref = '\namplitudes: [0.5, 0.6, 0.7, 0.1]\nDone.\n'
    ref = full_ref.translate(subst_table)
    print('ref:', ref)
    assert out == ref
    assert os.path.exists(output_file)
    _, written_data = wau.mcs_read(output_file)
    for ch in written_data:
        assert ch.shape[0] == 220500
    rms_list = np.round(wau.mcs_rms(written_data), decimals=3, out=None)
    print('rms_list:', rms_list)
    reference_list = [0.354, 0.424, 0.495, 0.071]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_wavaugmentate_amplitude_option_fail_case1():
    cmd = [prog, '-i', test_sound_1_file, '-o', output_file, '-a',
           '0.1, abc, 0.3, 0.4']
    print('\n', ' '.join(cmd))
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = s.translate(subst_table)
    print('out:', out)
    full_ref = 'Error: Amplitude list contains non number element: < abc>.'
    ref = full_ref.translate(subst_table)
    print('ref:', ref)
    assert out == ref


def test_wavaugmentate_amplitude_option_fail_case2():
    cmd = [prog, '-i', test_sound_1_file, '-o', output_file, '-a',
           '0.1, 0.3, 0.4']
    print('\n', ' '.join(cmd))
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = s.translate(subst_table)
    print('out:', out)
    full_ref = '\namplitudes: [0.1, 0.3, 0.4]\n\
Error: Amplitude list length <3> does not match number of channels. It should have <4> elements.\n'
    ref = full_ref.translate(subst_table)
    print('ref:', ref)
    assert out == ref


def test_wavaugmentate_delay_option():
    cmd = [prog, '-i', test_sound_1_file, '-o', output_file, '-d',
           "100, 200, 300, 0"]
    print('\n', ' '.join(cmd))
    if os.path.exists(output_file):
        os.remove(output_file)
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = s.translate(subst_table)
    print('out:', out)
    full_ref = '\ndelays: [100, 200, 300, 0]\nDone.\n'
    assert res.stdout == full_ref
    assert os.path.exists(output_file)
    ref = full_ref.translate(subst_table)
    print('ref:', ref)
    assert out == ref
    assert os.path.exists(output_file)
    _, written_data = wau.mcs_read(output_file)
    for ch in written_data:
        assert ch.shape[0] == 220513
    rms_list = np.round(wau.mcs_rms(written_data), decimals=3, out=None)
    print('rms_list:', rms_list)
    reference_list = [0.707, 0.707, 0.707, 0.707]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_wavaugmentate_delay_option_fail_case1():
    cmd = [prog, '-i', test_sound_1_file, '-o', output_file, '-d',
           '100, 389.1, 999, 456']
    print('\n', ' '.join(cmd))
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = s.translate(subst_table)
    print('out:', out)
    full_ref = 'Error: Delays list contains non integer element: <389.1>.\n'
    ref = full_ref.translate(subst_table)
    print('ref:', ref)
    assert out == ref


def test_wavaugmentate_delay_option_fail_case2():
    cmd = [prog, '-i', test_sound_1_file, '-o', output_file, '-d',
           '100, 200, 300']
    print('\n', ' '.join(cmd))
    res = sp.run(cmd, capture_output=True, text=True)
    s = str(res.stdout)
    out = s.translate(subst_table)
    print('out:', out)
    full_ref = '\ndelays: [100, 200, 300]\n\
Error: Delays list length <3> does not match number of channels. It should have <4> elements.\n'
    ref = full_ref.translate(subst_table)
    print('ref:', ref)
    assert out == ref


def test_WavaugPipeline_controls():

    test_sound_1 = wau.mcs_generate(f_list, t, fs)
    w = wau.WavaugPipeline(test_sound_1)
    print(w.data)

    w.amp([0.1, 0.3, 0.4, 1]).dly([100, 200, 300, 0])
    res1 = w.data

    w.put(np.zeros(1))
    res2 = w.put(test_sound_1).amp([0.1, 0.3, 0.4, 1]).dly([100, 200, 300, 0]).get()
    print(res2)
    assert np.array_equal(res1, res2)


def test_WavaugPipeline_wr_rd():

    w = wau.WavaugPipeline()
    if os.path.exists(test_sound_1_file):
        os.remove(test_sound_1_file)
    w.gen(f_list, t, fs).wr(test_sound_1_file)

    r = wau.WavaugPipeline()
    r.rd(test_sound_1_file)

    assert np.array_equal(w.data, r.data)


def test_WavaugPipeline_echo():
    d_list = [1E6, 2E6, 3E6, 0]
    a_list = [-0.3, -0.4, -0.5, 0]

    w = wau.WavaugPipeline()
    w.gen(f_list, t, fs).echo(d_list, a_list)
    rms_list = np.round(wau.mcs_rms(w.data), decimals=3, out=None)
    reference_list = [0.437, 0.461, 0.515, 0.559]
    for r, ref in zip(rms_list, reference_list):
        assert abs(r - ref) < 0.001


def test_WavaugPipeline_noise():
    n_list = [0.01, 0.02, 0.03, 0]
    w = wau.WavaugPipeline()
    w.gen(f_list, t, fs).ns(n_list, seed=42)
    rms_list = np.round(w.rms(), decimals=3, out=None)
    reference_list = [0.776, 0.952, 1.192, 0.707]
    for r, ref in zip(rms_list, reference_list):
        # Threshold increased, because noise is not repeatable with fixed seed.
        assert abs(r - ref) < 0.01