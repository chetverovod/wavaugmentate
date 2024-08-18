import numpy as np
import pyplnoise
from audiostretchy.stretch import stretch_audio
from scipy.io import wavfile


def mcs_rms(mcs_data, last_index=-1):
    """Return RMS of multichannel sound."""

    res = []
    for signal in mcs_data:
        res.append(np.sqrt(np.mean(signal[0:last_index]**2)))
    return res


def mcs_generate(frequency_list, duration, sample_rate=44100):
    """Function generates multichannel sound as a set of sin-waves.

    return numpy array of shape [channels, samples]
    """

    samples = np.arange(duration * sample_rate) / sample_rate
    channels = []
    for f in frequency_list:
        signal = np.sin(2 * np.pi * f * samples)
        signal = np.float32(signal)
        channels.append(signal)
    multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def mcs_write(path, mcs_data, sample_rate=44100):
    """ Save multichannel sound to wav-file.

    path : string or open file handle
    Output wav file.

    sample_rate : int
    The sample rate (in samples/sec).
    """
    buf = mcs_data.T.copy()
    wavfile.write(path, sample_rate, buf)


def mcs_read(path):
    """ Read multichannel sound from wav-file.

    return sample_rate, mcs_data.
    """
    sample_rate, buf = wavfile.read(path)
    mcs_data = buf.T.copy()
    return sample_rate, mcs_data


def mcs_file_info(path):
    """ Return information about multichannel sound from wav-file.

    return path, channels_count, sample_rate, length (in seconds) of file.
    """
    sample_rate, buf = wavfile.read(path)
    length = buf.shape[0] / sample_rate

    return {"path": path, "channels_count": buf.shape[1], "sample_rate": sample_rate, "length_s": length}


# ## Функции аугментации речевого сигнала

# %% [markdown]
# 1. Амплитуда (изменение громкости)
# 2. Фазовый сдвиг
# 3. Эхо
# 4. Инверсия делается как амплитуда со знаком минус.
# 5. Добавление шума
# 6. Растяжение во времени
# 8.  Изменение темпа
# 9.  Сдвиг высоты тона
# 10. Добавление фонового шума
# 11. Добавление тишины
# 12  Частотное  маскирование
# 13. Временное маскирование
# 14. Комбинации этих методов

def mcs_amplitude_control(mcs_data, amplitude_list):
    """ Change amplitude of multichannel sound."""
    channels = [] 
    for signal, amplitude in zip(mcs_data, amplitude_list):
        channels.append(signal * amplitude)
        multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def mcs_delay_control(mcs_data, delay_us_list, sampling_rate=44100):
    """Add delays of channels of multichannel sound. Output data become longer."""

    channels = []
    max_samples_delay = int(max(delay_us_list) * 1.E-6 * sampling_rate)  # In samples.

    for signal, delay in zip(mcs_data, delay_us_list):
        samples_delay = int(delay * 1.E-6 * sampling_rate)  # In samples.
        res = np.zeros(samples_delay)
        res = np.append(res, signal)
        if samples_delay < max_samples_delay:
            res = np.append(res, np.zeros(max_samples_delay - samples_delay))
        channels.append(res)
        multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def mcs_echo_control(mcs_data, delay_us_list, amplitude_list, sampling_rate=44100):
    """Add echo to multichannel sound.

    Returns:
        Output data become longer.
    """
    a = mcs_amplitude_control(mcs_data, amplitude_list)
    e = mcs_delay_control(a, delay_us_list)
    channels = []
    for d in mcs_data:
        zl = e.shape[1] - d.shape[0]
        channels.append(np.append(d, np.zeros(zl)))
    multichannel_sound = np.array(channels).copy() + e

    return multichannel_sound


def mcs_noise_control(mcs_data, noise_level_list, sampling_rate=44100, seed=-1):
    """ Add pink noise to channels of multichannel sound."""

    channels = []
    for signal, level in zip(mcs_data, noise_level_list):
        if seed != -1:
            pknoise = pyplnoise.PinkNoise(sampling_rate, 1e-2, 50.)
        else:
            pknoise = pyplnoise.PinkNoise(sampling_rate, 1e-2, 50., seed=seed)
        noise = pknoise.get_series(signal.shape[0])
        res = signal + level * np.array(noise)
        channels.append(res)
    multichannel_sound = np.array(channels).copy()
    return multichannel_sound


def mcs_stratch_control(mcs_data, ratio_list, sampling_rate=44100):
    """Add pink noise to channels of multichannel sound."""
    stretch_audio("input.wav", "output.wav", ratio=1.1)
    channels = []
    for signal, ratio in zip(mcs_data, ratio_list):
        pknoise = pyplnoise.PinkNoise(sampling_rate, 1e-2, 50.)
        noise = pknoise.get_series(signal.shape[0])
        res = signal + ratio * np.array(noise)
        channels.append(res)
    multichannel_sound = np.array(channels).copy()
    return multichannel_sound
