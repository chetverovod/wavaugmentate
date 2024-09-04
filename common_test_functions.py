"""Module providing test functions for wavaugmentate.py  module."""
import wavaugmentate as wau

FS = wau.DEF_FS
SIGNAL_TIME_LEN = 5
f_list = [400, 1000, 2333, 3700]  # Frequencies list.
frm_list = [60, 140, 230, 300]  # Speech formants list.

# Output files names.
TEST_SOUND_1_FILE = "./test_sounds/test_sound_1.wav"
TEST_SOUND_1_AC_FILE = "./test_sounds/test_sound_1_ac.wav"
TEST_SOUND_1_DELAY_FILE = "./test_sounds/test_sound_1_delay.wav"
TEST_SOUND_1_ECHO_FILE = "./test_sounds/test_sound_1_echo.wav"
TEST_SOUND_1_NOISE_FILE = "./test_sounds/test_sound_1_noise.wav"

OUTPUT_FILE = "./outputwav/out.wav"
PROG_NAME = "./" + wau.prog_name + ".py"

def shrink(s: str):
    """Drops white spaces, newlines, and tabs from a string."""

    subst_table = str.maketrans(
        {" ": None, "\n": None, "\t": None, "\r": None}
    )
    return s.translate(subst_table)
