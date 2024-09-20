"""Module providing test functions for wavaugmentate.py  module."""
import os
import sys
import mcs as ms
import wavaug as wau
# sys.path.append('/home/igor/Plastov/wavaugmentate/src/wavaugmentate')
sys.path.append(os.path.join(os.path.dirname(os.path.dirname( \
    os.path.dirname(__file__))), "/src/wavaugmentate"))


FS = ms.DEF_FS
SIGNAL_TIME_LEN = 5
f_list = [400, 1000, 2333, 3700]  # Frequencies list.
frm_list = [60, 140, 230, 300]  # Speech formants list.

# Output files names.
TESTS_DIR = "./tests/wavaugmentate/"
TEST_SOUND_1_FILE = TESTS_DIR + "test_sounds/test_sound_1.wav"
TEST_SOUND_1_AC_FILE = TESTS_DIR + "test_sounds/test_sound_1_ac.wav"
TEST_SOUND_1_DELAY_FILE = TESTS_DIR + "test_sounds/test_sound_1_delay.wav"
TEST_SOUND_1_ECHO_FILE = TESTS_DIR + "test_sounds/test_sound_1_echo.wav"
TEST_SOUND_1_NOISE_FILE = TESTS_DIR + "test_sounds/test_sound_1_noise.wav"

OUTPUTWAV_DIR = TESTS_DIR + "outputwav/"
OUTPUT_FILE = OUTPUTWAV_DIR + "out.wav"

SRC_DIR = "./src/wavaugmentate/"
PROG_NAME = SRC_DIR + wau.prog_name + ".py"

ABS_ERR = 0.0001


def shrink(text_for_shink: str):
    """Drops white spaces, newlines, and tabs from a string."""

    subst_table = str.maketrans(
        {" ": None, "\n": None, "\t": None, "\r": None}
    )
    return text_for_shink.translate(subst_table)
