"""Module providing test functions for wavaugmentate.py  module."""

import os
import sys
import logging as log
import tempfile
import mcs as ms
import wavaug as wau

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))), "/src/wavaugmentate"))


FS = ms.DEF_FS
SIGNAL_TIME_LEN = 5
freq_list = [400, 1000, 2333, 3700]  # Frequencies list.
frm_list = [60, 140, 230, 300]  # Speech formants list.

# Output files names.
TESTS_DIR = "./tests/wavaugmentate/"
OUTPUT_WAV_DIR = os.path.join(TESTS_DIR, "outputwav/")

SRC_DIR = "./src/wavaugmentate/"
PROG_NAME = os.path.join(SRC_DIR, f"{wau.prog_name}.py")


log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("wavaugmentate.log"),
        log.StreamHandler(),
    ],
)


ABS_ERR = 0.0001


def shrink(text_for_shrink: str):
    """Drops white spaces, newlines, and tabs from a string."""

    subst_table = str.maketrans(
        {" ": None, "\n": None, "\t": None, "\r": None}
    )
    return text_for_shrink.translate(subst_table)


def temp_ref_file_name() -> str:
    """Function creates temporary file name."""

    file_descriptor, temp_test_file_name = tempfile.mkstemp()
    os.close(file_descriptor)

    return temp_test_file_name


