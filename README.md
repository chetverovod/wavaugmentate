![](https://img.shields.io/badge/os-Linux-green) ![](https://readthedocs.org/projects/wavaugmentate/badge/?version=latest) ![](https://img.shields.io/badge/python-3.9-blue)
[![Python application](https://github.com/chetverovod/wavaugmentate/actions/workflows/python-app.yml/badge.svg?branch=dev)](https://github.com/chetverovod/wavaugmentate/actions/workflows/python-app.yml)
[![Pylint](https://github.com/chetverovod/wavaugmentate/actions/workflows/pylint.yml/badge.svg)](https://github.com/chetverovod/wavaugmentate/actions/workflows/pylint.yml)
![](https://img.shields.io/badge/license-MIT-blue.svg)

# **Wavaugmentate** 0.2.5


**Multichannel Audio Signal Augmentation Module**

![alt text](./pictures/title_image.png)
The module makes audio signal augmentation conversions. It provides the *MultiChannelSignal*, *SignalAugmentation* classes and *wavaug-cli* console utility.
  - *MultiChannalSignal* provides basic operations with multi-channel signals.
  - *SignalAugmentation* helps to perform augmentation of multi-channel signals for AI models learning purpose. 

PyPi: https://pypi.org/project/wavaugmentate
GitHub: https://github.com/chetverovod/wavaugmentate

# Installation
```shell
pip install wavaugmentate
```

# Input Data

WAV-file or NumPy array.
```
Array shape: (num_channels, num_samples).
```
# Output Data
Same types as in section [Input_data](#Input_data).

# Augmentation Methods 
1. Amplitude (volume change, inversion).
1. Time shift.
1. Echo.
1. Adding noise.
1. Time stretching. (**not implemented**)
1. Tempo change. (**not implemented**)
1. Pitch shift. (**not implemented**)
1. Adding silence. 
1. Frequency masking. (**not implemented**)
1. Time masking. (**not implemented**)
1. Combinations of methods.

# Additional Functionality
1. Generation multichannel tonal signals of desired frequency, amplitude, durance.
2. Generation multichannel speech-like signals of desired formants frequency, amplitude, durance.

# Interfaces
Signal augmentation can be applied by two ways:
1. As python module *Mcs*, *Aug* classes methods.
2. As console application *wavaugmentate* with CLI interface options.

## Python Module

Example 1 (procedural approach):
```Python
from wavaugmentate.mcs import MultiChannelSignal as Mcs
from wavaugmentate.aug import SignalAugmentation as Aug


# File name of original sound.
file_name = "./outputwav/sound.wav"

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

```
Original signal is shown on picture:
![Initial signal](./pictures/example_1_original_signal.png)

Output signal with augmented data (channel 1 contains original signal without changes):
![Augmented signal](./pictures/example_1_augmented_signal.png)


The same code as a chain of operations, Example 2:

```Python

from wavaugmentate.mcs import MultiChannel as Mcs
from wavaugmentate.aug import SignalAugmentation as Aug

# File name of original sound.
file_name = "./outputwav/sound.wav"

delay_list = [0, 150, 200, 250, 300, 350, 400]
amplitude_list = [1, 0.17, 0.2, 0.23, 0.3, 0.37, 0.4]

# Apply all transformations of Example 1 in chain.
ao_obj = Aug(Mcs().rd(file_name))
ao_obj.splt(7).dly(delay_list).amp(amplitude_list).get().wr(
"sound_augmented_by_chain.wav"
)

# Augmentation result saving to 7 files, each 1 by channel.
ao_obj.get().wrbc("sound_augmented_by_chain.wav")
 
```
## CLI

Use for help:
```
wavaug-cli -h
```

command line interface  provides the same functionality.

Example 3 (procedural approach):
```shell
wavaug-cli -i ./test_sounds/test_sound_1.wav -o ./outputwav/out.wav -d "100, 200, 300, 400"
wavaug-cli -i ./outputwav/out.wav -o ./outputwav/out.wav -a "0.1, 0.2, 0.3, 0.4"

```

Example 4 (OOP approach):
```shell
wavaug-cli -c 'rd("./test_sounds/test_sound_1.wav").dly([100, 200, 300, 400]).amp([0.1, 0.2, 0.3, 0.4]).wr("./outputwav/sound_delayed.wav")'

```
 ## How To
 ### Single file to several augmented
 Amplitudes and delays will be augmented by  code shown in example 5.

 Example 5 (single file augmentation):
 ```Python
from wavaugmentate.mcs import MultiChannel as Mcs
from wavaugmentate.aug import SignalAugmentation as Aug

file_name = "./outputwav/sound.wav"
mcs = Mcs()
mcs.rd(file_name)  # Read original file with single channel.
file_name_head = "sound_augmented"

# Suppose we need 15 augmented files.
aug_count = 15
for i in range(aug_count):
    signal = Aug(mcs.copy())
    # Apply random amplitude [0.3..1.7) and delay [70..130)
    # microseconds changes to each copy of original signal.
    signal.amp([1], [0.7]).dly([100], [30])
    name = file_name_head + f"_{i + 1}.wav"
    signal.get().write(name)    
```

# Unit Tests

Just run:
```shell
export  PYTHONPATH='./src/wavaugmentate'
python3 -m pytest
```
# Tests Coverage

```
pytest --cov-report term-missing --cov
```

Example of result:
```
---------- coverage: platform linux, python 3.11.4-final-0 -----------
Name                       Stmts   Miss  Cover
----------------------------------------------
common_test_functions.py      15      0   100%
test_mcs_class.py            385      0   100%
test_wavaugmentate.py        293      0   100%
wavaugmentate.py             507     38    93%
----------------------------------------------
TOTAL                       1200     38    97%

```

# Reference
MCS - multi channel signal, it is NumPy array with shape (M_channels, N_samples).
| #|        *Mcs* class method        |            CLI option           |  Method alias   |     Description     |
|--|------------------------|---------------------------------|-----------------|------------------------|
|1 | read(path)             | -c 'rd(path)'              | rd        | Read MCS from WAV-file.|
|2 | write(path)            | -c 'wr(path)'              | wr        | Save MCS to WAV-file.  |
|3 | file_info(path)        | --info                     | info          | Returns WAV-file info. |
|4 |        -               | -i path                    |  -             | Input WAV-file path.   |
|5 |        -               | -o path                    |  -             | Output WAV-file path.  |
|6 | amplitude_ctrl([c1,c2..cm]) | -a "c1,c2..Cm"             | amp | Change amplitudes of channels. |
|7 | delay_ctrl([t1,t2..tm])    | -d "t1,t2..tm"             | dly | Add delays to channels.        |
|8 | echo _ctrl([t1,t2..tm],[c1,c2..cm]) |-d "t1,t2..tm / c1,c2..Cm"| echo |Add echo to channels. |
|9 | noise_ctrl([c1,c2..cm]) | -n "c1,c2..Cm"             | ns | Add normal noise to channels. | 
|10| copy         | -                          | cpy | Makes copy of MCS. |
|11| generate([f1,f2,f3..fm],duration,fs)|-| gen |Creates MCS and generates sine signal for each channel.|
|12| merge() | -| mrg | Merges all channels to single and returns  mono MCS.|
|13| pause_detect(relative_level)|-| pdt | Searches pauses by selected levels. Returns array-mask.|
|14| pause_set(pause_map,pause_sz) | - | - | Set pauses length to selected values. Returns updated MCS.|
|15| rms() | - | rms | Returns list of RMS calculated for object channels.|
|16| side_by_side(mcs) | - | sbs | Appends channels from mcs data as new channels.| 
|17| split(m_channels) | - | splt | Splits single channel to m_channels copies.|  
|18| sum(mcs2) | - | sum | Adds mcs2 data channels values to object channels data sample by sample. | 
|19| write_by_channel(path) | - | wrbc | Save MCS object channels to separate WAV-files.  |


## Documentation

[Documentation on the Read the Docs](https://wavaugmentate.readthedocs.io/en/latest/)


For local documentation make clone of repository and look html-version of documentation (docs/_build/html/index.html):
[html-documentation](docs/_build/html/index.html)

### Rebuild Documentation
```shell
cd docs
make html
``` 

# Build Package 

Install *builder*:

```shell
python3 -m pip install --upgrade build
```
build package:

```shell
python3 -m build
```
