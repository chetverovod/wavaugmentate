# **Wavaugmentate**:  Multichannel Speech Signal Augmentation Processor

# Input Data

Multichannel WAV-file or NumPy array.
```
Array shape: (num_channels,num_samples).
```
# Output Data
Same types as in section [Input_data](#Input_data).

# Augmentation Methods 
1. Amplitude (volume change, inversion).
2. Time shift.
3. Echo.
4. Adding noise.
6. Time stretching. (**not implemented**)
7. Tempo change. (**not implemented**)
8. ​​Pitch shift. (**not implemented**)
9. Adding silence. (**not implemented**)
10. Frequency masking. (**not implemented**)
11. Time masking. (**not implemented**)
12. Combinations of methods.

# Interfaces
Signal augmentation can be applied by two ways:
1. Python module methods.
2. CLI interface options.

## Python Module

Example 1 (procedural approach):
```Python
import wavaugmentate as wau

# Read WAV-file to array.
fsmp, mcs = wau.read('./sound.wav')

# Apply delays.
delay_list = [100, 200, 300, 400]  # Corresponds to channels quantity. 
d = wau.delay_ctrl(mcs, delay_list)

# Apply amplitude changes.
amplitude_list = [0.1, 0.2, 0.3, 0.4]  # Corresponds to channels quantity. 
res = wau.amplitude_ctrl(d, amplitude_list)

# Augmentation result saving.
wau.write('./sound_delayed.wav', res, fsmp)
```
The same code as chain, Example 2 (OOP approach):

```Python
import wavaugmentate as wau

w = wau.WaChain()
w.rd('./sound.wav').dly([100, 200, 300, 400]).amp([0.1, 0.2, 0.3, 0.4]).wr('./sound_delayed.wav')
```
## CLI

use for details:
```
./wavaugmentate.py -h
```

command line interface  provides the same functionality.

Example 3 (procedural approach):
```shell
./wavaugmentate.py -i ./test_sounds/test_sound_1.wav -o ./outputwav/out.wav -d "100, 200, 300, 400"
./wavaugmentate.py -i ./outputwav/out.wav -o ./outputwav/out.wav -a "0.1, 0.2, 0.3, 0.4"

```

Example 4 (OOP approach):
```shell
./wavaugmentate.py -c 'rd("./sound.wav").dly([100, 200, 300, 400]).amp([0.1, 0.2, 0.3, 0.4]).wr("./sound_delayed.wav")'

```

# Unit Tests

Just run:
```shell
pytest
```

# Reference
MCS - multi channel signal, numpy array with shape (M_channels, N_samples).
| # |        Function        |       CLI option           |  Chain method   |        Description     |
|---|------------------------|----------------------------|-----------------|------------------------|
|1  | read(path)             | -c 'rd(path)'              | rd(path)        | Read MCS from WAV-file.|
|2  | write(path, mcs_data, fs)  | -c 'wr(path)'              | wr(path)        | Save MCS to WAV-file.  |
|3  | file_info(path)        | --info                     | info()          | Returns WAV-file info. |
|4  |        -               | -i path                    |  -              | Input WAV-file path.   |
|5  |        -               | -o path                    |  -              | Output WAV-file path.  |
|6  | amplitude_ctrl(mcs_data.[c1,c2..Cm])| -a "c1,c2..Cm"             | amp([c1,c2..Cm])| Change amplitudes of channels. |
|7  | delay_ctrl(mcs_data,[t1,t2..tm])    | -d "t1,t2..tm"             | dly([t1,t2..tm])| Add delays to channels.        |
|8  | echo _ctrl(mcs_data,[t1,t2..tm],[c1,c2..Cm])|-d "t1,t2..tm / c1,c2..Cm"|dly([t1,t2..tm],[c1,c2..Cm])|Add echo to channels. |
|9  | noise_ctrl(mcs_data,[c1,c2..Cm])| -n "c1,c2..Cm"             | ns([c1,c2..Cm]) | Add normal noise to channels. | 
|3  | copy(mcs_data)         | -                          | cpy(mcs_data)   | copy MCS. |
|3  | generate([f1,f2,f3 fm],duration,fs)|-|gen([f1,f2,f3 fm], duration, fs)|Creates MCS and generates sine signal for each channel.|
|3  | merge(mcs_data) | -|mrg(mcs_data)| Merges all channels to single and returns  mono MCS.|
|3  | pause_detect
|3  | rms(mcs_data) | -| rms() | Returns list of RMS calculated for channels.|
|3  | side_by_side(mcs_data1. mcs_data2)
|3  | split
|3  | sum

