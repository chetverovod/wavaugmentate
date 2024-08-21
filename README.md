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
wau.mcs_write('./sound_delayed.wav', res, fsmp)
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

