Usage
=====

Installation
------------

To use wavaugmentate, first install it using pip:

.. code-block:: console

    $ pip install wavaugmentate

As Console Application
----------------------

To change time delay between channels:

.. code-block:: console

    $ ./wavaugmentate.py -i ./test_sounds/test_sound_1.wav -o ./outputwav/out.wav -d "100, 200, 300, 400"


To change amplitudes of channels:

.. code-block:: console

    $ ./wavaugmentate.py -i ./outputwav/out.wav -o ./outputwav/out.wav -a "0.1, 0.2, 0.3, 0.4"


To apply sequence of augmentations just chain steps one by one:

.. code-block:: console

    $ ./wavaugmentate.py -c 'rd("./test_sounds/test_sound_1.wav").dly([100, 200, 300, 400]).amp([0.1, 0.2, 0.3, 0.4]).wr("./outputwav/sound_delayed.wav")'    


In Python Code
--------------

Augmentation step by step, Example 1:

.. code-block:: python
  
    from wavaugmentate.mcs import Mcs
    from wavaugmentate.aug import Aug

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
    

The same code as chain of operations, Example 2:

.. code-block:: python

    from wavaugmentate.mcs import Mcs
    from wavaugmentate.aug import Aug

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

 
How to get several augmented amplitudes and delays from single file.


Example 3 (single file augmentation):

.. code-block:: python

    from wavaugmentate.mcs import Mcs
    from wavaugmentate.aug import Aug

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

As Console Application
----------------------
use for help:

.. code-block:: console
   
    ./wavaug.py -h


command line interface  provides the same functionality.

Example 4 (procedural approach):

.. code-block:: console

    ./wavaug.py -i ./test_sounds/test_sound_1.wav -o ./outputwav/out.wav -d "100, 200, 300, 400"
    ./wavaug.py -i ./outputwav/out.wav -o ./outputwav/out.wav -a "0.1, 0.2, 0.3, 0.4"



Example 5 (OOP approach):

.. code-block:: console

    ./wavaug.py -c 'rd("./test_sounds/test_sound_1.wav").dly([100, 200, 300, 400]).amp([0.1, 0.2, 0.3, 0.4]).wr("./outputwav/sound_delayed.wav")'

