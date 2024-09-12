wavaugmentate module
====================

.. automodule:: wavaugmentate 

.. autofunction:: augmentate


Command line interface options
------------------------------

Actual console options can be observed by command:

.. code-block:: console

    $ python wavaugmentate.py -h

Example of response:

.. code-block:: console

    usage: wavaugmentate [-h] [-i IN_PATH] [-o OUT_PATH] [--info] [--amp AMPLITUDE_LIST] [--echo ECHO_LIST]
                     [--dly DELAY_LIST] [--ns NOISE_LIST] [--chain CHAIN_CODE]

    WAV audio files augmentation utility.

    options:
    -h, --help            show this help message and exit
    -i IN_PATH            Input audio file path.
    -o OUT_PATH           Output audio file path.
    --info                Print info about input audio file.
    --amp AMPLITUDE_LIST, -a AMPLITUDE_LIST
                            Change amplitude (volume) of channels in audio file. Provide coefficients for every
                            channel, example: -a "0.1, 0.2, 0.3, -1"
    --echo ECHO_LIST, -e ECHO_LIST
                            Add echo to channels in audio file. of channels in audio file. Provide coefficients
                            and delays (in microseconds) of reflected signal for every channel, example: -e "0.1,
                            0.2, 0.3, -1 / 100, 200, 0, 300"
    --dly DELAY_LIST, -d DELAY_LIST
                            Add time delays to channels in audio file. Provide delay for every channel in
                            microseconds, example: -d "100, 200, 300, 0"
    --ns NOISE_LIST, -n NOISE_LIST
                            Add normal noise to channels in audio file. Provide coefficients for every channel,
                            example: -n "0.1, 0.2, 0.3, -1"
    --chain CHAIN_CODE, -c CHAIN_CODE
                            Execute chain of transformations. example: -c 'gen([100,250,100], 3, 44100).amp([0.1,
                            0.2, 0.3]).wr("./sines.wav")'

    Text at the bottom of help




