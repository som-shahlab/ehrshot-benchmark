This is a fork of the EHRSHOT repository for the sole purpose of running MOTOR on EHRSHOT, which requires a slightly different setup and version of FEMR.

https://huggingface.co/StanfordShahLab/motor-t-base is the model used for these experiments.

In particular, this code requires femr 0.1.16, as seen and explained in https://github.com/som-shahlab/motor_tutorial/blob/main/MOTOR_Tutorial_Notebook.ipynb

https://github.com/som-shahlab/motor_tutorial/blob/main/MOTOR_Tutorial_Notebook.ipynb also contains the instructions for downloading femr.

This fork only supports two commands, generate 4_generate_motor_features and 6_eval

It's recommended to use the cannonical EHRSHOT release code for everything else.

