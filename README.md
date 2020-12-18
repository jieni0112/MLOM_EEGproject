# MLOM_EEGproject
EEG motor imagery classification project using tensorflow to be deployed onto STM32

The python files are for preprocessing EEG data according to [1].
The notebook files are the networks and tflite conversion code.

The data are bandpass filtered, split into time windows and transformed into frequency data by fft. The according to electrode projections in 2D, a 32x32 image with RGB channels is formed using Clough-Tocher interpolating algorithm.

Networks inspired from various sources are trained and testing on the dataset.

Then the trained network is converted into tflite model.



References:
[1] W. Fadel, C. Kollod, M. Wahdow, Y. Ibrahim and I. Ulbert, "Multi-Class Classification of Motor Imagery EEG Signals Using Image-Based Deep Recurrent Convolutional Neural Network," 2020 8th International Winter Conference on Brain-Computer Interface (BCI), Gangwon, Korea (South), 2020, pp. 1-4, doi: 10.1109/BCI48061.2020.9061622.
[2] Lun, X., Yu, Z., Chen, T., Wang, F. and Hou, Y., 2020. A Simplified CNN Classification Method for MI-EEG via the Electrode Pairs Signals. Frontiers in Human Neuroscience, 14.
[3] Yimin Hou et al 2020 J. Neural Eng. 17 016048
[4] SuperBruceJia, EEG-DL, (2020), GitHub repository
[5] tevisgehr, EEG-classification (2017), GitHub repository

