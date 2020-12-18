#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:13:51 2020

@author: jenny
"""

# LIbraries
import os.path as op

import numpy as np
import pandas as pd

import mne
from mne.datasets import fetch_fsaverage

import functions as fn

# read data
root_path = 'files/'
subjects = np.arange(1,110)
# count the labels T0 T1 T2
T0_c = 0;
T1_c = 0;
T2_c = 0;

fs = 160 # sampling frequency

labels_list = []
undefined_annotation = []

# load electrode positions in 2D
locations = pd.read_csv('projection.csv')

for i in subjects:
    # exclude S043, S088, S089, S092, S100, S104
    if(i not in { 43, 88, 89, 92, 100, 104}):
        if (i < 10):
            folder = '/S00' + str(i);
        elif (i < 100):
            folder = '/S0' + str(i);
        else:
            folder = '/S' + str(i);
        #print(folder + folder + 'R10.edf')
        
        #for j in range(14):  
        for j in [4,6,8,10,12,14]: # only do for imaginary runs
            if (j < 10):
                filename = 'R0' + str(j) + '.edf'
            else:
                filename = 'R' + str(j) + '.edf'
            #print(folder + folder + filename)
            raw = mne.io.read_raw_edf(root_path + folder + folder + filename)
            #raw = mne.io.read_raw_edf(root_path + '/S001/' + 'S001R10.edf')
            raw.annotations.save('saved-annotations.csv')
            annot_from_file = pd.read_csv('saved-annotations.csv')
            
            # change to panda dataframe
            df = raw. to_data_frame(picks=None, index=None, scalings=None, copy=True, start=None, stop=None, long_format=False, time_format='ms')
            # separate into 3 bands:
            mu, beta, delta = fn.band_pass_filter(df)
            # create image
            # extract time data from df
            df_time = df["time"]
            
            #labels_list = fn.image_creation(df, annot_from_file, mu, beta, delta, i, j+1, labels_list)
            #labels_list = fn.Interpolate_image(df_time, annot_from_file, locations,
             #                                  mu, beta, delta, i, j, labels_list)
            labels_list = fn.Interpolate_image2(df_time, annot_from_file, locations,
                                               mu, beta, delta, i, j, labels_list)
            
            """
            for k in range(len(annot_from_file)):
                annotation = annot_from_file.loc[k,'description'];
                if (annotation == 'T0'):
                    T0_c += 1;
                elif (annotation == 'T1'):
                    T1_c += 1;
                elif (annotation == 'T2'):
                    T2_c += 1;
                else:
                    undefined_annotation.append(annotation)    
            """
#np.savetxt('labels.csv',labels_list,delimiter=',',fmt='%s')


