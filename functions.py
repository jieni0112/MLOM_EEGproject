#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:45:30 2020

@author: jenny
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from PIL import Image
import os
from scipy.interpolate import griddata # cubic 2D uses CloughTocher

"""
Function bandpass filter and extract the three frequency band data 
input: panda dataframe 20000 x 65
first column of data is the time
output: 3 matrices of filtered data
"""
def band_pass_filter(df):
    
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    #print(raw.info)
    fs = 160 # sampling frequency
    #print(df.shape)
    matrix_size = df.shape
    
    
    # define frequency bands
    mu_low = 8;
    mu_high = 13;
    beta_low = 13;
    beta_high = 30;
    delta_low = 0.5;
    delta_high = 4;
    
    # bandpasss data
    mu = np.empty(matrix_size)
    beta = np.empty(matrix_size) 
    delta = np.empty(matrix_size)
    
    # store data in new dataframe
    for i in range(64):
        channel_i = df.iloc[:,i+1];
        mu[:,i+1] = butter_bandpass_filter(channel_i, mu_low, mu_high, fs, order=3)
        beta[:,i+1] = butter_bandpass_filter(channel_i, beta_low, beta_high, fs, order=3)
        delta[:,i+1] = butter_bandpass_filter(channel_i, delta_low, delta_high, fs, order=3)
        
    # mu, beta and delta are filtered signals respectively for df, time series
    t = df.iloc[:,0]
    mu[:,0] = t
    beta[:,0] = t
    delta[:,0] = t
    
    return mu, beta, delta

"""
produces 30 images for 1 subject, 1/14 runs
function input: df, annot_from_file, mu, beta, delta, subject number, run number to write to,
    and list of labels
output: saves 30 images in png form, and returns the updated list of labels
"""
def image_creation(df, annot_from_file, mu, beta, delta, subject, run, labels_list):
    
    # create cumulative list
    annot = np.multiply(annot_from_file['duration'],1000)
    cumu = np.array([])
    for i in range(len(annot)):
        cumu = np.append(cumu, np.sum(annot[0:i]))
    
    image_pixels = np.empty([10,64,3]) # image
    pixel_value = np.empty([1,3])
    
    # convert subject and run number to string for folder
    s = str(subject)
    r = str(run)
    
    # create the folders
    if os.path.exists("images/S" + s) is False:
        os.mkdir("images/S" + s)
    
    if os.path.exists("images/S" + s + "/R" + r) is False:
        os.mkdir("images/S" + s + "/R" + r)
    
    folder = "images/S" + s + "/R" + r + "/"
    
    for k in range(len(cumu)):
        offset = int(df[df['time']==cumu[k]].index.values)
        #print(offset)
        for i in range(64): # for 64 electrodes:      
            for j in range(10): # 10 pixels for 1 image (4s)
                # 400ms window
                # sample freq = 160Hz so 400ms = 64 datapoints
                
                pixel_data = np.array([])
                pixel_data = mu[offset + 64*j : offset + 64*(j+1), i+1]# first column is time
                # fft
                pixel_fft = np.fft.rfft(pixel_data)
                # pixel value calculation - sum of squared abs value
                pixel_value[0,0] = np.sum(np.square(np.abs(pixel_fft))) 
                #pixels_mu[j,i] = np.sum(np.square(np.abs(pixel_fft))) 
                # same for beta
                pixel_data = np.array([])
                pixel_data = beta[offset + 64*j : offset + 64*(j+1), i+1]# first column is time
                pixel_fft = np.fft.rfft(pixel_data)
                pixel_value[0,1] = np.sum(np.square(np.abs(pixel_fft))) 
                #pixels_beta[j,i] = np.sum(np.square(np.abs(pixel_fft)))
                
                # delta
                pixel_data = np.array([])
                pixel_data = delta[offset + 64*j : offset + 64*(j+1), i+1]# first column is time
                pixel_fft = np.fft.rfft(pixel_data)
                pixel_value[0,2] = np.sum(np.square(np.abs(pixel_fft)))
                #pixels_delta[j,i] = np.sum(np.square(np.abs(pixel_fft)))
                
                image_pixels[j,i] = pixel_value # j is the 10 windows (rows)
                # i is the 64 electrodes (columns)
                
        
        
        # combine 3 frequencies to form 1 image
        # Convert the pixels into an array using numpy
        new_image = Image.fromarray(image_pixels.astype(np.uint8))
        
        # assign label to the image:
        label = annot_from_file.loc[k,'description']
        #print(label)
        # add this label to the list of labels
        labels_list.append(label)
        
        # Use PIL to create an image from the new array of pixels
        new_image.save(folder + "image" + "S" + subject + "R" + run + "N" +  str(k+1) + ".png")
        
        # return labels list
        return labels_list
    
  
"""
produces 30 x 10 images for 1 subject for 1/6 selected
function input: df_time, annot_from_file, locations, mu, beta, delta, subject number, run number to write to,
    and list of labels
output: saves 300 images in png form, and returns the updated list of labels
"""
def Interpolate_image(df_time, annot_from_file, locations, mu, beta, delta, subject, run, labels_list):
    
    # create cumulative list
    annot = np.multiply(annot_from_file['duration'],1000)
    cumu = np.array([])
    for i in range(len(annot)):
        cumu = np.append(cumu, np.sum(annot[0:i]))
    
    # convert subject and run number to string for folder
    s = str(subject)
    r = str(run)
    
    # create the folders
    if os.path.exists("images_int/S" + s) is False:
        os.mkdir("images_int/S" + s)
    
    if os.path.exists("images_int/S" + s + "/R" + r) is False:
        os.mkdir("images_int/S" + s + "/R" + r)
    
    folder = "images_int/S" + s + "/R" + r + "/"
    
    # define temporary storage
    z = np.empty([64,3])
    pixel_value = np.empty([1,3])
    
    for k in range(len(cumu)):
        offset = int(df_time[df_time==cumu[k]].index.values)
        #offset = int(np.where(df_time == cumu[k]))
        #print(offset)
        for j in range(10): # each label 4s    
            for i in range(64): # for 64 electrodes:  
                # 400ms window
                # sample freq = 160Hz so 400ms = 64 datapoints
                
                pixel_data = np.array([])
                pixel_data = mu[offset + 64*j : offset + 64*(j+1), i+1]# first column is time
                # fft
                pixel_fft = np.fft.rfft(pixel_data)
                # pixel value calculation - sum of squared abs value
                pixel_value[0,0] = np.sum(np.square(np.abs(pixel_fft))) 
                #pixels_mu[j,i] = np.sum(np.square(np.abs(pixel_fft))) 
                # same for beta
                pixel_data = np.array([])
                pixel_data = beta[offset + 64*j : offset + 64*(j+1), i+1]# first column is time
                pixel_fft = np.fft.rfft(pixel_data)
                pixel_value[0,1] = np.sum(np.square(np.abs(pixel_fft))) 
                #pixels_beta[j,i] = np.sum(np.square(np.abs(pixel_fft)))
                
                # delta
                pixel_data = np.array([])
                pixel_data = delta[offset + 64*j : offset + 64*(j+1), i+1]# first column is time
                pixel_fft = np.fft.rfft(pixel_data)
                pixel_value[0,2] = np.sum(np.square(np.abs(pixel_fft)))
                #pixels_delta[j,i] = np.sum(np.square(np.abs(pixel_fft)))
                
                z[i,:] = pixel_value # pixel value is array of rgb values
                # i is the 64 electrodes (columns)
                
            x = locations.iloc[:,1]
            y = locations.iloc[:,2]
            xv, yv = np.mgrid[-0.75:0.75:32j, -0.75:0.75:32j] # 32 x 32 image
            axes = np.transpose([x,y])
            #interpolation
            grid_zR = griddata(axes, z[:,0], (xv, yv), method='cubic') # mu band
            grid_zG = griddata(axes, z[:,1], (xv, yv), method='cubic') # beta band
            grid_zB = griddata(axes, z[:,2], (xv, yv), method='cubic') # delta band
            
            #stack image
            image = np.dstack([grid_zR, grid_zG, grid_zB])
            
            # combine 3 frequencies to form 1 image
            # Convert the pixels into an array using numpy
            new_image = Image.fromarray(image.astype(np.uint8))
            
            # assign label to the image:
            label = annot_from_file.loc[k,'description']
            #print(label)
            # add this label to the list of labels
            labels_list.append(label)
            
            # Use PIL to create an image from the new array of pixels
            # L is out of 30, 10 x N for 1L have same label
            new_image.save(folder + "image" + "S" + str(subject) + "R" + str(run) 
                           + "L" + str(k+1) + "N" +  str(j+1) + ".png")
            
    # return labels list
    return labels_list

"""
hanning window from https://github.com/tevisgehr/EEG-Classification/blob/master/train_pipeline_5_no_overlap.py

"""
def theta_alpha_beta_averages(f,Y):
    theta_range = (0.5,4)
    alpha_range = (8,13)
    beta_range = (13,30)
    theta = Y[(f>theta_range[0]) & (f<=theta_range[1])].mean()
    alpha = Y[(f>alpha_range[0]) & (f<=alpha_range[1])].mean()
    beta = Y[(f>beta_range[0]) & (f<=beta_range[1])].mean()
    return theta, alpha, beta


"""
fft adopted from https://github.com/tevisgehr/EEG-Classification/blob/master/train_pipeline_5_no_overlap.py
"""
def get_fft(snippet):
    Fs = 160.0;  # sampling rate
    #Ts = len(snippet)/Fs/Fs; # sampling interval
    snippet_time = len(snippet)/Fs
    Ts = 1.0/Fs; # sampling interval
    #t = np.arange(0,snippet_time,Ts) # time vector

    # ff = 5;   # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    y = snippet
#     print('Ts: ',Ts)
#     print(t)
#     print(y.shape)
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    #Added in: (To remove bias.)
    #Y[0] = 0
    return frq,abs(Y)

"""
produces 30 x 10 images for 1 subject for 1/6 selected using a different fft implementation
function input: df_time, annot_from_file, locations, mu, beta, delta, subject number, run number to write to,
    and list of labels
output: saves 300 images in png form, and returns the updated list of labels
"""
def Interpolate_image2(df_time, annot_from_file, locations, mu, beta, delta, subject, run, labels_list):
    mu_range = (0.5,4)
    delta_range = (8,13)
    beta_range = (13,30)
    # create cumulative list
    annot = np.multiply(annot_from_file['duration'],1000)
    cumu = np.array([])
    for i in range(len(annot)):
        cumu = np.append(cumu, np.sum(annot[0:i]))
    
    # convert subject and run number to string for folder
    s = str(subject)
    r = str(run)
    
    """
    # create the folders
    if os.path.exists("images_int2/S" + s) is False:
        os.mkdir("images_int2/S" + s)
    
    if os.path.exists("images_int2/S" + s + "/R" + r) is False:
        os.mkdir("images_int2/S" + s + "/R" + r)
    
    folder = "images_int2/S" + s + "/R" + r + "/"
    """
    # define temporary storage
    z = np.empty([64,3])
    pixel_value = np.empty([1,3])
    
    for k in range(len(cumu)):
        offset = int(df_time[df_time==cumu[k]].index.values)
        #offset = int(np.where(df_time == cumu[k]))
        #print(offset)
        for j in range(10): # each label 4s    
            for i in range(64): # for 64 electrodes:  
                # 400ms window
                # sample freq = 160Hz so 400ms = 64 datapoints
                
                pixel_data = np.array([])
                pixel_data = mu[offset + 64*j : offset + 64*(j+1), i+1]# first column is time
                # fft
                #pixel_fft = np.fft.rfft(pixel_data)
                pixel_fft =  get_fft(pixel_data)
                # pixel value calculation - sum of squared abs value
                pixel_value[0,0] = np.sum(np.square(np.abs(pixel_fft))) 
                #pixels_mu[j,i] = np.sum(np.square(np.abs(pixel_fft))) 
                # same for beta
                pixel_data = np.array([])
                pixel_data = beta[offset + 64*j : offset + 64*(j+1), i+1]# first column is time
                #pixel_fft = np.fft.rfft(pixel_data)
                pixel_fft =  get_fft(pixel_data)
                pixel_value[0,1] = np.sum(np.square(np.abs(pixel_fft))) 
                #pixels_beta[j,i] = np.sum(np.square(np.abs(pixel_fft)))
                
                # delta
                pixel_data = np.array([])
                pixel_data = delta[offset + 64*j : offset + 64*(j+1), i+1]# first column is time
                #pixel_fft = np.fft.rfft(pixel_data)
                pixel_fft =  get_fft(pixel_data)
                pixel_value[0,2] = np.sum(np.square(np.abs(pixel_fft)))
                #pixels_delta[j,i] = np.sum(np.square(np.abs(pixel_fft)))
                
                z[i,:] = pixel_value # pixel value is array of rgb values
                # i is the 64 electrodes (columns)
                
            x = locations.iloc[:,1]
            y = locations.iloc[:,2]
            xv, yv = np.mgrid[-0.75:0.75:32j, -0.75:0.75:32j] # 32 x 32 image
            axes = np.transpose([x,y])
            #interpolation
            grid_zR = griddata(axes, z[:,0], (xv, yv), method='cubic') # mu band
            grid_zG = griddata(axes, z[:,1], (xv, yv), method='cubic') # beta band
            grid_zB = griddata(axes, z[:,2], (xv, yv), method='cubic') # delta band
            
            #stack image
            image = np.dstack([grid_zR, grid_zG, grid_zB])
            
            # combine 3 frequencies to form 1 image
            # Convert the pixels into an array using numpy
            new_image = Image.fromarray(image.astype(np.uint8))
            
            # assign label to the image:
            label = annot_from_file.loc[k,'description']
            #print(label)
            # add this label to the list of labels
            labels_list.append(label)
            
            # Use PIL to create an image from the new array of pixels
            # L is out of 30, 10 x N for 1L have same label
            new_image.save("images_int2/" + "image" + "S" + str(subject) + "R" + str(run) 
                           + "L" + str(k+1) + "N" +  str(j+1) + ".png")
            
    # return labels list
    return labels_list

