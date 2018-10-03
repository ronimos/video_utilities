# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:36:16 2018

@author: Avalanche
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

COLORS = ['b', 'g', 'r']

def chart_averages(array, xbins=None, ybins=None):
    
    length, width, heigth = array.shape[:3]
    if xbins==None: xbins=width
    if ybins==None: ybins=heigth
    xbsize = width//xbins
    ybsize = heigth//ybins
    fig, ax = plt.subplots(nrows=ybins, ncols=xbins)#, sharex='all', sharey='all')
    for i, xedge in enumerate(range(0, width, xbsize)):
        for j, yedge in enumerate(range(0, heigth, ybsize)):
            average = array[:, xedge: xedge+xbsize, yedge: yedge+ybsize].mean(axis=1).mean(axis=1)
            for ch, color in enumerate(COLORS):
                ax[j, i].plot(average[:, ch], c=color)
    plt.subplots_adjust(wspace=0, hspace=0)
    

def chart_frequencies_response(signal, fps):
    
    signal = np.concatenate([signal, signal[::-1]]*2, axis=0)
    frequencies = scipy.fftpack.fftfreq(signal.shape[0], d=1/fps)
    f_signal = scipy.fftpack.fft(signal)
    plt.plot(frequencies, f_signal)
    
    
    
    
            
        
    
    
    