# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:21:58 2018

@author: Ron Simenhois
"""

import cv2
import numpy as np
from video_utills import VideoUtills
from video_filters import Filters
import matplotlib.pyplot as plt

class MagnifyVideo:
    
    def __init__(self, video_obj=None):
        if video_obj==None:
            video_obj = VideoUtills()
        self.video = video_obj.load_video()
        if video_obj.roi_set:
            self.roi = video_obj.roi
        else:
            self.roi = video_obj.get_roi()
        self.fps = video_obj.fps
    
    
    def video_magnify_color(self, amplify=20.0, pyramid_levels=3, roi=None):
        '''
        Magnifies color changes by getting down pyramid Gaussian presentation of the 
        video and multiply the Furrier Transform presentation of the Gaussian 
        presentation video.
        ---------------------------------------------------------
        Params:
            amplify - (float)  
            pyramid_levels - (int) 
            roi - (dict) the area to amplfy in the form of:
                         {'x1':start column, 'x2': end column, 'y1': start row, 'y2': end row}
        return:
            magnified_video - (munpy array) A video array with the roi section magnfied
        '''
        if roi==None:
            roi = self.roi
        video = self.video
        roi_video = video[:, roi['y1']: roi['y2'], roi['x1']: roi['x2'],:].copy()
        video_data = Filters.gaussian_filter(roi_video, pyramid_levels)
        video_data = Filters.temporal_bandpass_filter(video_data, self.fps)
        video_data *= amplify
        magnified_video = self.insert_magmify_and_reconstract(video_data=video_data, pyramid_levels=pyramid_levels)
        return magnified_video


    def insert_magmify_and_reconstract(self, video_data, pyramid_levels):
        '''
        Reconstruct a video numpy array back from an amplfied Gaussian presentation of 
        the video and insert the amplfy section (roi) of each frame into the video 
        ---------------------------------------------------------
        Params:
            video_data - (numpy array) - a down pyramid of Gaussian presenvation of 
                          the video (in roi)
            pyramid_levels - (int) 
        return:
            magnified_video - (munpy array) A video array
        '''
        
        magnified_video = self.video.copy()
        length, width, heigth, channels = magnified_video.shape

        for idx in range(length):
            frame_data = video_data[idx].copy()
            for _ in range(pyramid_levels):
                frame_data = cv2.pyrUp(frame_data)
            if idx==0:
                video_roi = magnified_video[:, self.roi['y1']: self.roi['y1'] + frame_data.shape[0], \
                                  self.roi['x1']: self.roi['x1'] +  frame_data.shape[1]]
            video_roi[idx] = np.clip(video_roi[idx] + frame_data, 0, 255).astype(np.uint8)
        return magnified_video
    

    def video_magnify_motion(self, low=0.4, high=3.0, amplify=20, pyramid_levels=3):
        '''
        Magnifies motion changes by getting a list of down pyramids Laplacian 
        presentation of the video and multiply the changes between the frames 
        pyramids after butter bandpass filtering by the amplify coefficient.
        ---------------------------------------------------------
        Params:
            amplify - (float)  
            pyramid_levels - (int) 
            low - (float) low end number for the Butter banpass filter
            high - (float) high end number for the Butter banpass filter
        return:
            magnified_video - (munpy array) A video array
        '''
        
        video = self.video
        lap_video_list = Filters.laplacian_filter(video, pyramid_levels)
        filtered_frames_list=[]
        for i in range(pyramid_levels):
            filter_array=Filters.butter_bandpass_filter(self.fps, lap_video_list[i] , low, high)
            filter_array*=amplify
            filtered_frames_list.append(filter_array)
        reconsted = self.reconstruct_from_filter_frames_list(filtered_frames_list, pyramid_levels)
        amplify_video = np.clip(self.video + reconsted, 0, 255).astype(np.uint8)
        return amplify_video
    
    def reconstruct_from_filter_frames_list(self, video_pyramids_list, pyramid_levels=3):
        '''
        Reconstruct a video numpy array back from a list of down pyramids if the 
        video after amply 
        ---------------------------------------------------------
        Params:
            video_pyramids_list - (list of numpy array) - each array is a down 
            pyramid of magnfied laplacian images of different levels  
            pyramid_levels - (int) 
        return:
            magnified_video - (munpy array) A video array
        '''
        length = video_pyramids_list[0].shape[0]
        magnified_video = np.zeros_like(video_pyramids_list[-1])
        for idx in range(length):   
            up_pyr = video_pyramids_list[0][idx]
            for pyr_level in range(pyramid_levels-1):
                up_pyr = cv2.pyrUp(up_pyr) + video_pyramids_list[pyr_level + 1][idx]
            magnified_video [idx] = up_pyr
        return magnified_video

       
    def get_sub_roi(self, n_splits):
        
        roi = self.roi
        xedges = np.linspace(roi['x1'], roi['x2'], n_splits, stype=int)
        yedges = np.linspace(roi['y1'], roi['y2'], n_splits, stype=int)
        for x1, x2 in zip(xedges[:-1], xedges[1:]):
            for y1, y2 in zip(yedges[:-1], yedges[1:]):
                yield dict(x1=x1, x2=x2, y1=y1, y2=y2)

    
    def video_magnify_sub_roi(self, n_splits):
        
        for sub_roi in self.get_sub_roi:
            self.video_magnify(roi=sub_roi)
            
    def chart_pixel_intesity(self):
        
        video = self.video[1:]
        roi = self.roi
        
        fig, ax = plt.subplots(nrows=4, sharex=True)
        video_roi = video[:, roi['y1']: roi['y2'], roi['x1']: roi['x2'],:]
        means = video_roi.mean(axis=(1,2,3))
        ax[0].plot(means)
        ax[0].set_title('Original video average pixel intensity')
        
        video_data = Filters.gaussian_filter(video=video_roi)
        means = video_data.mean(axis=(1,2,3))
        ax[1].plot(means)
        ax[1].set_title('Gaussian video average pixel intensity')
        
        video_fft = Filters.temporal_bandpass_filter(video_data, self.fps)
        means = video_fft.mean(axis=(1,2,3))
        ax[2].plot(means)
        ax[2].set_title('Gaussian video average pixel intensity')

        plt.tight_layout()
            