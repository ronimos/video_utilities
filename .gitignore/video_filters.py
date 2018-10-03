# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:08:34 2018

@author: Ron Simenhois
"""
import numpy as np
import cv2
import scipy.signal
import scipy.fftpack
import pywt

class Filters:
    def __init__(self, video, fps):
        self.video = video
        self.fps = fps
        self.video_data = None
    
    @classmethod
    def gaussian_filter(cls, video, pyramid_levels=3):
        """
        This function creates a NumPy array of down size pyramids of 
        all the video frames. The down size pyramids use a Gaussian 
        fill when down size. This fuction update the self object's 
        video_data - the filtered video 
        ---------------------------------------------------------
        Params:
            self - the video array to filter is in the self object
            pyramid_levels (int) - The number of down pyramids levels
        return:
            None
        """
        v_length = video.shape[0]
        for idx in range(v_length):
            gauss_frame = np.ndarray(shape=video[idx].shape, dtype=np.float)
            gauss_frame = video[idx].copy()
            for i in range(pyramid_levels):
                gauss_frame = cv2.pyrDown(gauss_frame)
            if idx==0:
                video_data = np.ndarray(shape=(v_length,) + gauss_frame.shape, dtype=np.float)
            video_data[idx]=gauss_frame
        return video_data
    
    @classmethod    
    def laplacian_filter(cls, video, pyramid_levels=3):
        """
        This function creates a NumPy array of the laplacian video 
        frames after doing down size pyramid. This fuction update 
        the self object's 
        video_data - the filtered video 
        ---------------------------------------------------------
        Params:
            self - the video array to filter is in the self object
            pyramid_levels (int) - The number of down pyramids levels
        return:
            a tensor list with a laplacian pyramid for every frame
        """
        def build_laplacian_pyramid(frame, pyramid_levels):
            gaussian_pyramid = [frame.copy()]
            for  i in range(pyramid_levels):
                frame = cv2.pyrDown(frame)
                gaussian_pyramid.append(frame)   
                
            laplacian_pyramid = []
            for i in range(pyramid_levels, 0, -1):
                gaussian_frame = cv2.pyrUp(gaussian_pyramid[i])
                laplacian_frame = cv2.subtract(gaussian_pyramid[i-1],gaussian_frame)
                laplacian_pyramid.append(laplacian_frame)
            return laplacian_pyramid
            
        v_length = video.shape[0]
        
        laplacian_tensor_list=[]
        for idx in range(v_length):
            frame = video[idx]
            pyr = build_laplacian_pyramid(frame, pyramid_levels)
            if idx==0:                
                for i in range(pyramid_levels):
                    laplacian_tensor_list.append(np.zeros((v_length, pyr[i].shape[0], pyr[i].shape[1], 3)))
            for i in range(pyramid_levels):
                laplacian_tensor_list[i][idx] = pyr[i]

        return laplacian_tensor_list
        
        
    @classmethod
    def butter_bandpass_filter(cls, fps, pyr_level_data, lowcut=None, highcut=None, order=5):
        """
        A helper fuction to calculate a and b for the Ifilter
        in the butter_bandpass_filter fuction
        ------------------------------------------------------------
        Params:
            self - the video array to filter is in the self object
            lowcut - A scalar giving the critical frequencies. 
                    For a Butterworth filter. The lowcut / (0.5 x FPS) is the point at which 
                    the gain drops to 1/sqrt(2). 
            highcut - A scalar giving the critical frequencies. 
                    For a Butterworth filter. The highcut / (0.5 x FPS) is the point at which 
                    the gain drops to 1/sqrt(2). 
            order (int) - the filter order
        return:
            Filteres signal
        """
        if lowcut==None: lowcut = 0.045 * fps
        if highcut==None: highcut = 0.1 * fps        
        nyq = 0.5 * fps
        low = lowcut / nyq # Cutoff frequency
        high = highcut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype='band')   
                
        return scipy.signal.lfilter(b, a, pyr_level_data, axis=0)
            
                    
    @classmethod
    def temporal_bandpass_filter(cls, video_data, fps, freq_min=None, freq_max=None, num_dominant_freq=2, axis=0):
        """
        Builds Fourier Transform presentation for each "pixel" in the video's 
        Gaussian presentation.
        ------------------------------------------------------------
        Params:
            video_data - (numpy array - float) Gaussian presentation of the original video
            fps - (int) Original videos frames per second 
            freq_min - (float) low boundry frequancy for the fft.
            freq_max - (float) High 
            axis - 
        return:
            the imaginary portion of the ourier Transform presentation
        """
        v_length = video_data.shape[0]
        data = np.concatenate([video_data, video_data[::-1]]*4, axis=0)
        
        frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0/fps)
        fft_signal = scipy.fftpack.fft(data, axis=axis)
        
        if freq_max is None or freq_min is None:
            # Find the dominante frequancies
            power = np.abs(fft_signal)
            pos_mask = (frequencies>0)
            pos_freqs = frequencies[pos_mask]
            peak_freq = pos_freqs[power[pos_mask].argmax()]
            k_peak_freq = pos_freqs[np.argpartition(power[pos_mask], -num_dominant_freq)[-num_dominant_freq]]
            freq_min=min(peak_freq, k_peak_freq)
            freq_max=max(peak_freq, k_peak_freq)
            
        fft_signal[np.abs(frequencies)>freq_max] = 0
        fft_signal[np.abs(frequencies)<freq_min] = 0
        
        return np.real(scipy.fftpack.ifft(fft_signal, axis=0)[:v_length])
    @classmethod
    def wavelets_transform(cls, data):
        scale = np.arange(0, data.shape[0])
        coef, freq = pywt.cwt(data, scale, 'morl')
        
    def wavelets_plot(cls, coef):
        import matplotlib.pyplot as plt
        plt.figure(1, figsize=(12,12))
        plt.imshow(coef, camp='coolwarm', aspect='auto')
        
    def wavelets_plot_3D(cls, data, coef):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(1,1,1, projection='3d')
        x = np.arange(0,data.shape[0], 1)
        y = np.arange(1,129)
        x,y = np.meshgrid(x,y)
        
        ax.plot_surface(x,y, coef, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax.set_xlabel('Time', fontsize=20)
        ax.set_ylabel('Scale', fontsize=20)
        ax.set_zlabel('Amplitude', fontsize=20)
    
    

        