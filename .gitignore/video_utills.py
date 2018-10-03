# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 08:57:32 2018

@author: Ron Simenhois
"""
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

AUTO_MODE = 30
MANUAL_MODE = 0


class VideoUtills:
    
    def __init__(self, file_name=''):
        
        self.file_name = file_name
        self.drawing = False
        self.x = 0
        self.y = 0
        self.ix = 0
        self.iy = 0
        self.roi = dict(x1=self.x, y1=self.y, x2=self.ix, y2=self.iy)
        self.roi_set = False
               
    def load_video(self):
        """
        Invokes a TK ask for file name window to get a video file name 
        and loads a video file in a numpy array
        --------------------------------------------------------------
            params: self
            return: None
        """
        if self.file_name=='':
            Tk().withdraw()
            self.file_name = askopenfilename()
        cap = cv2.VideoCapture(self.file_name)
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = int(cap.get(cv2.CAP_PROP_FPS))
        
        video_buffer = np.ndarray(shape=(self.length, self.heigth, self.width, 3), dtype=np.uint8)
        for i in range(self.length):
            if not cap.isOpened():
                break
            ret, frame = cap.read()
            video_buffer[i, ...] = frame
        assert(i==self.length-1)
        self.video_buffer = video_buffer
        cap.release()
        self.ix = self.width-1
        self.iy = self.heigth-1 
        self.roi = dict(x1=self.x, y1=self.y, x2=self.ix, y2=self.iy)
        return video_buffer
    
    
    def __set_roi(self, event, x, y, flags, params):
        """
        A mouse callback function that draws a rectangle on a video frame 
        and save its corners as ROI for future analasys
        --------------------------------------------------------------
            params: self
                    (int) event: mouse event (left click up, down, mouse move...)
                    (int) x, y: mouse location
                    flags: Specific condition whenever a mouse event occurs.
                            EVENT_FLAG_LBUTTON
                            EVENT_FLAG_RBUTTON
                            EVENT_FLAG_MBUTTON
                            EVENT_FLAG_CTRLKEY
                            EVENT_FLAG_SHIFTKEY
                            EVENT_FLAG_ALTKEY
                    params: user specific parameters (not used)
            return: None
        """
        if event==cv2.EVENT_LBUTTONDOWN:
            self.ix = x
            self.iy = y
            self.drawing = True
        if event==cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.x = x
                self.y = y
                self.img = self.current_frame.copy()
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y,), (0, 0, 255), 3)
        if event==cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def dummy_mouse_call_back_func(self, event, x, y, flags, params):
        pass
    
    def play_video(self, video=None, window_name=None, mode=MANUAL_MODE, mouse_callback_func=None):
        """
        Play the video. Also alows ROI set.
        --------------------------------------------------------------
            params: self
                    (int) mode: MANUAL_MODE - 0 for flipping through the frame manualy
                                AUTO_MODE - 30 for aoutomatic play 30 ms between frames
            return: None
        """
        if mouse_callback_func==None:
            mouse_callback_func = self.dummy_mouse_call_back_func
        zoom = 1
        if video is None:
            video = self.video_buffer
        if window_name==None:
            window_name = 'Video Player'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback_func)
        idx=0
        self.current_frame = video[idx]
        self.img = self.current_frame.copy()
        while True:
            cv2.imshow(window_name, self.img)
            k = cv2.waitKey(30) & 0xFF
            if mode==MANUAL_MODE:
                if k==ord('f'):
                    idx = min(idx+1, self.length-1)
                    self.current_frame = cv2.resize(video[idx], None, fx=zoom, fy=zoom, \
                                                    interpolation=cv2.INTER_AREA)
                    self.img = self.current_frame.copy()
                if k==ord('b'):
                    idx = max(0, idx-1)
                    self.current_frame = cv2.resize(video[idx], None, fx=zoom, fy=zoom, \
                                                    interpolation=cv2.INTER_AREA)
                    self.img = self.current_frame.copy()
                if k==ord('i'):
                    zoom *= 1.33
                    self.current_frame = cv2.resize(video[idx], None, fx=zoom, fy=zoom, \
                                                    interpolation=cv2.INTER_AREA)
                    self.img = self.current_frame.copy()
                if k==ord('o'):
                    zoom *= 0.75
                    self.current_frame = cv2.resize(video[idx], None, fx=zoom, fy=zoom, \
                                                    interpolation=cv2.INTER_AREA)
                    self.img = self.current_frame.copy()

                if k==27:
                    break
        cv2.destroyAllWindows()
        self.roi = dict(x1=min(self.x, self.ix), x2=max(self.x, self.ix),
                        y1=min(self.y, self.iy), y2=max(self.y, self.iy))
        
    def get_roi(self, video=None, window_name=None):
        """
        An interface function that get the video ROI 
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            save_file: (str) saved video file location and name
        """
        if video is None:
            video = self.video_buffer
        self.play_video(video=video, window_name=window_name, mouse_callback_func=self.__set_roi)
        self.roi_set = True
        return self.roi
        
    def save_video(self, video=None, fps=None):
        """
        Opens a tkinter save file dialog to get save file name
        and save a video under the given name
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            save_file: (str) saved video file location and name
        """
        
        if video is None:
            video=self.video_buffer
        if fps==None:
            fps = self.fps
        Tk().withdraw()
        save_file = asksaveasfilename(defaultextension=".mp4")
        if save_file==None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename=save_file, fourcc=fourcc, fps=fps, 
                              frameSize=(self.width, self.heigth),isColor=True)
        for frame in video:
            out.write(frame)
        out.release()
        print('Video file is saved')
        return save_file
    
    @classmethod
    def _save_video(cls, video=None, fps=30):
        """
        Opens a tkinter save file dialog to get save file name
        and save a video under the given name
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            save_file: (str) saved video file location and name
        """
        
        if video is None:
            return
        length, heigth, width, ch = video.shape
        
        Tk().withdraw()
        save_file = asksaveasfilename(defaultextension=".mp4")
        if save_file==None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename=save_file, fourcc=fourcc, fps=fps, 
                              frameSize=(width, heigth),isColor=True)
        for frame in video:
            out.write(frame)
        out.release()
        print('Video file is saved')
        return save_file

        
    def trim_video(self, video=None):
        """
        Trim a video to between two specific frames
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            trimed_video: (np.array) the trimmed video
        """
        if video is None:
            video = self.video_buffer
        start = 0
        end = self.length-1
        window = 'Click "f" for next frame, "b" for back frame, "s & e" for trimmes video start and end' 
        idx = 0
        while True:
            cv2.imshow(window, video[idx])
            k = cv2.waitKey(0) & 0xFF
            if k==ord('f'):
                idx = min(idx+1, self.length-1)
            if k==ord('b'):
                idx = max(0, idx-1)
            if k==ord('s'):
                start = idx
            if k==ord('e'):
                end = idx
            if k==27:
                break
            cv2.destroyAllWindows()
        return video[start: end, ...]
    
    def crop_video(self, video=None):
        """
        Crops all frames in a video to draw ROI
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            croped video: (np.array) 
        """

        if video is None:
            video = self.video_buffer
        self.get_roi(video=video, window_name='Darw the ROI on any frame in the video and click Esc')
        roi = self.roi
        return video[:, roi['x1']: roi['x2'], roi['y1']: roi['y2'], :]
    
    def video_stabilizer(self, video=None):
        """
        Stabilize a video around object inside a marked ROI
        This function uses a LK oprical flow to calculate 
        movments and affin matrix to adjust the frames
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
        return:
            croped video: (np.array) 
        """
        
        if video is None:
            video = self.video_buffer
        stab_video = np.zeros_like(video, dtpye=np.uint8)
        roi = self.get_roi(video=video, window_name='Draw ROI to stabilize the video around it')

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=5,
                              blockSize=7)
    
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=8,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        m_dx, m_dy = 0, 0
    
        # Take first frame and find corners in it
        old_frame = video[0]
    
        rows, cols, depth = old_frame.shape
        old_roi = old_frame[roi['x1']: roi['x2'], roi['y1']: roi['y2']]
        old_gray = cv2.cvtColor(old_roi, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        p_start = p0.copy()
        
        for idx in range(self.length):
    
            # Get next frame
            frame = self.video_buffer[idx]
            roi = frame[roi['x1']: roi['x2'], roi['y1']: roi['y2']]
            frame_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # if the number of good features is < original fertues / 2 get new features
            if p1[np.where(st == 1)].shape[0] <= p_start.shape[0]/2:
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                p_start = p0.copy()
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            try:
                good_cur = p1[np.where(st == 1)]
                good_old = p0[np.where(st == 1)]
            except TypeError as e:
                print('TypeError, no good points are avaliabole, error: {0}'.format(e))
                print('Exit video stabilizer at frame {0} out of {1}'.format(idx, self.length))
                break
            dx = []
            dy = []  
    
            # Draw points and calculate
            for i, (cur, old) in enumerate(zip(good_cur, good_old)):
                a, b = cur.ravel()
                c, d = old.ravel()
                dx.append(c - a)
                dy.append(d - b)
    
            m_dx += np.mean(dx)
            m_dy += np.mean(dy)
    
            M = np.float32([[1, 0, m_dx], [0, 1, m_dy]])
            
            stab_video[:] = cv2.warpAffine(frame, M, (cols, rows), 
                                           cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP, 
                                           cv2.BORDER_CONSTANT).copy()

            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_cur.reshape(-1, 1, 2)
        
        return stab_video
    
    def track_motion(self, video=None, set_roi=False):
        """
        Track and displays the motion of items in a marked ROI. 
        This function uses a LK oprical flow to calculate 
        movments.
        ---------------------------------------------------------
        Params:
            self
            video (np.array) - numpy array that contains the video
            set_roi (bool) - 
        return:
            track_video: (np.array) - the numpy array of the video with 
                                      the motion of the traked items highlited
            motion_tracker: (list) - A list of numpy arrays with the location 
                                     of the traked items for each video frame                            
        """
        
        if video is None:
            video = self.video_buffer
        if set_roi:
            roi = self.get_roi(video=video)
      
        video_track = video.copy()
        motion_tracker = []
        # Generate different colors for tracking display 
        color = np.random.randint(0,255,(100,3))
                       
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 5,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 8,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        old_gray = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
        # Create mask for drawing
        mask = np.zeros_like(video[0])
        # Mask to dectate the features to track
        features_mask = np.zeros_like(old_gray)
        features_mask[roi['x1']: roi['x2'], roi['y1']: roi['y2']] = old_gray[roi['x1']: roi['x2'], roi['y1']: roi['y2']]
        # Find corners in first frame
        p0 = cv2.goodFeaturesToTrack(features_mask, mask = None, **feature_params)
        
        for idx in range(1, video.shape[0]):
            new_gray = cv2.cvtColor(video[idx], cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
            
            # Get good points
            good_old = p0[st==1]
            good_new = p1[st==1]
            motion_tracker.append(good_new)
            for i, (old, new) in enumerate(zip(good_old, good_new)):
                (ox, oy) = old.reval()
                (nx, ny) = new.ravel()
                mask = cv2.circle(mask, (nx, ny), 5, color[i].tolist(), -1)
            frame = cv2.add(video[idx], mask)
            video_track[idx] = frame
            
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF==27:
                break
            # Updat old frames and points before checking next frame
            
            old_gray = new_gray.copy()
            p0 = p1.resapr(-1,1,2)
        cv2.destroyAllWindows()
        
        return video_track, motion_tracker
    
    @classmethod
    def get_video_from_camera(cls):
        """
        Take a video for a computer camra and fave it as a mp4 file
        ---------------------------------------------------------
        Params:
        return:
            save_file: (str) - file name of the saved file
        """
        cap = cv2.VideoCapture(0)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = 30
        save_file = None
        
        Tk().withdraw()
        save_file = asksaveasfilename(defaultextension=".mp4")
        
        if save_file!=None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename=save_file, fourcc=fourcc, fps=fps, 
                                  frameSize=(width, heigth),isColor=True)
            
            while cap.isOpened():
                ret, frame = cap.read()
                cv2.imshow('Play video', frame)
                out.write(frame)
                if cv2.waitKey(30)==27:
                    break
            cv2.destroyAllWindows()
            out.release()
            cap.release()
        return save_file
    
    @classmethod
    def play_any_video(cls, video):
        
        zoom = 1.0
        window = 'i - zoom in, o - zoom out, f - forward, b - backward, Esc to exit'
        idx = 0
        while True:
            frame = cv2.resize(video[idx], fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
            cv2.imshow(window, frame)
            k = cv2.waitKey(0)
            if k==ord('i'):
                zoom *= 1.25
            if k==ord('o'):
                zoom /= 1.25
            if k==ord('f'):
                idx = min(idx+1, video.shape[0])
            if k==ord('b'):
                idx = max(idx-1, 0)
            if k==27:
                break
            
        
if __name__=='__main__':   
    VideoUtills.take_video()
        
      
        







            
            
        

        
        

        
            
            