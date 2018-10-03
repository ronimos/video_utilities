# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:21:31 2018

@author: Ron Simenhois
"""

from video_utills import VideoUtills
from video_magnify import MagnifyVideo
'''
A short example of how to load a video and get it prep for magnefication,
magnify and save
'''

video_obj = VideoUtills()
video_obj.load_video()

# Stabilize the video for color magnification:
video_obj.video_stabilizer()
video_obj.get_roi()

video_magnifier = MagnifyVideo(video_obj)
# Magnify canges in color

magnified = video_magnifier.video_magnify_color()

# Magnify changes in motion:
magnified = video_magnifier.video_magnify_motion(amplify=20)

# Save the video:
VideoUtills._save_video(video=magnified, fps=30)


'''
A short example of how to take a video with the computer camera and 
magnify it
'''

my_video = video_obj = VideoUtills.get_video_from_camera()
video_magnifyer = MagnifyVideo(my_video)
magnfied_video = video_magnifier.video_magnify_color()
VideoUtills.save_video(magnfied_video, fps=30)




