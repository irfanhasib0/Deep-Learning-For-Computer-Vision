import time
import ffmpeg
import numpy as np
import cv2
from PIL import Image
from collections import deque
from matplotlib import pyplot as plt

class StreamCapture:
    def __init__(self,src,enable_audio=True):
        self.enable_audio=enable_audio
        args = {
        "fflags": "nobuffer",
        "flags": "low_delay",
        }
        if src.startswith('rtsp://'): 
            args.update({
                "rtsp_transport": "tcp",
                #"scale":"360:240",
                #rtsp_flags = 'listen',
                #"probesize":32,
                #"analyzeduration":0,
                #"sync":"ext",
                #"use_wallclock_as_timestamps" : "1",
                #"f" : "segment",
                #"segment_time" : "900",
                #"vcodec" : "copy",
                #"tune" : "zerolatency",
                #"crf" : "18"
            })
        
        probe    = ffmpeg.probe(src)
        self.video_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
        if self.enable_audio : 
            self.audio_info = next(x for x in probe['streams'] if x['codec_type'] == 'audio')
        #print("fps: {}".format(cap_info['r_frame_rate']))
        up, down = str(self.video_info['r_frame_rate']).split('/')
        fps      = eval(up) / eval(down)
        print(fps)
        self.frame_width    = 640#320 #cap_info['width']           
        self.frame_height   = 480#240 #cap_info['height'] 
        #self.frame_width    = self.video_info['width']           
        #self.frame_height   = self.video_info['height'] 
        
        
        #print("fps: {}".format(fps))    
        self.process1 = (
            ffmpeg
            .input(src,**args)
            .filter('scale',self.frame_width ,self.frame_height)
            .filter('fps',fps=10,round='up')
            #.filter('segment',segment_time=900)
            .output('pipe:',format='rawvideo', pix_fmt='rgb24')
            .overwrite_output()
            .run_async(pipe_stdout=True)
        )
        
        if self.enable_audio:
            self.process2 = (
                ffmpeg
                .input(src,**args)
                #.filter('aecho',0.8,0.9,1000,0.3)
                .output('pipe:', ar=10000, format='s16le')
                .overwrite_output()
                .run_async(pipe_stdout=True)
            )
        
        self.frame_queue = deque(maxlen=5)
        self.audio_queue = deque(maxlen=5)
        #self.start_reading()
        
    def start_reading(self):
        while 1:
            frame,audio = read()
            self.frame_queue.append(frame)
            self.audio_queue.append(audio)
            if not len(frame) :
                self.close()
                print('Done !!')
                break
            
            
    def read(self):
        #self.process1.stdout.flush()
        in_bytes    = self.process1.stdout.read(self.frame_width * self.frame_height * 3)
        if self.enable_audio:
            self.process2.stdout.flush()
            audio_bytes = self.process2.stdout.read(4096)
            audio_data  = np.frombuffer(audio_bytes,np.dtype('int16'))#.newbyteorder('<'))
            #print(audio_data.mean(),audio_data.shape)
        else:
            audio_data = np.array([0])
            
        if not in_bytes:
            return [],[]
        frame = np.frombuffer(in_bytes, np.uint8)
        frame = frame.reshape([self.frame_height, self.frame_width, 3])
        
        #frame = cv2.resize(in_frame, (1280, 720))   
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame,audio_data
    
    def get_curr_data(self):
        return self.frame_queue[-1],self.audio_queue[-1]
    
    def close(self):
        self.process1.kill() 
        self.process2.kill()
        
        
        
