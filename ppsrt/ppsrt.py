# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 01:03:34 2016

@author: PRABLANC
"""

from pyaudio import PyAudio, paContinue, paFloat32
import readchar
import ProsodicModificationRealTime as pmrt
import time
#import scipy.signal as sp
#import numpy as np
#from scipy.io import wavfile

p_flag = False
pitch_rate = 1.0
global_in = []
global_out = []

def main():
    #==============================================================================
    # INITIALIZATION
    #==============================================================================
    global p_flag, pitch_rate
    PITCH_INC = 0.01
    frame_format = 'Float32'
    fs = 16000

    transformation = pmrt.ProsodicModificationRealTime(fs, frame_format)
    transformation.pitch_rate = pitch_rate
    BUFFER_PYAUDIO_SIZE = transformation.mid_buffer_size


    #==============================================================================
    # DEFINE CALLBACK FUNCTION
    #==============================================================================
    def callback(in_data, frame_count, time_info, flag):
        global p_flag, pitch_rate, global_in, global_out
        if flag:
            print("Playback Error: %i" % flag)
        if p_flag is True:
            transformation.pitch_rate = pitch_rate
            p_flag = False
        in_data = transformation.str2numpy(in_data)
        global_in.append(in_data)
        out_data = transformation.pitchshifting(in_data)
        global_out.append(out_data)
        out_data = transformation.numpy2str(out_data)
        return out_data, paContinue


    #==============================================================================
    # START AUDIO STREAM
    #==============================================================================
    pa = PyAudio()
    stream = pa.open(format = paFloat32,
                     channels = 1,
                     rate = fs,
                     frames_per_buffer = BUFFER_PYAUDIO_SIZE,
                     input = True,
                     output = True,
                     stream_callback = callback)

    #==============================================================================
    # START LOOP
    #==============================================================================

    print('Press either "+" or "-" to increase/lower the pitch of the voice.\nTo quit the record/play session, press "space"')
    keypress = readchar.readchar()
    while stream.is_active():
        time.sleep(0.1)

        if keypress == ' ':
            break
        elif keypress == '+':
            pitch_rate += PITCH_INC
            print(pitch_rate)
            p_flag = True
            keypress = readchar.readchar()
        elif keypress == '-':
            pitch_rate -= PITCH_INC
            print(pitch_rate)
            p_flag = True
            keypress = readchar.readchar()
        else:
            keypress = readchar.readchar()
    print('Press any key to quit ...')
    keypress = readchar.readchar()

    #i = np.array(global_in)
    #o = np.array(global_out)
    #wavfile.write('input.wav',fs,i.reshape([i.size, 1]))
    #wavfile.write('output.wav',fs,o.reshape([o.size, 1]))
    stream.close()
    pa.terminate()


if __name__ == "__main__":
    main()


