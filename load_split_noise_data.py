# --------------------------------------------------------------------
# Purpose: 
#    Load the noise dataset and split into short segments
#
# Authors and Copyright:
#    Mr.Wind 
#-----------------------------------------------------------------------

import wave
import numpy as np
import glob
import os
import random
from tqdm import tqdm



# ----------------------------------------------------
# * Function:
#    wav_read()---read the wave file
# * Arguments:
#    * filename -- The audio filename of wave file
# * Returns:
#    * waveData -- The read data
#    * framerate -- The sampling rate
# ---------------------------------------------------


def wav_read(filename):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # read audio file, in string format
    waveData = np.frombuffer(strData, dtype=np.int16)  # convert string to int16
    # waveData = waveData*1.0/(max(abs(waveData)))#wave normalization
    # waveData = waveData/32767 #wave normalization
    f.close()
    return waveData, framerate


# --------------------------------------------------------------------
# *Function:
#    wav_write()---Write the wave file
#
# * Arguments:
#    * waveData -- The data need to be wirtten, of shape (-1,1)
#    * filepath -- The output audio filepath of wave file
#    * filename -- The output audio filename of wave file
#    * fs       -- Sampling rate
# * Returns:
#    * None
# ---------------------------------------------------------------------

    
def wav_write(waveData, filepath, filename, fs):
    outData = np.array(waveData, dtype='int16')  # the data need to be loaded
    # outData = np.array(waveData*32767,dtype='int16')# load data normalization
    outfile = filepath + os.sep + filename
    outwave = wave.open(outfile, 'wb')  # define the load path and filename
    outwave.setnchannels(1)
    outwave.setsampwidth(2)
    outwave.setframerate(fs)
    outwave.writeframes(outData.tostring())  # outData:int16.
    outwave.close()



# --------------------------------------------------------------------
# *Function:
#    read_split_noise()---Read and split the noise wave files
#
# * Arguments:
#    * DATASET_PATH -- The dirpath of source speech data 
#    * out_data_path -- The output path for splited wav
#    * min_segment -- The minimum duration of each segmented wav
#    * max_segment -- The maximum duration of each segmented wav
#    * noise_index -- The start index for noise wav file
# * Returns:
#    * out_wav_name -- The filename of ouyput wav
# ---------------------------------------------------------------------


def read_split_noise(DATASET_PATH,out_data_path,segment_len,noise_index):

    file_list = glob.glob(os.path.join(DATASET_PATH, '*.wav'))
    l = len(file_list)
    for k in tqdm(range(0, l//10)):
        # print(wav_file)
        wav_file = file_list[k*10]
        noise_data, fs = wav_read(wav_file)
        if k == 0:
            cat_wav = noise_data.reshape(-1,1)
        else:
            cat_wav = np.append(cat_wav,noise_data.reshape(-1,1),axis=0)
        k = k+1
    print(cat_wav.shape)
    # Split the long data array into segments (segment_len)
    # segment_len = random.randint(min_segment,max_segment)
    num_segment = cat_wav.shape[0]//(fs*segment_len)
    use_length = num_segment*(fs*segment_len)
    split_wav = np.array_split(cat_wav[:(use_length),:], num_segment, axis=0)
    print(num_segment)
    for i in range (num_segment):
        out_wav_name = 'noise_' + str(noise_index) + '.wav'
        wav_write(split_wav[i], out_data_path, out_wav_name, fs)
        noise_index = noise_index+1
    print('The final noise index is:', noise_index-1)

    return out_wav_name, noise_index



def splitNoiseData(dataset_root_path, output_root_path="./noise_data", type="train"):
    print('Split Noise Segments: Start processing...')
    DATASET_PATH = os.path.join(dataset_root_path, type)
    out_data_path = os.path.join(output_root_path, type)
    if not os.path.exists(out_data_path):
        os.makedirs(out_data_path)
    segment_len = 10 # The splited audio length
    start_noise_index = 0 # Update index before using
    read_split_noise(DATASET_PATH, out_data_path, segment_len, start_noise_index)
    print('Processing done !')

