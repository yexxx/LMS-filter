import soundfile as sf
import os
import numpy as np

path = 'Paragon.flac'
path_with_noise = 'Paragon_with_noise.flac'
data,samplerate = sf.read(path)
data = np.array(data)
 #加噪声
data_with_noise = data + (np.random.rand(data.shape[0],data.shape[1])-0.5)*2
#print("max = " + str(np.min(data)) + "\nmin = " + str(np.min(data)))
if (os.path.exists(path_with_noise)==0):
    sf.write(path_with_noise, data_with_noise, samplerate) 