import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import sys

def plot(length,dir,title):
    data,samplerate = sf.read(dir)
    data = np.array(data)
    d = data[rand:rand+length,rand1]
    plt.figure(title)
    plt.plot(d)
    
if(len(sys.argv)>1):
    length = int(sys.argv[1])
else: 
    length = 200

data,_ = sf.read('Paragon.flac')
rand = np.int(np.random.rand()*(data.shape[0]-length-1))
rand1 = np.int(np.random.rand())
plot(length,'Paragon.flac',"Origin")
plot(length,'Paragon_with_noise.flac',"With_Noise")
plot(length,'Paragon_after_process.flac',"After_Process")
plt.show()
