import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import sys

def plot(length,dir,title):
    data,samplerate = sf.read(dir)
    data = np.array(data)
    d = data[rand:rand+length,rand1]
    plt.ylabel(title)
    plt.plot(d)

if(len(sys.argv)>1):
    length = int(sys.argv[1])
else: 
    length = 200

data,_ = sf.read('Paragon.flac')
rand = np.int(np.random.rand()*(data.shape[0]-length-1))
rand1 = np.int(np.random.rand())
plt.figure()
plt.subplot(3,1,1)
plot(length,'Paragon.flac',"Origin")
plt.subplot(3,1,2)
plot(length,'Paragon_with_noise.flac',"With_Noise")
plt.subplot(3,1,3)
plot(length,'Paragon_after_process.flac',"After_Process")
plt.show()
