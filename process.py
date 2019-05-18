import numpy as np
from create_noise import data_with_noise,samplerate
import soundfile as sf
import os

path_after_process = "Paragon_after_process.flac"


#xn:2*(n*1),dn:n*1,M,mu,
#w:M*n,en:n*1,yn:n*1

#e(n) = d(n) - y(n) = d(n) - x(n).T*w(n)
#w(n+1) = w(n) + 2*mu*e(n)*x(n)
def LMSFilter(xn,dn,M,mu,err):
    L = xn.shape[0]
    w = np.zeros(M)

    for k in range(L)[M:L]:
        x = xn[k-M:k][::-1]
        en = dn[k] - x.T.dot(w)
        if(en>err):
            break
        w = w + 2*mu*en*x
    
    yn = np.zeros(L)
    for k in range(L)[M:L]:
        x = xn[k-M:k][::-1]
        yn[k] = w.T.dot(x)

    return yn,w,en

dn,_ = sf.read('Paragon.flac')
dn1 = dn[:,0]
dn2 = dn[:,1]
M = 100
mu = 1e-6
err = 100
data_after_process = data_with_noise
data_after_process[:,0],_,_ = LMSFilter(data_with_noise[:,0],dn1,M,mu,err)
data_after_process[:,1],_,_ = LMSFilter(data_with_noise[:,1],dn2,M,mu,err)
if (os.path.exists(path_after_process)==0):
    sf.write(path_after_process, data_with_noise, samplerate)
    print("done!\n")

