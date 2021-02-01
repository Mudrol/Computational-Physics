import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['legend.handlelength'] = 2
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14



"""
ADD read-in here!!
- read the data in signal_data.txt here, e.g., as instructed in the assignment
- the data in the file is given as (t,f(t)) 
"""



dt = t[1]-t[0]
N=len(t)

# Fourier coefficients from numpy fft normalized by multiplication of dt
F = np.fft.fft(f)*dt

# frequencies from numpy fftfreq
freq = np.fft.fftfreq(len(F),d=dt)

# inverse Fourier with numpy ifft (normalization removed with division by dt)
iF = np.fft.ifft(F/dt)

# For positive frequencies
# - use either freq[:N//2] using the variable above or 
# - define freq = np.linspace(0, 1.0/dt/2, N//2)

# plot the Fourier transform over positive frequencies f (i.e., omega=2*pi*f)
fig, ax = plt.subplots()
ax.plot(freq[:N//2], np.abs(F[:N//2]))
ax.set_xlabel(r'frequency (Hz)')
ax.set_ylabel(r'$F(\omega/2\pi)$')

# plot the "signal" and test the inverse transform (should be same)
fig, ax = plt.subplots()
ax.plot(t, f,label=r'signal')
ax.plot(t,iF.real,'r--',label=r'inverse Fourier transform')
ax.set_xlabel(r'$t(s)$')
ax.set_ylabel(r'$f(t)$')
ax.legend(loc=0)


# remove desired frequencies
if 1==1:
    # set the Fourier coefficients into a new variable Fnew 
    # (normalization removed with division by dt)
    Fnew = 1.0*F/dt
    # set F-coefficients to zero for frequencies above 60 Hz
    inds = abs(freq)>60
    Fnew[inds]=0.0
    # set F-coefficients to zero for frequencies below 40 Hz
    inds = abs(freq)<40
    Fnew[inds]=0.0
    
    # Calculate resulting signal to fnew
    fnew = np.fft.ifft(Fnew)

    # plot original and modified signals
    fig, ax = plt.subplots()
    ax.plot(t, f,label=r'original signal')
    ax.plot(t,fnew.real,'r--',label=r'removed frequencies below 40 and above 60 Hz')
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel(r'$f(t)$')
    ax.legend(loc=0)

plt.show()
