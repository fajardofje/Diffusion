#Script to show spin brownian motion 
#Vs. displacement predictions
#Jesus Fajardo (2020)
#jesuserf@med.umich.edu

import numpy as np
from numpy.linalg import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal as sps
from scipy.misc import derivative
from scipy.stats import norm as normdist
from scipy.stats import kurtosis, skew
import matplotlib.mlab as mlab
from scipy.ndimage import gaussian_filter
plt.rcParams.update({'font.size': 22})
import math
from math import e

#Global Variables
#X_g = -10.66e-6 #glass susceptibility
#X_w = -9.06e-6 #water susceptibility
DeltaX = 0.1e-6#X_g - X_w #3.e-8 Susceptibility change
nel = 2048 #pixels per side (64 to 256 acording to Pathak et al (2008). 1024 acording to Han et al. (2011))
domain =  np.zeros((nel,nel))

Xlen = 1000.e-6 #domain side lenght in m
Ylen = Xlen
Delta = Xlen/nel #Voxel Size (m)
a = 4e-6#internal cylinder radius (m) 0.575 * 1e-3
b = 5e-6#external cylinder radius 0.775 * 1e-3
np.random.seed(12) #random seed for reproducibility
Theta = np.pi #Angle between external field and cylinder axis
PhiB0 = 0. #Rotation of bundle in x-y plane
B0mag = 9. #Main external field magnitude
B0 = np.cos(Theta)*B0mag*np.asarray([1., 0.]) #Main external field direction



sigma = 0.05#T/m 0.03 to 0.04
muu = Xlen/2




#Spin variables
gamma = 267.5e6 #rad/T (giromagnetic ratio)
TE = 100.e-3 #Experiment Time 
Deltat = 0.001e-3
D0 = 0.7e-9 #m2/s = um2/ms
porc = 20 #spin each porc in water
nesp = 0 #number of intraaxon pixels
nesps = 10000
xcoord = np.random.randint(nel/4,3*nel/4,nesps)
ycoord = np.random.randint(nel/4,3*nel/4,nesps)

count = 0

xcoord = np.asarray(xcoord, dtype = int) #These are in index units
ycoord = np.asarray(ycoord, dtype = int)

xcoord_f = np.asarray(xcoord*Delta, dtype = float) #These are in SI (float) units
ycoord_f = np.asarray(ycoord*Delta, dtype = float)

print('Total spins: ', nesp) 

#Evolving spin phases arrays
spinsphases = np.zeros((int(TE//Deltat), len(xcoord))) 

#Positions history
xcoordhist = np.zeros((int(TE//Deltat), len(xcoord)), dtype = int) #These are in index units too
ycoordhist = np.zeros((int(TE//Deltat), len(ycoord)), dtype = int)

xcoordhist[0] = xcoord #These are in index units too
ycoordhist[0] = ycoord

#These 2 are for visualization purposes only
Traject1 = np.copy(domain) #trajectories array over field
Traject2 = np.copy(domain) #trajectories array over field

count = 0
semilla = 124
for i in range(int(TE//Deltat)): #Loop for counting inner spin phases
    if i%1000 == 0:
        print('Spin phases ',100*count/(int(TE//Deltat)),' % calculated')
    #Place the spins in the distorted field

    #np.random.seed(semilla+i)
    xcoord_f += np.random.normal(0., np.sqrt(2*Deltat*D0), len(xcoord_f)) #Position updates (These are in floats to not-forget the previous exact location)
    #np.random.seed(semilla+i+1)
    ycoord_f += np.random.normal(0., np.sqrt(2*Deltat*D0), len(ycoord_f))
    
    if any(ele > Xlen for ele in xcoord_f) : #Periodic conditions
        xcoord_f[np.where(xcoord_f > Xlen)[0]] = Xlen - abs(xcoord_f[np.where(xcoord_f > Xlen)[0]]-Xlen)
    if any(ele > Xlen for ele in ycoord_f) :
        ycoord_f[np.where(ycoord_f > Xlen)[0]] = Xlen - abs(ycoord_f[np.where(ycoord_f > Xlen)[0]]-Xlen)
    if any(ele < 0. for ele in xcoord_f) :
        xcoord_f[np.where(xcoord_f < 0.)[0]] = 0. + abs(xcoord_f[np.where(xcoord_f < 0.)[0]])
    if any(ele < 0. for ele in ycoord_f) :
        ycoord_f[np.where(ycoord_f < 0.)[0]] = 0. + abs(ycoord_f[np.where(ycoord_f < 0.)[0]])
    if i == int(TE//Deltat)-1: #This is because of i+1 in even arrays
        break
    xcoordhist[i+1] += np.asarray(xcoord_f/Delta, dtype=int) #These are in index units too
    ycoordhist[i+1] += np.asarray(ycoord_f/Delta, dtype=int)
    count += 1

#Spins Mean displacement
distancias = []
dist_theor = []
for i in range(len(xcoordhist)):
    r0 = np.array([ xcoordhist[0], ycoordhist[0] ])*Delta
    rf = np.array([ xcoordhist[i],ycoordhist[i] ])*Delta
    dx = abs(rf[0] - r0[0])
    dy = abs(rf[1] - r0[1])
    Delta_r = np.sqrt(dx**2 + dy**2)
    rm = np.mean(Delta_r**2)
    distancias.append(rm)
    dist_theor.append((4*D0*i*Deltat))
    print('Ensemble lenght: ', rm)
    print('Expected lenght: ', (4*D0*i*Deltat))
distancias = np.asarray(distancias)
dist_theor = np.asarray(dist_theor)

#Size Reference Bar
xarr = np.array([100,100+int((nel/Xlen)*1e-6)])
yarr = np.array([100,100])
#Plot Animated field
fig = plt.figure()
i = 0
imax = int(TE//Deltat)
plt.imshow(np.zeros((len(domain),len(domain))), cmap = 'hot', animated=True, alpha = 1.)
plt.text(xarr[0], yarr[0]-15, '1$\mu$m', color = 'white', fontsize=12)
plt.plot(xarr, yarr, '-w', markersize = 15.)
def animate(i):
    im = plt.scatter(xcoordhist[i],ycoordhist[i], color = 'blue', s=4.0)
    return im,
ani = animation.FuncAnimation(fig, animate, frames=imax, interval=1, repeat_delay=0, blit=True)
plt.show()

lc = 4*D0*TE

plt.plot(np.linspace(0,100,len(distancias)), distancias*1.e12 , '-b', alpha = 0.5,linewidth = 10, label=r'Montecarlo')
plt.plot(np.linspace(0,100,len(distancias)), dist_theor*1.e12 , '-k', linewidth = 2, label=r'$4D_0$TE')

plt.xlabel('TE (ms)')
plt.ylabel(r'$<\vec{\Delta_r}^2>$ $(\mu m^2)$')
#plt.yscale('log')
#plt.xscale('log')
plt.grid(True)
plt.legend()
plt.show()
