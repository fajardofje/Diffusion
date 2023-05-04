#Script to show spin brownian motion 
#Vs. displacement predictions for a cylinder-like restriction
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
from scipy.misc import derivative
from scipy.stats import kurtosis, skew
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
import matplotlib.mlab as mlab
from scipy.ndimage import gaussian_filter
from scipy import special
plt.rcParams.update({'font.size': 22})
import math
from math import e

#Global Variables
#X_g = -10.66e-6 #glass susceptibility
#X_w = -9.06e-6 #water susceptibility
DeltaX = 0.1e-6#X_g - X_w #3.e-8 Susceptibility change
nel = 2048 #pixels per side (64 to 256 acording to Pathak et al (2008). 1024 acording to Han et al. (2011))
domain = np.zeros((nel,nel))  #internal domain
maskout = np.load('outmask.npy') #mask to extract extraaxonal data
maskin = np.load('intmask.npy') #mask to extract intraaxonal data


Xlen = 15.e-6 #domain side lenght in m
Ylen = Xlen
Delta = Xlen/nel #Voxel Size (m)
np.random.seed(12) #random seed for reproducibility
Theta = np.pi #Angle between external field and cylinder axis
PhiB0 = 0. #Rotation of bundle in x-y plane
B0mag = 9. #Main external field magnitude
B0 = np.cos(Theta)*B0mag*np.asarray([1., 0.]) #Main external field direction



sigma = 0.05#T/m Tipo de 0.03 a 0.04 es medio gausiana
muu = Xlen/2


#Spin variables
gamma = 267.5e6 #rad/T (giromagnetic ratio)
TE = 60.e-3 #Experiment Time 
Deltat = 0.001e-3
D0 = 2.3e-9 #m2/s = 2um2/ms
porc = 20 #spin each porc in water
nesp = 0 #number of intraaxon pixels
nint = 0
next = 0


xcoordin = []
#xcoordmiel = []
#xcoordout = []
ycoordin = []
#ycoordout = []

nesps = 1000
xcoords = np.random.randint(1,nel,nesps)
ycoords = np.random.randint(1,nel,nesps)

count = 0
for i in range(0 , len(xcoords)): #Loop for counting & generating spins in water
    print('Water spins ',100**count/(len(xcoords)),' % placed')
    if maskin[xcoords[i],ycoords[i]] == 0:
        nint += 1
        xcoordin.append(xcoords[i])
        ycoordin.append(ycoords[i])


xcoordin = np.asarray(xcoordin, dtype = int)
#xcoordout = np.asarray(xcoordout, dtype = int)
ycoordin = np.asarray(ycoordin, dtype = int)
#ycoordout = np.asarray(ycoordout, dtype = int)

xcoordin_f = np.asarray(xcoordin*Delta, dtype = float)
#xcoordout_f = np.asarray(xcoordout*Delta, dtype = float)
ycoordin_f = np.asarray(ycoordin*Delta, dtype = float)
#ycoordout_f = np.asarray(ycoordout*Delta, dtype = float)

print('Elements intraaxon: ', nint)
print('Elements extraaxon: ', next)            
print('Total spins: ', next+nint) 

#Evolving spin phases arrays
#spinsphases = np.zeros((int(TE//Deltat), len(xcoord))) 
intspins = np.zeros((int(TE//Deltat), len(xcoordin)))
#outspins = np.zeros((int(TE//Deltat), len(xcoordout)))

#Positions history
xcoordinhist = np.zeros((int(TE//Deltat), len(xcoordin)), dtype = int) #These are in index units too
ycoordinhist = np.zeros((int(TE//Deltat), len(ycoordin)), dtype = int)
D_w_hist = np.zeros((int(TE//Deltat)), dtype = float) #array con historico del valor medio delta omegas
#xcoordouthist = np.zeros((int(TE//Deltat), len(xcoordout)), dtype = int) #These are in index units too
#ycoordouthist = np.zeros((int(TE//Deltat), len(ycoordout)), dtype = int)


xcoordinhist[0] = xcoordin #These are in index units too
ycoordinhist[0] = ycoordin
#xcoordouthist[0] = xcoordout #These are in index units too
#ycoordouthist[0] = ycoordout


#These 2 are for visualization purposes only
Traject1 = np.copy(domain)#Trajeactories array over field
Traject2 = np.copy(domain) #trajectories array over field

count = 0
tau_c = 0.
for i in range(int(TE//Deltat)): #Loop for counting inner spin phases
    if i%1000 == 0:
        print('Spin phases ',100*count/(int(TE//Deltat)),' % calculated')

    #Fields FID########################################################################
    #nnerfields = DeltaBG_up[xcoordinhist[i-1], ycoordinhist[i-1]] 
    #outerfields = DeltaBG_up[xcoordouthist[i-1], ycoordouthist[i-1]]
    ###################################################################################
    
    #Place the spins in the distorted field
    #intspins[i] = intspins[i-1] + gamma*innerfields*Deltat
    
    #outspins[i] = outspins[i-1] + gamma*outerfields*Deltat
    cfactor = 1.
    increment_xin = np.random.normal(0., cfactor*np.sqrt(2*Deltat*D0), len(xcoordin_f))
    increment_yin = np.random.normal(0., cfactor*np.sqrt(2*Deltat*D0), len(xcoordin_f))
    #increment_xout = np.random.normal(0., 0.7*np.sqrt(2*Deltat*D0), len(xcoordout_f))
    #increment_yout = np.random.normal(0., 0.7*np.sqrt(2*Deltat*D0), len(xcoordout_f))
    xcoordin_f += increment_xin
    ycoordin_f += increment_yin

    #xcoordout_f += increment_xout 
    #ycoordout_f += increment_yout
    
    if any(ele > Xlen for ele in xcoordin_f) : #Periodic conditions
        xcoordin_f[np.where(xcoordin_f > Xlen)[0]] = Xlen - abs(xcoordin_f[np.where(xcoordin_f > Xlen)[0]]-Xlen)
    if any(ele > Xlen for ele in ycoordin_f) :
        ycoordin_f[np.where(ycoordin_f > Xlen)[0]] = Xlen - abs(ycoordin_f[np.where(ycoordin_f > Xlen)[0]]-Xlen)
    if any(ele < 0. for ele in xcoordin_f) :
        xcoordin_f[np.where(xcoordin_f < 0.)[0]] = 0. + abs(xcoordin_f[np.where(xcoordin_f < 0.)[0]])
    if any(ele < 0. for ele in ycoordin_f) :
        ycoordin_f[np.where(ycoordin_f < 0.)[0]] = 0. + abs(ycoordin_f[np.where(ycoordin_f < 0.)[0]])
    if any(maskin[(xcoordin_f/Delta).astype(int),(ycoordin_f/Delta).astype(int)] == 1) : #Choque con la mielina
        xcoordin_f[np.where(maskin[(xcoordin_f/Delta).astype(int),(ycoordin_f/Delta).astype(int)] == 1)[0]] -= increment_xin[np.where(maskin[(xcoordin_f/Delta).astype(int),(ycoordin_f/Delta).astype(int)] == 1)[0]]
        ycoordin_f[np.where(maskin[(xcoordin_f/Delta).astype(int),(ycoordin_f/Delta).astype(int)] == 1)[0]] -= increment_yin[np.where(maskin[(xcoordin_f/Delta).astype(int),(ycoordin_f/Delta).astype(int)] == 1)[0]]
    '''   
    if any(ele > Xlen for ele in xcoordout_f) : #Periodic conditions
        xcoordout_f[np.where(xcoordout_f > Xlen)[0]] = Xlen - abs(xcoordout_f[np.where(xcoordout_f > Xlen)[0]]-Xlen)
    if any(ele > Xlen for ele in ycoordout_f) :
        ycoordout_f[np.where(ycoordout_f > Xlen)[0]] = Xlen - abs(ycoordout_f[np.where(ycoordout_f > Xlen)[0]]-Xlen)
    if any(ele < 0. for ele in xcoordout_f) :
        xcoordout_f[np.where(xcoordout_f < 0.)[0]] = 0. + abs(xcoordout_f[np.where(xcoordout_f < 0.)[0]])
    if any(ele < 0. for ele in ycoordout_f) :
        ycoordout_f[np.where(ycoordout_f < 0.)[0]] = 0. + abs(ycoordout_f[np.where(ycoordout_f < 0.)[0]])
    
    if any(maskout[(xcoordout_f/Delta).astype(int),(ycoordout_f/Delta).astype(int)] == 1) :  #Choque con la mielina
        xcoordout_f[np.where(maskout[(xcoordout_f/Delta).astype(int),(ycoordout_f/Delta).astype(int)] == 1)[0]] -= 2*increment_xout[np.where(maskout[(xcoordout_f/Delta).astype(int),(ycoordout_f/Delta).astype(int)] == 1)[0]]
        ycoordout_f[np.where(maskout[(xcoordout_f/Delta).astype(int),(ycoordout_f/Delta).astype(int)] == 1)[0]] -= 2*increment_yout[np.where(maskout[(xcoordout_f/Delta).astype(int),(ycoordout_f/Delta).astype(int)] == 1)[0]]
    '''
    if i == int(TE//Deltat)-1: #Este Break es por el i+1 de los arrrays enteros
        break
    xcoordinhist[i+1] = np.asarray(xcoordin_f/Delta, dtype=int) #These are in index units too
    ycoordinhist[i+1] = np.asarray(ycoordin_f/Delta, dtype=int)
    #_w_hist[i] = np.mean(DeltaBG_up[xcoordinhist[0], ycoordinhist[0]]*DeltaBG_up[xcoordinhist[i], ycoordinhist[i]])/np.mean((DeltaBG_up[xcoordinhist[0], ycoordinhist[0]])**2)
    tau_c += D_w_hist[i]*Deltat
    '''
    print('numerador: ',numerador)
    print('denominador: ',denominador)
    print('Meanphase: ', "{:e}".format(numerador/denominador))
    print('Meanphase: ', "{:e}".format(D_w_hist[i]))
    print('tau_c: ', tau_c)
    input()
    '''
    #xcoordouthist[i+1] = np.asarray(xcoordout_f/Delta, dtype=int)
    #ycoordouthist[i+1] = np.asarray(ycoordout_f/Delta, dtype=int)
    count += 1


#allspins = np.concatenate((intspins,outspins), axis = 1)
#allspins = np.concatenate((intspins,outspins), axis = 1)

#Recorrido medio de spines
distancias = []
dist_theor = []
for i in range(len(xcoordinhist)):
    r0_in = np.array([ xcoordinhist[0], ycoordinhist[0] ])*Delta
    rf_in = np.array([ xcoordinhist[i],ycoordinhist[i] ])*Delta
    dx = abs(rf_in[0] - r0_in[0])
    dy = abs(rf_in[1] - r0_in[1])
    Delta_r_in = np.sqrt(dx**2 + dy**2)
    rm_in = np.mean(dx**2)
    distancias.append(rm_in)
print('long ensamble interno: ', rm_in)    
xaxis = np.linspace(0,60,len(distancias))
distancias = (np.asarray(distancias))



#Size Reference Bar
xarr = np.array([100,100+int((nel/Xlen)*10e-6)])
yarr = np.array([100,100])
#Plot Animated field
fig = plt.figure()
i = 0
imax = int(TE//Deltat)
plt.imshow(maskin.T, cmap = 'seismic', animated=True, alpha = 0.5)
plt.text(xarr[0], yarr[0]-15, '10$\mu$m', color = 'white', fontsize=12)
plt.plot(xarr, yarr, '-w', markersize = 15.)
def animate(i):
    im1 = plt.scatter(xcoordinhist[i,:],ycoordinhist[i,:], color = 'black', s=6.0)
    #im2 = plt.scatter(xcoordouthist[i],ycoordouthist[i], color = 'red', s=6.0, alpha = 0.)
    return im1,#, im2
ani = animation.FuncAnimation(fig, animate, frames=imax, interval=50, repeat_delay=0, blit=True)
plt.show()


#plt.plot(xaxis, (distancias*1.e12) , '-k', linewidth = 2, label=r'$<\vec{\Delta_x}^2>$')
#spl = UnivariateSpline(xaxis, (distancias*1.e12), k=5)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#suavizada = make_interp_spline(xaxis, (distancias*1.e12))#spl(xaxis)
#X_ = np.linspace(xaxis.min(), xaxis.max(), 200)
#Y_ = suavizada(X_)
suavizada = smooth((distancias*1.e12), 200) 

print(suavizada)
plt.plot(xaxis, (distancias*1.e12) , '-b', linewidth = 2, label=r'$<\vec{\Delta_x}^2>$')
#plt.plot(xaxis, suavizada , '-g', linewidth = 2, label=r'$<\vec{\Delta_x}^2>$ suavizada')
#plt.plot(xaxis, (np.gradient(distancias*1.e12)) , '-k', linewidth = 2, label=r'derivada suavizada')
#plt.plot(xaxis, np.gradient(suavizada) , '-b', linewidth = 2, label=r'derivada suavizada')
plt.axhline( y=((0.37*10.e-6)**2)*1.e12, color='r')
plt.xlabel('TE (ms)')
plt.ylabel(r'$<\vec{\Delta_x}^2>$ $(\mu m^2)$')
#plt.yscale('log')
#plt.xscale('log')
plt.grid(True)
plt.legend()
plt.show()


