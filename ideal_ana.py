# Delensed BB
import numpy as np
import basic

# define parameters
D = '../../data/sodelens/20190707_test/'
Tcmb = 2.726e6    # CMB temperature
lmax = 3000       # maximum multipole of output cl
dlmin = 2
dlmax = 2048 

# load unlensed and lensed Cls
ucl  = basic.aps.read_cambcls('../../data/cls/ffp10_scalCls.dat',2,lmax,5)/Tcmb**2
lcl  = basic.aps.read_cambcls('../../data/cls/ffp10_lensedCls.dat',2,lmax,4,bb=True)/Tcmb**2

print('compute delensed BB')
EE, BB, pp = np.loadtxt(D+'cmbcls.dat',unpack=True,usecols=(2,3,7))
WE = np.ones(dlmax+1)
W0 = np.zeros(dlmax+1)
W1 = np.zeros(dlmax+1)
W2 = np.zeros(dlmax+1)
Ag0 = np.loadtxt(D+'al_TT.dat',unpack=True)[1]
Ag1 = np.loadtxt(D+'al_EB.dat',unpack=True)[1]
Ag2 = np.loadtxt(D+'al_MV.dat',unpack=True)[1]
for l in range(dlmin,dlmax+1):
    W0[l] = pp[l]/(pp[l]+Ag0[l]) #* pp0[l]/(pp[l]+Ag0[l])
    W1[l] = pp[l]/(pp[l]+Ag1[l]) #* pp1[l]/(pp[l]+Ag1[l])
    W2[l] = pp[l]/(pp[l]+Ag2[l]) #* pp1[l]/(pp[l]+Ag1[l])
    #W0[l] = pp[l]/(Ag0[l])
    #W1[l] = pp[l]/(Ag1[l])

#W0 = WE
#W1 = WE
bb0 = basic.delens.resbb(lmax,dlmin,dlmax,EE[:dlmax+1],pp[:dlmax+1],WE,W0)
bb1 = basic.delens.resbb(lmax,dlmin,dlmax,EE[:dlmax+1],pp[:dlmax+1],WE,W1)
bb2 = basic.delens.resbb(lmax,dlmin,dlmax,EE[:dlmax+1],pp[:dlmax+1],WE,W2)
#bb2 = basic.delens.lintemplate(lmax,dlmin,dlmax,EE[:dlmax+1],pp[:dlmax+1],WE,W0)
#bb3 = basic.delens.lintemplate(lmax,dlmin,dlmax,EE[:dlmax+1],pp[:dlmax+1],WE,W1)

print('save')
#np.savetxt(D+'exp.dat',np.array((np.linspace(0,lmax,lmax+1),bb0,bb1,bb2,bb3,lcl[1,:],lcl[2,:],ucl[3,:])).T)
np.savetxt(D+'exp.dat',np.array((np.linspace(0,lmax,lmax+1),bb0,bb1,bb2,lcl[1,:],lcl[2,:],ucl[3,:])).T)

