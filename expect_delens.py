# Delensed BB
import numpy as np
import basic
import prjlib

p, f, r = prjlib.analysis_init()
#prjlib.make_delens_filter(p,f,r)
p.dlmin = 20
p.dlmax = 5000

print('compute delensed BB')
#W0 = r.wlk['TT'][:p.dlmax+1,0]*r.kL[:p.dlmax+1]
#W1 = r.wlk['EB'][:p.dlmax+1,0]*r.kL[:p.dlmax+1]
Nli = np.loadtxt('../../data/forecast/so/noise/SO_LAT_Nell_P_goal_fsky0p4.txt',unpack=True,usecols=(1,2,3,4,5,6))/2.72e6**2
Nl = 1./np.sum(1./Nli,axis=0)

W0 = np.ones(p.dlmax+1)
W1 = np.ones(p.dlmax+1)
obs = np.loadtxt(f.cmb.scl,unpack=True,usecols=(1,2,3,4))
sig = np.loadtxt(f.cmb.scl,unpack=True,usecols=(5,6,7,8))

WE = r.ucl[1,:p.dlmax+1]/(r.ucl[1,:p.dlmax+1]+Nl[:p.dlmax+1])
we = np.ones(p.dlmax+1)
WE[:100]  = 0.
we[:100]  = 0.
W0[3001:] = 0.
W1[3001:] = 0.

bb0 = basic.delens.resbb(p.lmax,p.dlmin,p.dlmax,r.ucl[1,:p.dlmax+1],r.ucl[3,:p.dlmax+1],WE,W0)
bb1 = basic.delens.resbb(p.lmax,p.dlmin,p.dlmax,r.ucl[1,:p.dlmax+1],r.ucl[3,:p.dlmax+1],we,W1)
bb2 = basic.delens.lintemplate(p.lmax,p.dlmin,p.dlmax,r.ucl[1,:p.dlmax+1],r.ucl[3,:p.dlmax+1],WE,W0)
bb3 = basic.delens.lintemplate(p.lmax,p.dlmin,p.dlmax,r.ucl[1,:p.dlmax+1],r.ucl[3,:p.dlmax+1],we,W1)

print('save')
np.savetxt('test_exp.dat',np.array((r.eL,bb0,bb1,bb2,bb3,r.lcl[1,:],r.lcl[2,:],r.ucl[3,:])).T)

