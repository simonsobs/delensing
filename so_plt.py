import numpy as np
import basic
from matplotlib.pyplot import *

'''
D = '/project/projectdirs/sobs/delensing/20190626_masked_template_test/'
b, bb, ii, bi = np.loadtxt(D+'test_wEs_wwB_wklss_nE0_nI0.dat',unpack=True,usecols=(0,1,2,3))
res0 = 1.-bi**2/bb/ii
for E in ['Es','Eoco']:
    b, bb, LL, bL = np.loadtxt(D+'test_w'+E+'_wwB_wklss_nE1_nI1.dat',unpack=True,usecols=(0,1,2,3))
    res1 = 1.-bL**2/bb/LL
    xlabel('$\ell$')
    ylabel('Residual $BB$')
    xlim(2,100)
    ylim(.1,.7)
    plot(b,res0,'r--',label='idealistic')
    plot(b,res1,'r-',label='masked')
    savefig('fig_'+E+'.png')
    #show()
    clf()
'''

D = '/project/projectdirs/sobs/delensing/20190707_test/'
L, al = np.loadtxt(D+'al_MV.dat',unpack=True)
L, cl = np.loadtxt(D+'qlens.dat',unpack=True,usecols=(0,4))
ucl = basic.aps.read_cambcls('../../data/cls/ffp10_scalCls.dat',2,4000,5)/2.72e6**2
xlim(2,3000)
xscale('log')
#ylim(0,6e-19)
plot(L,L**2*(L+1)**2*(cl-al)/4.,'r-',label='sim (MV)')
plot(L,L**2*(L+1)**2*ucl[3,:]/4.,'k--',label='input')
legend(loc=0,frameon=False)
savefig('fig_al.png')
show()


'''
D = '/project/projectdirs/sobs/delensing/20190707_test/'
L, LT0, LT1, LT2 = np.loadtxt(D+'resbb.dat',unpack=True,usecols=(0,2,6,8))
l, lt0, lt1, lt2 = np.loadtxt(D+'exp.dat',unpack=True,usecols=(0,1,2,3))
xlim(2,150)
ylim(0,6e-19)
plot(l,lt2,'r--')
plot(L,LT2,'r-',label='tempalte (MV)')
cl = basic.aps.read_cambcls('../../data/cls/ffp10_lensedCls.dat',2,500,4,bb=True)/2.72e6**2
plot(l[:500+1],cl[2,:],'c-',label='lensing BB')
legend(loc=0,frameon=False)
savefig('fig_test.png')
show()
'''


'''
Tcmb = 2.72e6
Y = np.loadtxt('/project/projectdirs/sobs/delensing/multidelensing_clbb_SOv3_updated.dat')
l = Y[:,0]
ClBB = Y[:,1]               # C_l^BB
Planck = Y[:,2]            # Planck only
SO = Y[:,3]                  # SO only
C = Y[:,4]                     # CIB only
CW = Y[:,5]                 # CIB + WISE
CWP = Y[:,6]               # CIB + WISE + Planck
CWS = Y[:,7]               # CIB + WISE + SO
CWSD= Y[:,8]             # CIB + WISE + SO + DES (with tomographic bins)
CWSLg_single = Y[:,9]          # CIB + WISE + SO + LSST gold (single bin)
CWSLg_tomo = Y[:,10]         # CIB + WISE + SO + LSST gold (with tomographic bins)  
CWSLopt_tomo = Y[:,11]       # CIB + WISE + SO + LSST opt (with tomographic bins)
xlim(8,2000)
xscale('log')
#ylim(0,3e-19)
xlabel('$\ell$')
ylabel('$C_\ell^{BB}$')
plot(l,ClBB/Tcmb**2,'k-',label='lensing BB')
plot(l,SO/Tcmb**2,'g-',label='SO alone')
plot(l,CWSLg_tomo/Tcmb**2,'r-',label='SO+CIB+WISE+LSST')
legend(loc=0,frameon=False)
savefig('fig_lss.png')
show()
'''

