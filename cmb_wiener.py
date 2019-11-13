# Wiener filtered multipoles
import numpy as np
import healpy as hp
import pickle

# from cmblensplus
import curvedsky

# local
import prjlib


def getbeam(t,lmax,nu=['93','145']):
    bl = np.ones((len(nu),lmax+1))
    for ki, freq in enumerate(nu):
        bl[ki,:] = prjlib.get_beam(t,freq,lmax)
    return bl


def loadQU(nside,t,win,i,nu=['93','145'],s=[3.,5.],Enoise='la'):
    
    W  = hp.pixelfunc.ud_grade(win,nside)
    M  = W/(W+1e-30) # survey binary mask
    Tcmb = 2.72e6

    QU = np.zeros((2,len(nu),12*nside**2))
    for ki, freq in enumerate(nu):
        f0 = prjlib.filename_init(t=t,freq=freq)
        Qs, Us = hp.fitsfunc.read_map(f0.cmb.lcdm[i],field=1), hp.fitsfunc.read_map(f0.cmb.lcdm[i],field=2)
        Qn, Un = hp.fitsfunc.read_map(f0.cmb.nois[i],field=1), hp.fitsfunc.read_map(f0.cmb.nois[i],field=2)
        QU[0,ki,:] = M*hp.pixelfunc.ud_grade(Qs+Qn,nside)/Tcmb
        QU[1,ki,:] = M*hp.pixelfunc.ud_grade(Us+Un,nside)/Tcmb

    # inv noise covariance
    Nij = QU*0.
    for ki, sigma in enumerate(s):
        Nij[0,ki,:] = W * (sigma*(np.pi/10800.)/2.726e6)**(-2)
        Nij[1,ki,:] = Nij[0,ki,:]

    return QU, Nij


def Emode_Wiener(i,fcmb,lmax,cl,wla,wsa,k1=2,k2=2,nside=1024,Enoise='la'):
    
    bla = getbeam('la',lmax)
    bsa = getbeam('sa',lmax)
    QU1, Nij1 = loadQU(nside,'la',wla,i,s=[11.3,14.1],Enoise=Enoise)
    if Enoise == 'co':
        QU2, Nij2 = loadQU(512,'sa',wsa,i,s=[3.68,4.67],Enoise=Enoise)
        wElm, wBlm = curvedsky.cninv.cnfilter_so(2,k1,k2,lmax,cl,bla,bsa,12*nside**2,12*512**2,Nij1,Nij2,QU1,QU2,1000,filter='W')
    if Enoise == 'la':
        wElm, wBlm = curvedsky.cninv.cnfilter_lat(2,k1,lmax,cl,bla,12*nside**2,Nij1,QU1,1000,filter='W')
    pickle.dump((wElm),open(fcmb.walm['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((wBlm),open(fcmb.walm['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def func(snmin,snmax,olmax,dlmax,tcl,eL,Enoise):

    WLAT = hp.fitsfunc.read_map('/project/projectdirs/sobs/delensing/mask/la.fits')
    WSAT = hp.fitsfunc.read_map('/project/projectdirs/sobs/delensing/mask/sa.fits')

    for i in range(snmin,snmax):
        print(i)
        Emode_Wiener(i,f.filt[Enoise],dlmax,tcl[1:3,:dlmax+1],WLAT,WSAT,Enoise=Enoise)


p, f, r = prjlib.analysis_init()
func(p.snmin,p.snmax,p.olmax,p.dlmax,r.lcl,r.eL,'la')
func(p.snmin,p.snmax,p.olmax,p.dlmax,r.lcl,r.eL,'co')


