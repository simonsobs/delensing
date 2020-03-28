# map -> alm
import numpy as np
import healpy as hp
import pickle
import os

# from cmblensplus
import curvedsky
import constants
import cmb
import misctools

# local
import prjlib


def map2alm(nside,lmax,fmap,w,bl,vb=False):

    Tcmb = constants.Tcmb

    # load map
    Tmap = w * hp.fitsfunc.read_map(fmap,field=0,verbose=vb)/Tcmb
    Qmap = w * hp.fitsfunc.read_map(fmap,field=1,verbose=vb)/Tcmb
    Umap = w * hp.fitsfunc.read_map(fmap,field=2,verbose=vb)/Tcmb

    # map to alm
    alm = {}
    alm['T'] = curvedsky.utils.hp_map2alm(nside,lmax,lmax,Tmap)
    alm['E'], alm['B'] = curvedsky.utils.hp_map2alm_spin(nside,lmax,lmax,2,Qmap,Umap)

    # beam deconvolution
    for m in constants.mtype:
        alm[m] *= 1./bl[:,None]

    return alm


def map2alm_rlz(t,snmin,snmax,freq,nside,lmax,fcmb,w,verbose=True,overwrite=False,mtype=['T','E','B']):

    # beam
    bl = prjlib.get_beam(t,freq,lmax)

    # map -> alm
    for i in range(snmin,snmax+1):

        if not overwrite and os.path.exists(fcmb.alms['o']['T'][i]) and os.path.exists(fcmb.alms['o']['E'][i]) and os.path.exists(fcmb.alms['o']['B'][i]):
            if verbose: print('Files exist:',fcmb.alms['o']['T'][i],'and E/B')
            continue

        if verbose: print("map to alm", i, t)
        salm = map2alm(nside,lmax,fcmb.lcdm[i],w,bl)

        if t == 'id':
            oalm = salm
        else:
            nalm = map2alm(nside,lmax,fcmb.nois[i],w,bl)
            oalm = {}
            for m in mtype:
                oalm[m] = salm[m] + nalm[m]

        # save to files
        for m in mtype:
            pickle.dump((oalm[m]),open(fcmb.alms['o'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            if t != 'id': 
                pickle.dump((salm[m]),open(fcmb.alms['s'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump((nalm[m]),open(fcmb.alms['n'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def alm_comb_freq(t,snmin,snmax,fcmb,verbose=True,overwrite=False,lcut=5000,freqs=['93','145','225'],mtype=['T','E','B']):
    
    for i in range(snmin,snmax+1):

        if verbose: print("map to alm", i, t)
        for mi, m in enumerate(mtype):

            if misctools.check_path(fcmb.alms['o'][m][i],overwrite=overwrite,verbose=verbose): continue

            salm, nalm, Wl = 0., 0., 0.
            for freq in freqs:
                f0 = prjlib.filename_init(t=t,froeq=freq)
                Nl = np.loadtxt(f0.cmb.scl['n'],unpack=True)[mi+1]
                Nl[0:2] = 1.
                Il = 1./Nl
                salm += pickle.load(open(f0.cmb.alms['s'][m][i],"rb"))*Il[:,None]
                nalm += pickle.load(open(f0.cmb.alms['n'][m][i],"rb"))*Il[:,None]
                Wl   += Il
            salm *= 1./Wl[:,None]
            nalm *= 1./Wl[:,None]
            salm[lcut+1:,:] = 0.
            nalm[lcut+1:,:] = 0.
            pickle.dump((salm+nalm),open(fcmb.alms['o'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((salm),open(fcmb.alms['s'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((nalm),open(fcmb.alms['n'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def aps(snmin,snmax,lmax,fcmb,w2,stype=['o','s','n'],mtype=['T','E','B'],verbose=True,overwrite=False):

    # compute aps for each rlz
    cl = {}
    for s in stype:
        if verbose: print('stype =',s)
        odd = True
        if s in ['o','s','n']: odd = False
        cl[s] = cmb.aps(snmin,snmax,lmax,fcmb.alms[s],odd=odd,mtype=mtype,verbose=verbose,overwrite=overwrite,w2=w2,fname=fcmb.cl[s])

    # save average to files
    L = np.linspace(0,lmax,lmax+1)
    for s in stype:
        if misctools.check_path(fcmb.scl[s],verbose=verbose,overwrite=overwrite): return
        mcl = np.mean(cl[s][np.max(0,1-snmin):,:,:],axis=0)
        vcl = np.std(cl[s][np.max(0,1-snmin):,:,:],axis=0)
        np.savetxt(fcmb.scl[s],np.concatenate((L[None,:],mcl,vcl)).T)


def apsx(snmin,snmax,lmax,fcmb,gcmb,w2,verbose=True,overwrite=False):

    xl = cmb.apsx(snmin,snmax,lmax,fcmb.alms['w'],gcmb.alms['o'],verbose=verbose)/w2

    # save average to files
    L = np.linspace(0,lmax,lmax+1)
    mxl = np.mean(xl[np.max(0,1-snmin):,:,:],axis=0)
    vxl = np.std(xl[np.max(0,1-snmin):,:,:],axis=0)
    np.savetxt(fcmb.scl['x'],np.concatenate((L[None,:],mxl,vxl)).T)



def getbeam(t,lmax,nu=['93','145']):
    bl = np.ones((len(nu),lmax+1))
    for ki, freq in enumerate(nu):
        bl[ki,:] = prjlib.get_beam(t,freq,lmax)
    return bl



def loadT(nside,t,win,i,nu=['93','145'],s=[3.,5.],verbose=False):
    
    W  = hp.pixelfunc.ud_grade(win,nside)
    M  = W/(W+1e-30) # survey binary mask
    Tcmb = constants.Tcmb

    T = np.zeros((1,len(nu),12*nside**2))
    for ki, freq in enumerate(nu):
        f0 = prjlib.filename_init(t=t,freq=freq)
        Ts = hp.fitsfunc.read_map(f0.cmb.lcdm[i],field=0,verbose=verbose)
        Tn = hp.fitsfunc.read_map(f0.cmb.nois[i],field=0,verbose=verbose)
        T[0,ki,:] = M * hp.pixelfunc.ud_grade(Ts+Tn,nside)/Tcmb

    # inv noise covariance
    Nij = T*0.
    for ki, sigma in enumerate(s):
        Nij[0,ki,:] = W * (sigma*(np.pi/10800.)/Tcmb)**(-2)

    return T, Nij


def loadQU(nside,t,win,i,nu=['93','145'],s=[3.,5.],verbose=False):
    
    W = hp.pixelfunc.ud_grade(win,nside)
    M = W/(W+1e-30) # survey binary mask
    Tcmb = constants.Tcmb

    QU = np.zeros((2,len(nu),12*nside**2))
    for ki, freq in enumerate(nu):
        f0 = prjlib.filename_init(t=t,freq=freq)
        Qs = hp.fitsfunc.read_map(f0.cmb.lcdm[i],field=1,verbose=verbose)
        Us = hp.fitsfunc.read_map(f0.cmb.lcdm[i],field=2,verbose=verbose)
        Qn = hp.fitsfunc.read_map(f0.cmb.nois[i],field=1,verbose=verbose)
        Un = hp.fitsfunc.read_map(f0.cmb.nois[i],field=2,verbose=verbose)
        QU[0,ki,:] = M * hp.pixelfunc.ud_grade(Qs+Qn,nside)/Tcmb
        QU[1,ki,:] = M * hp.pixelfunc.ud_grade(Us+Un,nside)/Tcmb

    # inv noise covariance
    Nij = QU*0.
    for ki, sigma in enumerate(s):
        Nij[0,ki,:] = W * (sigma*(np.pi/10800.)/Tcmb)**(-2)
        Nij[1,ki,:] = Nij[0,ki,:]

    return QU, Nij


def map2alm_wiener(i,t,falm,nsidela,lmax,cl,wla,wsa,nsidesa=512,verbose=False,chn=1,lmaxs=[0],nsides=[0],nsides1=[0],itns=[1000],eps=[1e-6],reducmn=False,mn=3,nu=['93','145','225'],lTmax=1000,lTcut=100):

    bla = getbeam('la',lmax,nu=nu)
    bsa = getbeam('sa',lmax,nu=nu)
    npixla = 12*nsidela**2
    npixsa = 12*nsidesa**2

    # load LAT
    print('loading maps')
    QUla, iNla = loadQU(nsidela,'la',wla,i,nu=nu,s=[11.3,14.1,31.1])

    if t == 'co':
        QUsa, iNsa = loadQU(nsidesa,'sa',wsa,i,nu=nu,s=[3.68,4.67,8.91]) # Load SAT
        Elm, Blm = curvedsky.cninv.cnfilter_freq_nside(2,mn,mn,npixla,npixsa,lmax,cl[1:3,:],bla,bsa,iNla,iNsa,QUla,QUsa,chn,lmaxs=lmaxs,nsides0=nsides,nsides1=nsides1,itns=itns,eps=eps,filter='W',reducmn=reducmn)

    if t == 'la':
        '''
        #T0, iNT = loadT(nsidela,'la',wla,i,s=[8.,10.])
        T0, iNT = loadT(nsidela,'la',wla,i,s=[10.],nu=['145'])
        Tlm = pickle.load(open(falm['T'][12],"rb"))*0.
        eps[0] = 1e-7
        itns[0] = 3000
        lmaxs[0] = lTmax
        cl[0,:lTcut+1] = 0.
        Tlm[:lTmax+1,:lTmax+1] = curvedsky.cninv.cnfilter_freq(1,1,npixla,lTmax,cl[0:1,:lTmax+1],bla[1:2,:lTmax+1],iNT,T0,chn,lmaxs=lmaxs,nsides=nsides,itns=itns,eps=eps,filter='',ro=1)
        pickle.dump((Tlm),open(falm['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        '''
        Elm, Blm = curvedsky.cninv.cnfilter_freq(2,mn,npixla,lmax,cl[1:3,:],bla,iNla,QUla,chn,lmaxs=lmaxs,nsides=nsides,itns=itns,eps=eps,filter='W')

    pickle.dump((Elm),open(falm['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((Blm),open(falm['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def map2alm_wiener_rlz(t,falm,nside,snmin,snmax,dlmax,tcl,fmask='/project/projectdirs/sobs/delensing/mask/',overwrite=False,verbose=False,chn=1,lmaxs=[0],nsides=[0],nsides1=[0],itns=[1000],eps=[1e-6],reducmn=False):

    print('loading LAT/SAT mask')
    wla = hp.fitsfunc.read_map(fmask+'la.fits',verbose=verbose)
    wsa = hp.fitsfunc.read_map(fmask+'sa.fits',verbose=verbose)

    for i in range(snmin,snmax+1):

        if misctools.check_path(falm['E'][i],overwrite=overwrite): continue
        if verbose: print(i)

        map2alm_wiener(i,t,falm,nside,dlmax,tcl[:4,:dlmax+1],wla,wsa,verbose=verbose,chn=chn,lmaxs=lmaxs,nsides=nsides,nsides1=nsides1,itns=itns,eps=eps,reducmn=reducmn)



def map2alm_wiener_diag(snmin,snmax,falm,film,lmin,lmax,cls,ocls,mtype=['T','E','B'],overwrite=False):

    Fl = {}
    for m in mtype:
        Fl[m] = np.zeros((lmax+1))

    for l in range(lmin,lmax+1):
        Fl['T'][l] = cls[0,l]/ocls[0,l]
        Fl['E'][l] = cls[1,l]/ocls[1,l]
        Fl['B'][l] = cls[2,l]/ocls[2,l]

    for i in range(snmin,snmax+1):

        for m in mtype:
            if misctools.check_path(film[m][i],overwrite=overwrite): continue
            alm = pickle.load(open(falm[m][i],"rb"))
            alm *= Fl[m][:,None]
            pickle.dump((alm),open(film[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def wiener_iso(snmin,snmax,lmin,lmax,fwlm,cls,ocls,ncls,mtype=['T','E','B'],overwrite=False):

    Fl = {}
    for m in mtype:
        Fl[m] = np.zeros((lmax+1))

    for l in range(lmin,lmax+1):
        Fl['T'][l] = cls[0,l]/ocls[0,l]
        Fl['E'][l] = cls[1,l]/ocls[1,l]
        Fl['B'][l] = cls[2,l]/ocls[2,l]

    __, fid, __ = prjlib.analysis_init(t='id',snmin=snmin,snmax=snmax,lmax=lmax)

    for i in range(snmin,snmax+1):

        for mi, m in enumerate(mtype):

            if misctools.check_path(fwlm[m][i],overwrite=overwrite): continue

            alm = pickle.load(open(fid.cmb.alms['o'][m][i],"rb"))
            alm += curvedsky.utils.gauss1alm(lmax,ncls[mi,:])
            alm *= Fl[m][:,None]
            pickle.dump((alm),open(fwlm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


'''
def map2alm_convert(snmin,snmax,film,fwlm,lmin,lmax,cls,overwrite=False):

    for i in range(snmin,snmax+1):

        for mi, m in enumerate(['T','E','B']):
            if misctools.check_path(fwlm[m][i],overwrite=overwrite): continue
            alm = pickle.load(open(film[m][i],"rb"))
            alm *= cls[mi,:,None]
            pickle.dump((alm),open(fwlm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
'''


