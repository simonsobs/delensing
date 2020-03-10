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

        if not overwrite and os.path.exists(fcmb.oalm['T'][i]) and os.path.exists(fcmb.oalm['E'][i]) and os.path.exists(fcmb.oalm['B'][i]):
            if verbose: print('Files exist:',fcmb.oalm['T'][i],'and E/B')
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
            pickle.dump((oalm[m]),open(fcmb.oalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            if t != 'id': 
                pickle.dump((salm[m]),open(fcmb.salm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump((nalm[m]),open(fcmb.nalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def alm_comb_freq(t,snmin,snmax,fcmb,verbose=True,overwrite=False,lcut=5000,freqs=['93','145'],mtype=['T','E','B']):
    
    for i in range(snmin,snmax+1):

        if verbose: print("map to alm", i, t)
        for mi, m in enumerate(mtype):

            if not overwrite and os.path.exists(fcmb.oalm[m][i]):
                if verbose: print('File exist:',fcmb.oalm[m][i])
                continue

            salm, nalm, Wl = 0., 0., 0.
            for freq in freqs:
                f0 = prjlib.filename_init(t=t,freq=freq)
                Nl = np.loadtxt(f0.cmb.scl,unpack=True)[mi+9]
                Nl[0:2] = 1.
                Il = 1./Nl
                salm += pickle.load(open(f0.cmb.salm[m][i],"rb"))*Il[:,None]
                nalm += pickle.load(open(f0.cmb.nalm[m][i],"rb"))*Il[:,None]
                Wl   += Il
            salm *= 1./Wl[:,None]
            nalm *= 1./Wl[:,None]
            salm[lcut+1:,:] = 0.
            nalm[lcut+1:,:] = 0.
            pickle.dump((salm+nalm),open(fcmb.oalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((salm),open(fcmb.salm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((nalm),open(fcmb.nalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def aps(t,snmin,snmax,lmax,fcmb,w,w2,w4,verbose=True,overwrite=False):

    if not overwrite and os.path.exists(fcmb.scl):
        if verbose: print('File exist:',fcmb.scl)
        return

    # aps for each rlz
    cl = {}
    cl['o'] = cmb.aps(snmin,snmax,lmax,fcmb.oalm,odd=False,verbose=verbose)
    if t != 'id':
        cl['s'] = cmb.aps(snmin,snmax,lmax,fcmb.salm,odd=False,verbose=verbose)
        cl['n'] = cmb.aps(snmin,snmax,lmax,fcmb.nalm,odd=False,verbose=verbose)
    else:
        cl['s'], cl['n'] = cl['o'], cl['o']*0.

    # save to files
    L = np.linspace(0,lmax,lmax+1)
    if verbose: print('save sim')
    mcl = {}
    for s in ['o','s','n']:
        mcl[s] = np.mean(cl[s][1:,:,:],axis=0)/w2
    np.savetxt(fcmb.scl,np.concatenate((L[None,:],mcl['o'],mcl['s'],mcl['n'])).T)



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
        T[0,ki,:] = M*hp.pixelfunc.ud_grade(Ts+Tn,nside)/Tcmb

    # inv noise covariance
    Nij = T*0.
    for ki, sigma in enumerate(s):
        Nij[0,ki,:] = W * (sigma*(np.pi/10800.)/2.726e6)**(-2)

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


def map2alm_wiener(i,t,falm,nsidela,lmax,cl,wla,wsa,nsidesa=512,verbose=False,chn=1,lmaxs=[0],nsides=[0],nsides1=[0],itns=[1000],eps=[1e-6],reducmn=False,lTmax=3000):

    bla = getbeam('la',lmax)
    bsa = getbeam('sa',lmax)
    npixla = 12*nsidela**2
    npixsa = 12*nsidesa**2

    # load LAT
    print('loading maps')
    QUla, iNla = loadQU(nsidela,'la',wla,i,s=[11.3,14.1])

    if t == 'co':
        QUsa, iNsa = loadQU(nsidesa,'sa',wsa,i,s=[3.68,4.67]) # Load SAT
        Elm, Blm = curvedsky.cninv.cnfilter_freq_nside(2,2,2,npixla,npixsa,lmax,cl[1:3,:],bla,bsa,iNla,iNsa,QUla,QUsa,chn,lmaxs=lmaxs,nsides0=nsides,nsides1=nsides1,itns=itns,eps=eps,filter='W',reducmn=reducmn)

    if t == 'la':
        T0, iNT = loadT(nside,'la',wla,i,s=[8.,10.])
        Elm, Blm = curvedsky.cninv.cnfilter_freq(2,2,npixla,lmax,cl[1:3,:],bla,iNla,QUsa,chn,lmaxs=lmaxs,nsides=nsides,itns=itns,eps=eps,filter='')
        Tlm = 0*Elm
        lmaxs[0] = lTmax
        Tlm[:lTmax+1,:lTmax+1] = curvedsky.cninv.cnfilter_freq(1,2,npixla,lTmax,cl[0:1,:lTmax+1],bla[:,:lTmax+1],iNT,T0,chn,lmaxs=lmaxs,nsides=nsides,itns=itns,eps=eps,filter='')
        pickle.dump((Tlm),open(falm['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    pickle.dump((Elm),open(falm['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((Blm),open(falm['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def map2alm_wiener_rlz(t,falm,nside,snmin,snmax,dlmax,tcl,fmask='/project/projectdirs/sobs/delensing/mask/',overwrite=False,verbose=False,chn=1,lmaxs=[0],nsides=[0],nsides1=[0],itns=[1000],eps=[1e-6],reducmn=False):

    print('loading LAT/SAT mask')
    wla = hp.fitsfunc.read_map(fmask+'la.fits',verbose=verbose)
    wsa = hp.fitsfunc.read_map(fmask+'sa.fits',verbose=verbose)

    for i in range(snmin,snmax+1):

        misctools.check_path(falm['E'][i],overwrite=overwrite)
        if verbose: print(i)

        map2alm_wiener(i,t,falm,nside,dlmax,tcl[:4,:dlmax+1],wla,wsa,verbose=verbose,chn=chn,lmaxs=lmaxs,nsides=nsides,nsides1=nsides1,itns=itns,eps=eps,reducmn=reducmn)



def map2alm_wiener_diag(snmin,snmax,falm,film,lmin,lmax,ocltt):

    Fl = np.zeros((lmax+1))

    for l in range(lmin,lmax+1):
        Fl[l] = 1./ocltt[l]

    for i in range(snmin,snmax+1):

        Talm = pickle.load(open(falm['T'][i],"rb"))
        Talm *= Fl[:,None]
        pickle.dump((Talm),open(film['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


'''
def alm_comb_telescope(p,f,r,salcut=1000):
    pla = prjlib.params(t='la',freq='coadd')
    psa = prjlib.params(t='sa',freq='coadd')
    fla = prjlib.filename(pla)
    fsa = prjlib.filename(psa)
    wla, wla2, wla4 = prjlib.window(fla.cmb.amask,'la')
    wsa, wsa2, wsa4 = prjlib.window(fsa.cmb.amask,'sa')
    for i in range(p.snmin,p.snmax):
        print("map to alm", i)
        for mi, m in enumerate(constants.mtype):
            n0 = np.loadtxt(fla.cmb.scl,unpack=True)[mi+9]
            n1 = np.loadtxt(fsa.cmb.scl,unpack=True)[mi+9]
            n0[0:2] = 1.
            n1[0:2] = 1.
            n0[salcut+1:] = 0.
            n1[salcut+1:] = 1.
            salmla = pickle.load(open(fla.cmb.salm[m][i],"rb"))
            salmsa = pickle.load(open(fsa.cmb.salm[m][i],"rb"))
            nalmla = pickle.load(open(fla.cmb.nalm[m][i],"rb"))
            nalmsa = pickle.load(open(fsa.cmb.nalm[m][i],"rb"))
            salm = (n1[:,None]*salmla + n0[:,None]*salmsa)/(n0[:,None]+n1[:,None])
            nalm = (n1[:,None]*nalmla + n0[:,None]*nalmsa)/(n0[:,None]+n1[:,None])
            oalm = salm + nalm
            pickle.dump((oalm),open(f.cmb.oalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((salm),open(f.cmb.salm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((nalm),open(f.cmb.nalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
'''


