# Linear template delensing
import numpy as np
import healpy as hp
import pickle

# from cmblensplus
import curvedsky

# local
import prjlib


def remap(p,wElm,Balm,wglm):
    print('precompute shift vector')
    beta = curvedsky.delens.shiftvec(p.npix,p.dlmax,wglm[:p.dlmax+1,:p.dlmax+1])
    print('remap TQU')
    Talm = wElm[:p.dlmax+1,:p.dlmax+1]
    Trlm, Erlm, Brlm = curvedsky.delens.remap_tp(p.npix,p.dlmax,beta,np.array((Talm,wElm[:p.dlmax+1,:p.dlmax+1],Balm[:p.dlmax+1,:p.dlmax+1])))


def multiplywindow(W,npix,nside,lmax,Ealm,Balm):
    Q, U = W*curvedsky.utils.hp_alm2map_spin(npix,lmax,lmax,2,Ealm,Balm)
    ealm, balm = curvedsky.utils.hp_map2alm_spin(nside,lmax,lmax,2,Q,U)
    return ealm, balm


def Emode_ideal(lmax,felm,Enoise='cv'):

    Ealm = pickle.load(open(felm,"rb"))[0:lmax+1,0:lmax+1]
    if Enoise in ['la','co']:
        if Enoise == 'la':
            wle, NE = prjlib.getwElm(2,lmax,t=Enoise,freq='coadd')
        if Enoise == 'co':
            fla = prjlib.filename_init(t='la',freq='coadd')
            fsa = prjlib.filename_init(t='sa',freq='coadd')
            sig, nla = np.loadtxt(fla.cmb.scl,unpack=True,usecols=(6,10))
            nsa = np.loadtxt(fsa.cmb.scl,unpack=True)[10]
            NE = nla*nsa/(nla+nsa)
            NE[1001:lmax+1] = nla[1001:lmax+1]
            wle = np.zeros((lmax+1,lmax+1))
            for l in range(2,lmax+1):
                wle[l,0:l+1] = sig[l,None]/(sig[l,None]+NE[l,None])
        nlm = curvedsky.utils.gauss1alm(lmax,NE[0:lmax+1])
        wElm = (Ealm+nlm)*wle
    else:
        wElm = Ealm
    return wElm


def Emode_diag(i,lmax,Enoise):

    f = prjlib.filename_init(t=Enoise,freq='coadd')
    wle, NE = prjlib.getwElm(2,lmax,t=Enoise,freq='coadd')
    Ealm = pickle.load(open(f.cmb.oalm['E'][i],"rb"))[0:lmax+1,0:lmax+1]
    wElm = Ealm * wle
    return wElm


def getphi(i,p,f,r,gtype='lss',glmax=2008):

    print('load true kappa') # true phi
    glm = np.complex128(hp.fitsfunc.read_alm(f.palm[i]))
    glm = curvedsky.utils.lm_healpy2healpix(len(glm),glm,5100)[:p.lmax+1,:p.lmax+1]

    wlk = np.zeros((p.lmax+1))
    for l in range(p.dlmin,p.dlmax+1):
        wlk[l] = 1./r.kL[l]

    if gtype == 'lss': # multi-tracer phi
        print('load mass tracer kappa')
        Glm = np.load(f.galm[i])
        glm *= 0
        glm[:glmax+1,:glmax+1] = curvedsky.utils.lm_healpy2healpix(len(Glm),Glm,glmax)
        glm[:20,:20] = 0.
        glm *= r.kL[:,None]
        wglm = wlk[:,None]*glm

    if gtype == 'cmb': # reconstructed kappa
        q = 'TT'
        glm, clm = pickle.load(open(f.quad[q].alm[i],"rb"))
        wglm = r.wlk[q]*glm

    return wglm


def func(snmax,olmax,p,f,r,method,gtype='lss',Enoise='la',samemask=True,doremap=False):

    fid = prjlib.filename_init(t='id')
    psa = prjlib.params(t='sa',freq='145')

    if Enoise in ['la','cv']:
        WE = hp.fitsfunc.read_map('/project/projectdirs/sobs/delensing/mask/la.fits')
    if Enoise == 'co':
        WE = hp.fitsfunc.read_map('/project/projectdirs/sobs/delensing/mask/co.fits')

    WSAT = hp.fitsfunc.read_map('/project/projectdirs/sobs/delensing/mask/sa.fits')
    WB = WSAT
    if method=='ideal': WE = 1.

    WE1 = np.average(WE)
    WB1 = np.average(WB)
    cl = np.zeros((snmax,6,olmax+1))

    for i in range(snmax):

        print(i)
        if method == 'wiener':
            wElm = pickle.load(open(f.filt[Enoise].walm['E'][i],"rb"))
        elif method == 'ideal':
            wElm = Emode_ideal(p.dlmax,fid.cmb.oalm['E'][i],Enoise=Enoise)
        else:
            wElm = Emode_diag(i,p.dlmax,Enoise)

        # load phi
        wglm = getphi(i,p,f,r,gtype)

        # construct lensing template
        dalm = curvedsky.delens.lensingb(olmax,p.dlmin,p.dlmax,p.dlmin,p.dlmax,wElm,wglm[:p.dlmax+1,:p.dlmax+1])
        pickle.dump((dalm/WE1),open(f.delens[method,Enoise].alm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        if method!='simple':
            ealm, dalm = multiplywindow(WB,psa.npix,psa.nside,olmax,0*dalm,dalm)
            pickle.dump((dalm/WB1),open(f.delens[method,Enoise].wlm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

        # load B-modes to be delensed at SAT
        Balm = pickle.load(open(fid.cmb.oalm['B'][i],"rb"))[0:olmax+1,0:olmax+1]
        if method=='samemask': ealm, Balm = multiplywindow(WE,p.npix,p.nside,olmax,0*Balm,Balm)
        ealm, Balm = multiplywindow(WB,psa.npix,psa.nside,olmax,0*Balm,Balm)

        # remap
        if doremap: remap(p,wElm,Balm,wglm)

        print('compute cls', i)
        cl[i,0,:] = curvedsky.utils.alm2cl(olmax,Balm)
        cl[i,1,:] = curvedsky.utils.alm2cl(olmax,dalm)
        cl[i,2,:] = curvedsky.utils.alm2cl(olmax,dalm,Balm)
        cl[i,3,:] = curvedsky.utils.alm2cl(olmax,wElm)

    print('save sim')
    mcl = np.mean(cl,axis=0)
    np.savetxt(f.delens[method,Enoise].scl,np.concatenate((r.eL[None,:olmax+1],mcl)).T)


p, f, r = prjlib.analysis_init()
ms = ['ideal','samemask','wiener']

for method in ms:
    func(p.snmax,p.olmax,p,f,r,method,gtype='lss',Enoise='la')
    func(p.snmax,p.olmax,p,f,r,method,gtype='lss',Enoise='co')


