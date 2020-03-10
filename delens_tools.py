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


def Emode_ideal(lmax,felm,t):
    # fullsky with some isotropic noise

    Ealm = pickle.load(open(felm,"rb"))[0:lmax+1,0:lmax+1]
    if t in ['la','co']:
        if t == 'la':
            wle, NE = prjlib.getwElm(2,lmax,t='la',freq='coadd')
        if t == 'co':
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


def getphi(dlmin,dlmax,fpalm,fgalm,kL,gtype='lss',glmax=2008):

    if gtype in ['lss','id']:

        print('load true kappa') # true phi
        glm = np.complex128(hp.fitsfunc.read_alm(fpalm))
        glm = curvedsky.utils.lm_healpy2healpix(len(glm),glm,5100)[:dlmax+1,:dlmax+1]


    if gtype == 'lss': # multi-tracer phi

        wlk = np.zeros((dlmax+1))
        for l in range(dlmin,dlmax+1):
            wlk[l] = 1./kL[l]

        print('load mass tracer kappa')
        Glm = np.load(fgalm)
        glm *= 0
        glm[:glmax+1,:glmax+1] = curvedsky.utils.lm_healpy2healpix(len(Glm),Glm,glmax)
        glm[:20,:20] = 0.
        glm *= kL[:dlmax+1,None]
        wglm = wlk[:,None]*glm

    #if gtype == 'cmb': # reconstructed kappa
    #    q = 'TT'
    #    glm, clm = pickle.load(open(p.quad.f[q].alm[i],"rb"))
    #    wglm = r.wlk[q]*glm

    return wglm


def delens_rlz(t,snmin,snmax,olmax,dlmin,dlmax,kL,p,f,method,fmask='/project/projectdirs/sobs/delensing/mask/',gtype='lss',verbose=True,doremap=False):

    fid = prjlib.filename_init(t='id')
    psa = prjlib.params(t='sa',freq='145')

    if t in ['la','id']:
        WE = hp.fitsfunc.read_map(fmask+'la.fits',verbose=verbose)
    if t == 'co':
        WE = hp.fitsfunc.read_map(fmask+'co.fits',verbose=verbose)

    WSAT = hp.fitsfunc.read_map(fmask+'sa.fits',verbose=verbose)
    WB = WSAT
    if method=='ideal': WE = 1.

    WE1 = np.average(WE)
    WB1 = np.average(WB)
    cl = np.zeros((snmax+1,6,olmax+1))

    for i in range(snmin,snmax+1):

        if verbose: print(i)

        # load E modes
        if method == 'wiener':
            wElm = pickle.load(open(f.cmb.walm['E'][i],"rb"))
        elif method == 'ideal': # fullsky with isotropic noise
            wElm = Emode_ideal(dlmax,fid.cmb.oalm['E'][i],t)
        else: # diagonal filtering
            f = prjlib.filename_init(t=t,freq='coadd')
            wle, __ = prjlib.getwElm(2,dlmax,t=t,freq='coadd')
            wElm = wle * pickle.load(open(f.cmb.oalm['E'][i],"rb"))[0:dlmax+1,0:dlmax+1]

        # load phi
        wglm = getphi(dlmin,dlmax,f.palm[i],f.galm[i],kL,gtype)

        # construct lensing template
        dalm = curvedsky.delens.lensingb(olmax,dlmin,dlmax,dlmin,dlmax,wElm,wglm[:dlmax+1,:dlmax+1])
        pickle.dump((dalm/WE1),open(f.delens[method].alm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        if method!='simple':
            ealm, dalm = multiplywindow(WB,psa.npix,psa.nside,olmax,0*dalm,dalm)
            pickle.dump((dalm/WB1),open(f.delens[method].wlm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

        # load B-modes to be delensed at SAT
        Balm = pickle.load(open(fid.cmb.oalm['B'][i],"rb"))[0:olmax+1,0:olmax+1]
        if method=='samemask': ealm, Balm = multiplywindow(WE,p.npix,p.nside,olmax,0*Balm,Balm)
        ealm, Balm = multiplywindow(WB,psa.npix,psa.nside,olmax,0*Balm,Balm)

        # remap
        if doremap: remap(p,wElm,Balm,wglm)

        if verbose: print('compute cls', i)
        cl[i,0,:] = curvedsky.utils.alm2cl(olmax,Balm)
        cl[i,1,:] = curvedsky.utils.alm2cl(olmax,dalm)
        cl[i,2,:] = curvedsky.utils.alm2cl(olmax,dalm,Balm)
        cl[i,3,:dlmax+1] = curvedsky.utils.alm2cl(dlmax,wElm)

    if verbose: print('save sim')
    mcl = np.mean(cl[1:,:,:],axis=0)
    np.savetxt(f.delens[method].scl,np.concatenate((np.linspace(0,olmax,olmax+1)[None,:olmax+1],mcl)).T)


