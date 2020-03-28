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


def Emode_ideal(lmax,felm,t,lc=1000):
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
            NE = np.zeros(lmax+1)
            NE[:lc+1] = nla[:lc+1]*nsa[:lc+1]/(nla[:lc+1]+nsa[:lc+1])
            NE[lc+1:] = nla[lc+1:lmax+1]
            wle = np.zeros(lmax+1)
            wle[2:] = sig[2:]/(sig[2:]+NE[2:])
        nlm  = curvedsky.utils.gauss1alm(lmax,NE)
        wElm = (Ealm+nlm)*wle[:,None]
    else:
        wElm = Ealm
    return wElm


def getphi(dlmin,dlmax,fpalm,fgalm,kL,gtype='lss',glmax=2008):

    if gtype in ['lss','id']:
        glm = prjlib.load_input_plm(fpalm,dlmax)

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


def prep_klms(pqf,rlz,lmax,wfac=None,remove_mean=True,qlist=['TT','TE','EE','TB','EB','PO','TP'],ep=1e-30):
    
    klm, ulm, iAl = {}, {}, {}

    for q in ['TT','TE','EE','TB','EB']:
        if q in qlist:
            iAl[q] = 1. / ( np.loadtxt(pqf[q].al,unpack=True)[1,:lmax+1] + ep)
            klm[q] = pickle.load(open(pqf[q].alm[rlz],"rb"))[0][:lmax+1,:lmax+1]
            if remove_mean and q!='TE':
                mlm = pickle.load(open(pqf[q].mf,"rb"))[0][:lmax+1,:lmax+1]
                klm[q] -= mlm
            ulm[q] = klm[q] * iAl[q][:,None]
    
    if 'PO' in qlist:
        AlPO = 1./(iAl['EE']+iAl['EB'])
        klm['PO'] = AlPO[:,None]*(ulm['EE']+ulm['EB'])

    if 'TP' in qlist:
        AlTP = 1./(iAl['TT']+iAl['TE']+iAl['EE']+iAl['EB'])
        klm['TP'] = AlTP[:,None]*(ulm['TT']+ulm['TE']+ulm['EE']+ulm['EB'])

    if wfac is not None:
        klm['TP'] = AlTP[:,None]*(ulm['TT']*wfac['fsky']/wfac['w2']+ulm['TE']*wfac['fsky']/wfac['w1']+ulm['EE']+ulm['EB'])
        #klm['TP'] = AlTP[:,None]*(ulm['TT']*wfac['lam4']/wfac['law4']+ulm['TE']*wfac['lam4']/wfac['lam2w2']+ulm['EE']+ulm['EB'])

    return klm


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
            __, dalm = prjlib.multiplywindow(WB,psa.npix,psa.nside,olmax,0*dalm,dalm)
            pickle.dump((dalm/WB1),open(f.delens[method].wlm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

        # load B-modes to be delensed at SAT
        Balm = pickle.load(open(fid.cmb.oalm['B'][i],"rb"))[0:olmax+1,0:olmax+1]
        if method=='samemask':  __, Balm = prjlib.multiplywindow(WE,p.npix,p.nside,olmax,0*Balm,Balm)
        __, Balm = prjlib.multiplywindow(WB,psa.npix,psa.nside,olmax,0*Balm,Balm)

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


