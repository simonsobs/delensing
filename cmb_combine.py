# * map -> alm
import numpy as np
import healpy as hp
import curvedsky
import prjlib
import pickle
import constants
import cmb
from memory_profiler import profile


def alm_comb_freq(p,f,lcut=5000):
    
    t  = p.telescope

    for i in range(p.snmin,p.snmax):
        print("map to alm", i, t)
        for mi, m in enumerate(constants.mtype):
            salm = 0.
            nalm = 0.
            Wl   = 0.
            #for freq in ['93','145','225','280']:
            for freq in ['93','145']:
                f0 = prjlib.filename_init(t=t,freq=freq)
                Nl = np.loadtxt(f0.cmb.scl,unpack=True)[mi+9]
                Nl[0:2] = 1.
                Il = 1./Nl
                salm += pickle.load(open(f0.cmb.salm[m][i],"rb"))*Il[:,None]
                nalm += pickle.load(open(f0.cmb.nalm[m][i],"rb"))*Il[:,None]
                Wl   += Il
            salm *= 1./Wl[:,None]
            nalm *= 1./Wl[:,None]
            if lcut < p.lmax:
                salm[lcut+1:,:] = 0.
                nalm[lcut+1:,:] = 0.
            oalm = salm + nalm
            pickle.dump((oalm),open(f.cmb.oalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((salm),open(f.cmb.salm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((nalm),open(f.cmb.nalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def alm_comb_telescope(p,f,r,salcut=1000):
    
    pla = prjlib.params(t='la',freq='coadd')
    psa = prjlib.params(t='sa',freq='coadd')

    fla = prjlib.filename(pla)
    fsa = prjlib.filename(psa)

    wla, wla2, wla4 = prjlib.window(fla.cmb.amask,'la')
    wsa, wsa2, wsa4 = prjlib.window(fsa.cmb.amask,'sa')
    #wsa = hp.pixelfunc.ud_grade(wsa,pla.nside)
    #hp.fitsfunc.write_map(f.cmb.amask,wla+wsa,overwrite=True)

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


def aps(p,f,r):

    w, w2, w4 = prjlib.window(f.cmb.amask,p.telescope)

    # aps for each rlz
    cl, cb = {}, {}
    cl['o'], cb['o'] = cmb.aps(p.snmin,p.snmax,p.bn,p.binspc,p.lmax,f.cmb.oalm)
    cl['s'], cb['s'] = cmb.aps(p.snmin,p.snmax,p.bn,p.binspc,p.lmax,f.cmb.salm)
    cl['n'], cb['n'] = cmb.aps(p.snmin,p.snmax,p.bn,p.binspc,p.lmax,f.cmb.nalm)

    # save to files
    if p.snmax>=2:
        print('save sim')
        mcl, mcb, vcb = {}, {}, {}
        for s in ['o','s','n']:
            mcl[s] = np.mean(cl[s][1:,:,:],axis=0)/w2
            mcb[s] = np.mean(cb[s][1:,:,:],axis=0)/w2
            vcb[s] = np.std(cb[s][1:,:,:],axis=0)/w2
        np.savetxt(f.cmb.scl,np.concatenate((r.eL[None,:],mcl['o'],mcl['s'],mcl['n'])).T)
        np.savetxt(f.cmb.scb,np.concatenate((r.bc[None,:],mcb['o'],mcb['s'],mcb['n'],vcb['o'],vcb['s'],vcb['n'])).T)

    if p.snmin==0:
        print('save real')
        np.savetxt(f.cmb.ocl,np.concatenate((r.eL[None,:],cl['o'][0,:,:]/w2,cl['s'][0,:,:]/w2,cl['n'][0,:,:]/w2)).T)
        np.savetxt(f.cmb.ocb,np.concatenate((r.bc[None,:],cb['o'][0,:,:]/w2,cb['s'][0,:,:]/w2,cb['n'][0,:,:]/w2)).T)



if __name__ == '__main__':

    for t, lc in [('sa',1000),('la',3000)]:
        p, f, r = prjlib.analysis_init(t=t,freq='coadd')
        alm_comb_freq(p,f,lcut=lc)
        aps(p,f,r)

    p, f, r = prjlib.analysis_init(t='co',freq='coadd')
    alm_comb_telescope(p,f,r)
    aps(p,f,r)


