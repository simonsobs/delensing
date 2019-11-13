# map -> alm
import numpy as np
import healpy as hp
import pickle

# from cmblensplus
import curvedsky
import constants
import cmb

# local
import prjlib


def map2alm(nside,lmax,fmap,W,bl):

    Tcmb = constants.Tcmb

    # load map
    Tmap = W*hp.fitsfunc.read_map(fmap,field=0)/Tcmb
    Qmap = W*hp.fitsfunc.read_map(fmap,field=1)/Tcmb
    Umap = W*hp.fitsfunc.read_map(fmap,field=2)/Tcmb

    # map to alm
    alm = {}
    alm['T'] = curvedsky.utils.hp_map2alm(nside,lmax,lmax,Tmap)
    alm['E'], alm['B'] = curvedsky.utils.hp_map2alm_spin(nside,lmax,lmax,2,Qmap,Umap)

    # beam deconvolution
    for m in constants.mtype:
        alm[m] *= 1./ bl[:,None]

    return alm


def map2alm_all(p,f,w):

    # beam
    bl = prjlib.get_beam(p.telescope,p.freq,p.lmax)

    # map -> alm
    for i in range(p.snmin,p.snmax):

        print("map to alm", i, p.telescope)
        salm = map2alm(p.nside,p.lmax,f.cmb.lcdm[i],w,bl)

        if p.telescope == 'id':
            oalm = salm
        else:
            nalm = map2alm(p.nside,p.lmax,f.cmb.nois[i],w,bl)
            oalm = {}
            for m in constants.mtype:
                oalm[m] = salm[m] + nalm[m]

        # save to files
        for m in constants.mtype:
            pickle.dump((oalm[m]),open(f.cmb.oalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            if p.telescope != 'id': 
                pickle.dump((salm[m]),open(f.cmb.salm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump((nalm[m]),open(f.cmb.nalm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def aps(p,f,eL,bc,w,w2,w4):

    # aps for each rlz
    cl, cb = {}, {}
    cl['o'], cb['o'] = cmb.aps(p.snmin,p.snmax,p.bn,p.binspc,p.lmax,f.cmb.oalm)
    if p.telescope != 'id':
        cl['s'], cb['s'] = cmb.aps(p.snmin,p.snmax,p.bn,p.binspc,p.lmax,f.cmb.salm)
        cl['n'], cb['n'] = cmb.aps(p.snmin,p.snmax,p.bn,p.binspc,p.lmax,f.cmb.nalm)
    else:
        cl['s'], cb['s'], cl['n'], cb['n'] = cl['o'], cb['o'], cl['o']*0., cb['o']*0.

    # save to files
    if p.snmax>=2:
        print('save sim')
        mcl, mcb, vcb = {}, {}, {}
        for s in ['o','s','n']:
            mcl[s] = np.mean(cl[s][1:,:,:],axis=0)/w2
            mcb[s] = np.mean(cb[s][1:,:,:],axis=0)/w2
            vcb[s] = np.std(cb[s][1:,:,:],axis=0)/w4
        np.savetxt(f.cmb.scl,np.concatenate((eL[None,:],mcl['o'],mcl['s'],mcl['n'])).T)
        np.savetxt(f.cmb.scb,np.concatenate((bc[None,:],mcb['o'],mcb['s'],mcb['n'],vcb['o'],vcb['s'],vcb['n'])).T)

    if p.snmin==0:
        print('save real')
        np.savetxt(f.cmb.ocl,np.concatenate((eL[None,:],cl['o'][0,:,:]/w2,cl['s'][0,:,:]/w2,cl['n'][0,:,:]/w2)).T)
        np.savetxt(f.cmb.ocb,np.concatenate((bc[None,:],cb['o'][0,:,:]/w2,cb['s'][0,:,:]/w2,cb['n'][0,:,:]/w2)).T)


if __name__ == '__main__':

    # example
    for freq in ['93','145']:
        for t in ['sa','la']: # t should be sa, la or id
            p, f, r = prjlib.analysis_init(t=t,freq=freq) # define parameters, filenames
            w, w2, w4 = prjlib.window(f.cmb.amask,p.telescope) # load window function
            map2alm_all(p,f,w) # map -> alm
            aps(p,f,r.eL,r.bc,w,w2,w4) # calculate cl

