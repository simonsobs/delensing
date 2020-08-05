# map -> alm
import numpy as np
import healpy as hp
import pickle
import os
import sys
import tqdm

# from SO pipeline
from mapsims import SONoiseSimulator
#from mapsims import SOStandalonePrecomputedCMB
#from mapsims import Channel as SOChannel
#from mapsims import noise

# from cmblensplus/wrap/
import curvedsky as CS

# from cmblensplus/utils/
import constants
import cmb
import misctools

# local
import prjlib


class sim_map:

    def __init__(self,doreal=False,telescope='la',ntype='base',submap='00',snmin=0,snmax=100,overwrite=False,verbose=True):

        self.telescope = str.upper(telescope)
        if not telescope in ['sa','la']:
            sys.exit('only SAT and LAT are supported for mbs sim')
        
        self.doreal = doreal
        self.rlz = np.linspace(snmin,snmax,snmax-snmin+1,dtype=np.int)

        self.overwrite = overwrite
        self.verbose = verbose

        self.ntype  = ntype
        if 'base' in ntype:
            self.mode = 'baseline'
        elif 'goal' in ntype:
            self.mode = 'goal'
        elif ntype == '':
            self.mode = 'signal'
            print('signal calculation')
        else:
            sys.exit('unknown noise level')

        if 'roll' in ntype:
            self.roll = int(ntype[ntype.find('roll')+4:])
        else:
            self.roll = 0
        
        # tube (LT or ST)
        self.tube = str.upper(telescope[0]) + 'T' + submap[0]
        self.tube_subid = int(submap[1])

        self.nside, self.npix = prjlib.mapres(telescope)

        # set directory
        d = prjlib.data_directory()
        d_map = d['cmb'] + 'map/'

        # output map filename
        ids = prjlib.rlz_index(doreal=doreal)
        if ntype == '':
            #cmb signal map
            self.fmap = [d_map+'/cmb_uKCMB_'+telescope+submap+'_nside'+str(self.nside)+'_'+x+'.fits' for x in ids]
        else:
            #cmb noise map
            self.fmap = [d_map+'/noise_uKCMB_'+telescope+submap+'_'+ntype+'_nside'+str(self.nside)+'_'+x+'.fits' for x in ids]


    def SOsim(self):
        # Simulate CMB and noise maps

        #ch  = SOChannel(self.telescope,self.band)

        if self.verbose:
            print(self.mode,self.roll)

        for i in tqdm.tqdm(self.rlz,ncols=100,desc='generate map'):
    
            if misctools.check_path(self.fmap[i],overwrite=self.overwrite,verbose=self.verbose): continue

            if self.ntype == '':
                # To avoid bugs in MBS code for loading signals, here we directly load alm files and convert it to a map.
                # load alm
                sim = SONoiseSimulator(nside=self.nside, apply_beam_correction=False, rolloff_ell=self.roll)

                fname = "/global/project/projectdirs/sobs/v4_sims/mbs/cmb/fullskyLensedUnabberatedCMB_alm_set00_"+str(i).zfill(5)+".fits"
                Tlm, Elm, Blm = np.complex128( hp.read_alm( fname, hdu=(1, 2, 3) ) )

                # beam
                lmax  = hp.sphtfunc.Alm.getlmax( len(Tlm) )
                l = np.linspace(0,lmax,lmax+1)
                theta = sim.tubes[self.tube][self.tube_subid].beam / sim.tubes[self.tube][self.tube_subid].beam._unit
                bl = np.exp( -l*(l+1)*(theta*np.pi/10800.)**2/16./np.log(2.) )

                # convolution and produce a map
                Tlm = hp.almxfl( Tlm, bl )
                Elm = hp.almxfl( Elm, bl )
                Blm = hp.almxfl( Blm, bl )
                map = hp.alm2map( np.array((Tlm,Elm,Blm)), self.nside, verbose=False )

            else:
                # noise simulation
                sim = SONoiseSimulator(nside=self.nside, apply_beam_correction=False, sensitivity_mode=self.mode, rolloff_ell=self.roll)
                map = sim.simulate(tube=self.tube)[self.tube_subid][0]

            # save to file
            hp.fitsfunc.write_map(self.fmap[i],map,overwrite=True)


def map2alm_core(nside,lmax,fmap,w,bl):

    Tcmb = constants.Tcmb

    # load map
    Tmap = w * hp.fitsfunc.read_map(fmap,field=0,verbose=False)/Tcmb
    Qmap = w * hp.fitsfunc.read_map(fmap,field=1,verbose=False)/Tcmb
    Umap = w * hp.fitsfunc.read_map(fmap,field=2,verbose=False)/Tcmb

    # map to alm
    alm = {}
    alm['T'] = CS.utils.hp_map2alm(nside,lmax,lmax,Tmap)
    alm['E'], alm['B'] = CS.utils.hp_map2alm_spin(nside,lmax,lmax,2,Qmap,Umap)

    # beam deconvolution
    for m in constants.mtype:
        alm[m] /= bl[:,None]

    return alm


def map2alm(t,rlz,submap,nside,lmax,fcmb,w,verbose=True,overwrite=False,mtype=['T','E','B'],roll=2):

    # beam
    bl = prjlib.get_beam(t,submap,lmax)

    # map -> alm
    for i in tqdm.tqdm(rlz,ncols=100,desc='map2alm (submap id='+submap+')'):

        if not overwrite and os.path.exists(fcmb.alms['o']['T'][i]) and os.path.exists(fcmb.alms['o']['E'][i]) and os.path.exists(fcmb.alms['o']['B'][i]):
            if verbose: print('Files exist:',fcmb.alms['o']['T'][i],'and E/B')
            continue

        salm = map2alm_core(nside,lmax,fcmb.lcdm[i],w,bl)

        if t == 'id':
            oalm = salm.copy()
        else:
            nalm = map2alm_core(nside,lmax,fcmb.nois[i],w,bl)
            oalm = {}
            for m in mtype:
                oalm[m] = salm[m] + nalm[m]
                # remove low-ell for roll-off effect
                if roll > 2:
                    oalm[m][:roll,:] = 0.

        # save to files
        for m in mtype:
            pickle.dump((oalm[m]),open(fcmb.alms['o'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            if t != 'id': 
                pickle.dump((salm[m]),open(fcmb.alms['s'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump((nalm[m]),open(fcmb.alms['n'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def alm_comb_submap(rlz,fcmbsub,fcmbcomb,verbose=True,overwrite=False,submaps=['00'],mtype=[(0,'T'),(1,'E'),(2,'B')],roll=2):
    
    for i in tqdm.tqdm(rlz,ncols=100,desc='alm combine'):

        for (mi, m) in mtype:

            if misctools.check_path(fcmbcomb.alms['o'][m][i],overwrite=overwrite,verbose=verbose): continue

            salm, nalm, Wl = 0., 0., 0.
            for sub in submaps:
                Nl = np.loadtxt(fcmbsub[sub].scl['n'],unpack=True)[mi+1]
                Nl[0:2] = 1.
                Il = 1./Nl
                salm += pickle.load(open(fcmbsub[sub].alms['s'][m][i],"rb")) * Il[:,None]
                nalm += pickle.load(open(fcmbsub[sub].alms['n'][m][i],"rb")) * Il[:,None]
                Wl   += Il
            salm /= Wl[:,None]
            nalm /= Wl[:,None]
            oalm = salm + nalm

            # remove low-ell for roll-off effect
            if roll > 2:
                oalm[:roll,:] = 0.

            pickle.dump((salm),open(fcmbcomb.alms['s'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((nalm),open(fcmbcomb.alms['n'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((oalm),open(fcmbcomb.alms['o'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def aps(rlz,lmax,fcmb,w2,stype=['o','s','n'],mtype=['T','E','B'],**kwargs_ov):

    # compute aps for each rlz
    L = np.linspace(0,lmax,lmax+1)
    
    for s in stype:
        
        if misctools.check_path(fcmb.scl[s],**kwargs_ov): continue
        
        if kwargs_ov['verbose']: print('stype =',s)
        
        cl = cmb.aps(rlz,lmax,fcmb.alms[s],odd=False,mtype=mtype,**kwargs_ov,w2=w2,fname=fcmb.cl[s])

        # save average to files
        mcl = np.mean(cl,axis=0)
        vcl = np.std(cl,axis=0)
        np.savetxt(fcmb.scl[s],np.concatenate((L[None,:],mcl,vcl)).T)


def apsx(rlz,lmax,fcmb,gcmb,w2,**kwargs_ov):

    xl = cmb.apsx(rlz,lmax,fcmb.alms['o'],gcmb.alms['o'],**kwargs_ov)/w2

    # save average to files
    L = np.linspace(0,lmax,lmax+1)
    mxl = np.mean(xl,axis=0)
    vxl = np.std(xl,axis=0)
    np.savetxt(fcmb.scl['x'],np.concatenate((L[None,:],mxl,vxl)).T)



#////////////////////////////////////////////////////////////////////////////////
# Wiener filter
#////////////////////////////////////////////////////////////////////////////////

class wiener_objects:
    # define parameteres and arrays for wiener filtering

    def __init__(self,t,tqu,submaps,ntype,nside):

        # telescope ("la" or "co")
        self.t     = t
        
        # tqu = 1 for T and 2 for Q/U
        self.tqu   = tqu
        
        # list of submaps to be combined
        self.submaps = submaps
        
        # nside of maps
        self.nside = nside

        # noise model
        if self.t=='la':
            self.Nside = 2048
            self.lmax  = 4096
            if 'base' in ntype:
                self.sigma = np.array([8.,10.,22.])
            if 'goal' in ntype:
                self.sigma = np.array([5.8,6.3,15.])
                
        if self.t=='sa':
            self.Nside = 512
            self.lmax  = 2048
            if 'base' in ntype:
                self.sigma = np.array([2.6,3.3,6.3])
            if 'goal' in ntype:
                self.sigma = np.array([1.9,2.1,4.2])

        # pixel number of map
        self.npix  = 12*self.nside**2

        # beam function + pixel window
        self.bl = prjlib.get_beams(self.t,self.lmax,self.submaps)
        if self.nside != self.Nside:
            self.bl *= hp.sphtfunc.pixwin(self.nside)[:self.lmax+1] / hp.sphtfunc.pixwin(self.Nside)[:self.lmax+1]

        # prepare arrays
        self.maps = np.zeros((self.tqu,len(self.submaps),self.npix))
        self.invN = np.zeros((self.tqu,len(self.submaps),self.npix))

        # hitcount map to be used for inverse noise modeling
        self.W = prjlib.hitmap(self.t,self.nside)
        
        # define survey boundary
        self.M, __ = prjlib.window(self.t,nside=self.nside,ascale=0.)


    def load_maps(self,fmap,i,Tcmb=2.72e6,verbose=False):

        # loading submaps and multiply survey boundary
        
        for ki, submap in enumerate(self.submaps):
        
            if self.tqu == 1:  # temperature

                Ts = hp.fitsfunc.read_map(fmap[submap].lcdm[i],field=0,verbose=verbose)
                Tn = hp.fitsfunc.read_map(fmap[submap].nois[i],field=0,verbose=verbose)
                self.maps[0,ki,:] = self.M * hp.pixelfunc.ud_grade(Ts+Tn,self.nside)/Tcmb
        
            if self.tqu == 2:  # polarization
        
                Qs = hp.fitsfunc.read_map(fmap[submap].lcdm[i],field=1,verbose=verbose)
                Us = hp.fitsfunc.read_map(fmap[submap].lcdm[i],field=2,verbose=verbose)
                Qn = hp.fitsfunc.read_map(fmap[submap].nois[i],field=1,verbose=verbose)
                Un = hp.fitsfunc.read_map(fmap[submap].nois[i],field=2,verbose=verbose)

                self.maps[0,ki,:] = self.M * hp.pixelfunc.ud_grade(Qs+Qn,self.nside)/Tcmb
                self.maps[1,ki,:] = self.M * hp.pixelfunc.ud_grade(Us+Un,self.nside)/Tcmb


    def load_invN(self,Tcmb=2.72e6):  
        
        # inv noise covariance at each submap constructed from hitcount map

        for ki, sigma in enumerate(self.sigma):

            # temperature
            self.invN[0,ki,:] = self.W * (sigma*(np.pi/10800.)/Tcmb)**(-2)

            # polarization
            if self.tqu == 2:
                self.invN[:,ki,:] *= 2.
                self.invN[1,ki,:] = self.invN[0,ki,:]



def cinv_core(i,t,wla,wsa,lmax,falm,cl,lTmax=1000,lTcut=100,**kwargs):

    mn  = len(wla.bl[:,0])

    # temperature
    if wla.tqu==1:
        if t == 'la':
            cl[0,:lTcut+1] = 0.
            Tlm = CS.cninv.cnfilter_freq(1,mn,wla.nside,lmax,cl[0:1,:],wla.bl,wla.invN,wla.maps,**kwargs)
            pickle.dump((Tlm),open(falm['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    # polarization
    if wla.tqu==2:
        if t == 'co':
            Elm, Blm = CS.cninv.cnfilter_freq_nside(2,mn,mn,wla.nside,wsa.nside,lmax,cl[1:3,:],wla.bl[:,:lmax+1],wsa.bl,wla.invN,wsa.invN,wla.maps,wsa.maps,**kwargs)
        if t == 'la':
            Elm, Blm = CS.cninv.cnfilter_freq(2,mn,wla.nside,lmax,cl[1:3,:],wla.bl,wla.invN,wla.maps,**kwargs)

        pickle.dump((Elm),open(falm['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((Blm),open(falm['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def cinv(tqu,rlz,t,lmax,ntype,fmap,falm,cl,submaps=[],fmapsa='',overwrite=False,verbose=False,**kwargs):

    # prepare objects for wiener filtering
    if t == 'la':
        wla = wiener_objects('la',tqu,submaps,ntype,kwargs['nsides'][0])
    if t == 'co':
        wla = wiener_objects('la',tqu,submaps,ntype,kwargs['nsides0'][0])
    wiener_objects.load_invN(wla)

    wsa = None
    if t == 'co':
        wsa = wiener_objects('sa',tqu,submaps,ntype,kwargs['nsides1'][0])
        wiener_objects.load_invN(wsa)

    # roll-off effect
    roll = int(ntype[ntype.find('roll')+4:])
    if roll > 2:
        cl[:3,:roll] = 0. 

    # start loop for realizations
    for i in tqdm.tqdm(rlz,ncols=100,desc='cinv'):

        if misctools.check_path(falm['E'][i],overwrite=overwrite): continue

        wiener_objects.load_maps(wla,fmap,i)
        if t=='co':
            wiener_objects.load_maps(wsa,fmapsa,i)

        cinv_core(i,t,wla,wsa,lmax,falm,cl[:4,:lmax+1],verbose=verbose,**kwargs)


def iso_noise(rlz,lmin,lmax,fslm,falm,ncls,mtype=['T','E','B'],**kwargs_ov):

    # generate isotropic noise from a given noise spectrum and add it to fullsky signal
    for i in tqdm.tqdm(rlz,ncols=100,desc='iso noise'):

        for mi, m in enumerate(mtype): # loop for T/E/B

            if misctools.check_path(falm[m][i],**kwargs_ov): continue

            alm = pickle.load(open(fslm[m][i],"rb"))
            alm += CS.utils.gauss1alm(lmax,ncls[mi,:])
            pickle.dump((alm),open(falm[m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def interface(kwargs_ov={},kwargs_cmb={},run=[]):

    telescope = kwargs_cmb['t']
    snmin = kwargs_cmb['snmin']
    snmax = kwargs_cmb['snmax']
    ntype = kwargs_cmb['ntype']

    #submaps = ['00','01','10','11','20','21','30','31','40','41','50','51']
    submaps = ['20','21','30','31','40','41','50','51']

    
    if 'simmap' in run:

        # simulate CMB map for LAT and SAT
        
        if telescope in ['la','sa']:

            for submap in submaps:
                
                # signal sim
                sobj = sim_map(telescope=telescope,submap=submap,snmin=snmin,snmax=snmax,ntype='',**kwargs_ov)
                sim_map.SOsim(sobj)
                # noise sim
                sobj = sim_map(telescope=telescope,submap=submap,snmin=snmin,snmax=snmax,ntype=ntype,**kwargs_ov)
                sim_map.SOsim(sobj)


    if 'calcalm' in run:  
        
        # map -> alm for each submap (and com) and telescope

        if '_iso' in ntype:  
            # This noise type is used for checking impact of non-isotropic noise
            # compute alms for isotropic noise
            # need pre-computed submap-coadded spectrum and "id" case

            if kwargs_cmb['fltr'] != 'none':
                sys.exit('isotropic noise calculation is only valid for none-filtered case')
    
            if telescope in ['la','co']:

                # for isotropic noise spectrum and diagonal wiener filtering
                pc = prjlib.analysis_init(t=telescope,submap='com',snmin=snmin,snmax=snmax,ntype=ntype.replace('_iso',''))
                ncl = prjlib.loadocl(pc.fcmb.scl['n'],lTmin=pc.lTmin,lTmax=pc.lTmax)

                # setup filenames for input and output
                inp = prjlib.analysis_init(t='id',ntype='cv',snmin=snmin,snmax=snmax) # to use fullsky signal
                out = prjlib.analysis_init(t=telescope,submap='com',fltr='none',snmin=snmin,snmax=snmax,ntype=ntype)

                # alm and aps
                iso_noise(pc.rlz,pc.roll,pc.lmax,inp.fcmb.alms['o'],out.fcmb.alms['o'],ncl[0:3,:],**kwargs_ov)
                aps(pc.rlz,pc.lmax,out.fcmb,1.,stype=['o'],**kwargs_ov)

        else:

            if kwargs_cmb['fltr'] == 'none':
                
                # works for "la", "sa" and "id"

                if telescope == 'co':
                    sys.exit('does not support none filter case for LA+SA')
        
                if telescope == 'id': # map -> alm for fullsky case
                    stype = ['o']
                    ntype = 'cv'
                    submaps = ['00']
                else:
                    stype = ['s','n','o']

                # load survey window
                w_mask, __ = prjlib.window(telescope,ascale=kwargs_cmb['ascale'])

                # map -> alm for each submap
                for submap in submaps:
                    pobj = prjlib.analysis_init(t=telescope,submap=submap,snmin=snmin,snmax=snmax,ntype=ntype) # define parameters, filenames
                    
                    sim = SONoiseSimulator(nside=pobj.nside)
                    w_hit, __ = sim.get_hitmaps(tube=str.upper(telescope[0])+'T'+submap[0])
                
                    w = w_mask * w_hit[int(submap[1])]
                    wn = prjlib.calc_wfactor(w)

                    map2alm(pobj.telescope,pobj.rlz,submap,pobj.nside,pobj.lmax,pobj.fcmb,w,roll=pobj.roll,**kwargs_ov) # map -> alm
                    aps(pobj.rlz,pobj.lmax,pobj.fcmb,wn[2],stype=stype,**kwargs_ov)

                # combine alm over submaps
                if telescope in ['la','sa']:
                    
                    pobj = prjlib.analysis_init(t=telescope,submap='com',snmin=snmin,snmax=snmax,ntype=ntype)
                    fmap = prjlib.filename_submaps(submaps,t=telescope,ntype=ntype)
                    alm_comb_submap(pobj.rlz,fmap,pobj.fcmb,roll=pobj.roll,submaps=submaps,**kwargs_ov)
                    aps(pobj.rlz,pobj.lmax,pobj.fcmb,wn[2],**kwargs_ov)


            elif kwargs_cmb['fltr'] == 'cinv':  # full wiener filtering

                # works for "la" and "co"

                if telescope == 'sa':
                    sys.exit('does not support cinv filter case for SA')

                if telescope == 'id':
                    sys.exit('does not support cinv filter case for idealistic case')

                pw = prjlib.analysis_init(t=telescope,submap='com',fltr='cinv',snmin=snmin,snmax=snmax,ntype=ntype)
        
                if telescope == 'la':

                    mtypes = ['T','E','B']

                    # filenames
                    fmap = prjlib.filename_submaps(submaps,t=telescope,ntype=ntype)
 
                    # Temperature
                    cinv_params = {\
                        'chn' : 7, \
                        'eps' : [1e-4,.1,.1,.1,.1,.1,0.], \
                        'lmaxs' : [4096,2048,2048,1024,512,256,20], \
                        'nsides' : [2048,2048,1024,512,256,128,64], \
                        'itns' : [100,5,5,5,5,5,0], \
                        'ro' : 1, \
                        'filter' : 'W' \
                    }
                    cinv(1,pw.rlz,telescope,4096,ntype,fmap,pw.fcmb.alms['o'],pw.lcl,submaps=submaps,**cinv_params,**kwargs_ov)

                    # Polarization
                    cinv_params = {\
                        'chn' : 6, \
                        'eps' : [1e-5,.1,.1,.1,.1,0.], \
                        'lmaxs' : [4096,2048,1024,512,256,20], \
                        'nsides' : [2048,1024,512,256,128,64], \
                        'itns' : [100,7,5,3,3,0], \
                        'ro' : 1, \
                        'filter' : 'W' \
                    }
                    cinv(2,pw.rlz,telescope,4096,ntype,fmap,pw.fcmb.alms['o'],pw.lcl,submaps=submaps,**cinv_params,**kwargs_ov)


                if telescope == 'co':
        
                    mtypes = ['E','B']
                    cinv_params = {\
                        'chn' : 6, \
                        'eps' : [1e-5,.1,.1,.1,.1,0.], \
                        'lmaxs' : [2048,1000,400,200,100,20], \
                        'nsides0' : [1024,512,256,128,128,64], \
                        'nsides1' : [512,256,256,128,64,64], \
                        'itns' : [200,9,3,3,7,0], \
                        'ro' : 1, \
                        'reducmn' : 2, \
                        'filter' : 'W' \
                    }
                    cinv_params = {\
                        'chn' : 1, \
                        'eps' : [1e-5], \
                        'lmaxs' : [2048], \
                        'nsides0' : [1024], \
                        'nsides1' : [512], \
                        'itns' : [1000], \
                        'ro' : 1, \
                        'reducmn' : 0, \
                        'filter' : 'W' \
                    }

                    fmapla = prjlib.filename_submaps(submaps,t='la',ntype=ntype)
                    fmapsa = prjlib.filename_submaps(submaps,t='sa',ntype=ntype)
 
                    cinv(2,pw.rlz,telescope,2048,ntype,fmapla,pw.fcmb.alms['o'],pw.lcl,submaps=submaps,fmapsa=fmapsa,**cinv_params,**kwargs_ov)
            
                # compute aps
                pI = prjlib.analysis_init(t='id',ntype='cv',snmin=snmin,snmax=snmax) # for cross with input
                wn = prjlib.wfac(telescope,binary=True)

                aps(pw.rlz,pw.lmax,pw.fcmb,wn[0],stype=['o'],mtype=mtypes,**kwargs_ov)
                apsx(pw.rlz,pw.lmax,pw.fcmb,pI.fcmb,wn[0],mtype=mtypes,**kwargs_ov)



