# map -> alm
import numpy as np
import healpy as hp
import pickle
import os
import sys

# from SO pipeline
from mapsims import SONoiseSimulator
from mapsims import SOStandalonePrecomputedCMB
from mapsims import SOChannel
from mapsims import noise

# from cmblensplus/wrap/
import curvedsky as cs

# from cmblensplus/utils/
import constants
import cmb
import misctools

# local
import prjlib


class sim_map:

    def __init__(self,doreal=False,telescope='la',ntype='base',freq='145',snmin=0,snmax=100,overwrite=False,verbose=True):

        self.telescope = str.upper(telescope)
        self.ntype  = ntype
        self.band   = int(freq)
        self.doreal = doreal
        self.rlz = np.linspace(snmin,snmax,snmax-snmin+1,dtype=np.int)
        self.overwrite = overwrite
        self.verbose = verbose

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

        self.nside, self.npix = prjlib.mapres(telescope)

        # set directory
        d = prjlib.data_directory()
        d_map = d['cmb'] + 'map/'

        # map filename
        ids = prjlib.rlz_index(doreal=doreal)
        if ntype == '':
            #cmb signal map
            if telescope == 'id': # use LAT signal sim
                self.fmap = [d_map+'/cmb_uKCMB_la145_nside'+str(self.nside)+'_'+x+'.fits' for x in ids]
            else:
                self.fmap = [d_map+'/cmb_uKCMB_'+telescope+freq+'_nside'+str(self.nside)+'_'+x+'.fits' for x in ids]
        else:
            #cmb noise map
            self.fmap = [d_map+'/noise_uKCMB_'+telescope+freq+'_'+ntype+'_nside'+str(self.nside)+'_'+x+'.fits' for x in ids]


    def SOsim(self):
        # Simulate CMB and noise maps

        ch  = SOChannel(telescope=self.telescope,band=self.band)
    
        if self.verbose:  print(self.mode,self.roll)

        for i in self.rlz:
    
            if misctools.check_path(self.fmap[i],overwrite=self.overwrite,verbose=self.verbose): continue
            if self.verbose:  misctools.progress(i,self.rlz,addtext='sim map for '+self.mode)
        
            if self.ntype == '':
                # signal simulation
                sim = SOStandalonePrecomputedCMB(i,nside=self.nside,input_units='uK_CMB')
                map = SOStandalonePrecomputedCMB.simulate(sim,ch)
            else:
                # noise simulation
                sim = SONoiseSimulator(nside=self.nside,apply_beam_correction=False,sensitivity_mode=self.mode,rolloff_ell=self.roll)
                map = SONoiseSimulator.simulate(sim,ch)

            # save to file
            hp.fitsfunc.write_map(self.fmap[i],map,overwrite=True)


def output_hitmap(**kwargs_ov):

    for telescope in ['LA','SA']:

        nside, __ = prjlib.mapres(telescope)
        f = prjlib.hitmap_filename(telescope,nside)

        if misctools.check_path(f,**kwargs_ov): continue

        s = noise.SONoiseSimulator(nside)
        w = s.hitmap[telescope]
    
        hp.fitsfunc.write_map(f,w,overwrite=kwargs_ov['overwrite'])


def map2alm_core(nside,lmax,fmap,w,bl):

    Tcmb = constants.Tcmb

    # load map
    Tmap = w * hp.fitsfunc.read_map(fmap,field=0,verbose=False)/Tcmb
    Qmap = w * hp.fitsfunc.read_map(fmap,field=1,verbose=False)/Tcmb
    Umap = w * hp.fitsfunc.read_map(fmap,field=2,verbose=False)/Tcmb

    # map to alm
    alm = {}
    alm['T'] = cs.utils.hp_map2alm(nside,lmax,lmax,Tmap)
    alm['E'], alm['B'] = cs.utils.hp_map2alm_spin(nside,lmax,lmax,2,Qmap,Umap)

    # beam deconvolution
    for m in constants.mtype:
        alm[m] /= bl[:,None]

    return alm


def map2alm(t,rlz,freq,nside,lmax,fcmb,w,verbose=True,overwrite=False,mtype=['T','E','B'],roll=2):

    # beam
    bl = prjlib.get_beam(t,freq,lmax)

    # map -> alm
    for i in rlz:

        if not overwrite and os.path.exists(fcmb.alms['o']['T'][i]) and os.path.exists(fcmb.alms['o']['E'][i]) and os.path.exists(fcmb.alms['o']['B'][i]):
            if verbose: print('Files exist:',fcmb.alms['o']['T'][i],'and E/B')
            continue

        if verbose: print("map to alm", i, t)
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



def alm_comb_freq(rlz,fcmbfreq,fcmbcomb,verbose=True,overwrite=False,freqs=['93','145','225'],mtype=[(0,'T'),(1,'E'),(2,'B')],roll=2):
    
    for i in rlz:

        if verbose: print("map to alm", i)
        for (mi, m) in mtype:

            if misctools.check_path(fcmbcomb.alms['o'][m][i],overwrite=overwrite,verbose=verbose): continue

            salm, nalm, Wl = 0., 0., 0.
            for freq in freqs:
                Nl = np.loadtxt(fcmbfreq[freq].scl['n'],unpack=True)[mi+1]
                Nl[0:2] = 1.
                Il = 1./Nl
                salm += pickle.load(open(fcmbfreq[freq].alms['s'][m][i],"rb"))*Il[:,None]
                nalm += pickle.load(open(fcmbfreq[freq].alms['n'][m][i],"rb"))*Il[:,None]
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



def getbeam(t,lmax,nu=['93','145','225']):

    bl = np.ones((len(nu),lmax+1))
    
    for ki, freq in enumerate(nu):
        bl[ki,:] = prjlib.get_beam(t,freq,lmax)
    return bl


#////////////////////////////////////////////////////////////////////////////////
# Wiener filter
#////////////////////////////////////////////////////////////////////////////////

class wiener_objects:

    def __init__(self,t,tqu,freqs,ntype,nside):

        self.t     = t
        self.tqu   = tqu
        self.freqs = freqs
        self.nside = nside

        if t=='la':
            self.Nside = 2048
            self.lmax  = 4096
            if 'base' in ntype:
                self.sigma = np.array([8.,10.,22.])
            if 'goal' in ntype:
                self.sigma = np.array([5.8,6.3,15.])
                
        if t=='sa':
            self.Nside = 512
            self.lmax  = 2048
            if 'base' in ntype:
                self.sigma = np.array([2.6,3.3,6.3])
            if 'goal' in ntype:
                self.sigma = np.array([1.9,2.1,4.2])

        self.npix  = 12*self.nside**2

        self.bl = getbeam(t,self.lmax,nu=freqs)
        if self.nside != self.Nside:
            self.bl *= hp.sphtfunc.pixwin(self.nside)[:self.lmax+1] / hp.sphtfunc.pixwin(self.Nside)[:self.lmax+1]

        self.maps = np.zeros((tqu,len(freqs),self.npix))
        self.invN = np.zeros((tqu,len(freqs),self.npix))

        self.M, __ = prjlib.window(t,nside=self.nside,ascale=0.)
        self.W = prjlib.hitmap(t,self.nside) 


    def load_maps(self,fmap,i,Tcmb=2.72e6,verbose=False):

        for ki, freq in enumerate(self.freqs):
        
            if self.tqu == 1:
            
                Ts = hp.fitsfunc.read_map(fmap[freq].lcdm[i],field=0,verbose=verbose)
                Tn = hp.fitsfunc.read_map(fmap[freq].nois[i],field=0,verbose=verbose)
                self.maps[0,ki,:] = self.M * hp.pixelfunc.ud_grade(Ts+Tn,self.nside)/Tcmb
        
            if self.tqu == 2:
        
                Qs = hp.fitsfunc.read_map(fmap[freq].lcdm[i],field=1,verbose=verbose)
                Us = hp.fitsfunc.read_map(fmap[freq].lcdm[i],field=2,verbose=verbose)
                Qn = hp.fitsfunc.read_map(fmap[freq].nois[i],field=1,verbose=verbose)
                Un = hp.fitsfunc.read_map(fmap[freq].nois[i],field=2,verbose=verbose)

                self.maps[0,ki,:] = self.M * hp.pixelfunc.ud_grade(Qs+Qn,self.nside)/Tcmb
                self.maps[1,ki,:] = self.M * hp.pixelfunc.ud_grade(Us+Un,self.nside)/Tcmb


    def load_invN(self,Tcmb=2.72e6):  # inv noise covariance

        for ki, sigma in enumerate(self.sigma):

            self.invN[0,ki,:] = self.W * (sigma*(np.pi/10800.)/Tcmb)**(-2)

            if self.tqu == 2:
                self.invN[:,ki,:] *= 2.
                self.invN[1,ki,:] = self.invN[0,ki,:]



def cinv_core(i,t,wla,wsa,lmax,falm,cl,lTmax=1000,lTcut=100,**kwargs):

    mn  = len(wla.bl[:,0])

    if wla.tqu==1:
        if t == 'la':
            cl[0,:lTcut+1] = 0.
            Tlm = cs.cninv.cnfilter_freq(1,mn,wla.nside,lmax,cl[0:1,:],wla.bl,wla.invN,wla.maps,**kwargs)
            pickle.dump((Tlm),open(falm['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    if wla.tqu==2:
        if t == 'co':
            Elm, Blm = cs.cninv.cnfilter_freq_nside(2,mn,mn,wla.nside,wsa.nside,lmax,cl[1:3,:],wla.bl,wsa.bl,wla.invN,wsa.invN,wla.maps,wsa.maps,**kwargs)
        if t == 'la':
            Elm, Blm = cs.cninv.cnfilter_freq(2,mn,wla.nside,lmax,cl[1:3,:],wla.bl,wla.invN,wla.maps,**kwargs)

        pickle.dump((Elm),open(falm['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((Blm),open(falm['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def cinv(tqu,rlz,t,lmax,ntype,fmap,falm,cl,freqs=[],fmapsa='',overwrite=False,verbose=False,**kwargs):

    # prepare objects for wiener filtering
    wla = wiener_objects('la',tqu,freqs,ntype,kwargs['nsides'][0])
    wiener_objects.load_invN(wla)

    wsa = None
    if t=='co':
        wsa = wiener_objects('sa',tqu,freqs,ntype,kwargs['nsides1'][0])
        wiener_objects.load_invN(wsa)

    # roll-off effect
    roll = int(ntype[ntype.find('roll')+4:])
    if roll > 2:
        cl[:3,:roll] = 0. 

    # start loop for realizations
    for i in rlz:

        if misctools.check_path(falm['E'][i],overwrite=overwrite): continue
        if verbose: print(i)

        wiener_objects.load_maps(wla,fmap,i)
        if t=='co':
            wiener_objects.load_maps(wsa,fmapsa,i)

        cinv_core(i,t,wla,wsa,lmax,falm,cl[:4,:lmax+1],verbose=False,**kwargs)


def interface(telescope,freqs,snmin,snmax,ntype,ascale=5.,kwargs_ov={},run=['map2alm','combfreq','wiener_iso','wiener_diag']):


    if 'hitmap' in run:

        output_hitmap(**kwargs_ov)
        

    if 'simmap' in run:

        if telescope in ['la','sa']:

            for freq in freqs:
                
                # signal sim
                sobj = sim_map(telescope=telescope,freq=freq,snmin=snmin,snmax=snmax,ntype='',**kwargs_ov)
                sim_map.SOsim(sobj)
                # noise sim
                sobj = sim_map(telescope=telescope,freq=freq,snmin=snmin,snmax=snmax,ntype=ntype,**kwargs_ov)
                sim_map.SOsim(sobj)


    if 'ylm' in run:  # map -> alm for each freq (and coadd) and telescope

        if telescope == 'id': # map -> alm for fullsky case
            stype = ['o']
            ntype = 'cv'
            freqs = ['145']
        else:
            stype = ['s','n','o']

        # load survey window
        w, wn = prjlib.window(telescope,ascale=ascale) 
        
        # map -> alm for each freq
        for freq in freqs:
            p = prjlib.analysis_init(t=telescope,freq=freq,snmin=snmin,snmax=snmax,ntype=ntype) # define parameters, filenames
            map2alm(p.telescope,p.rlz,freq,p.nside,p.lmax,p.fcmb,w,roll=p.roll,**kwargs_ov) # map -> alm
            aps(p.rlz,p.lmax,p.fcmb,wn[2],stype=stype,**kwargs_ov)

        # combine alm over freqs    
        if telescope in ['la','sa']:
            p = prjlib.analysis_init(t=telescope,freq='coadd',snmin=snmin,snmax=snmax,ntype=ntype)
            fmap = prjlib.filename_freqs(freqs,t=telescope,ntype=ntype)
            alm_comb_freq(p.rlz,fmap,p.fcmb,roll=p.roll,**kwargs_ov)
            aps(p.rlz,p.lmax,p.fcmb,wn[2],**kwargs_ov)


    if 'cinv' in run:  # full wiener filtering

        if telescope == 'la':

            pw = prjlib.analysis_init(t=telescope,freq='coadd',fltr='cinv',snmin=snmin,snmax=snmax,ntype=ntype)
            wn = prjlib.wfac(telescope,binary=True)

            # filenames
            fmap = prjlib.filename_freqs(freqs,t=telescope,ntype=ntype)
 
            cinv_params = {\
                'chn' : 7, \
                'eps' : [1e-4,.1,.1,.1,.1,.1,0.], \
                'lmaxs' : [4096,2048,2048,1024,512,256,20], \
                'nsides' : [2048,2048,1024,512,256,128,64], \
                'itns' : [100,5,5,5,5,5,0], \
                'ro' : 1, \
                'filter' : 'W' \
            }
            # Temperature
            cinv(1,pw.rlz,telescope,4096,ntype,fmap,pw.fcmb.alms['o'],pw.lcl,freqs=freqs,**cinv_params,**kwargs_ov)
            # Polarization
            cinv(2,pw.rlz,telescope,4096,ntype,fmap,pw.fcmb.alms['o'],pw.lcl,freqs=freqs,**cinv_params,**kwargs_ov)

            # aps
            aps(pw.rlz,4096,pw.fcmb,wn[0],stype=['o'],mtype=['T','E','B'],**kwargs_ov)


        if telescope == 'co':
        
            p = prjlib.analysis_init(t='co',freq='coadd',fltr='cinv',snmin=snmin,snmax=snmax,ntype=ntype)

            cinv_params = {\
                'chn' : 6, \
                'eps' : [1e-6,.1,.1,.1,.1,0.], \
                'lmaxs' : [2048,1000,400,200,100,20], \
                'nsides' : [1024,512,256,512,128,64], \
                'nsides1' : [512,256,256,128,64,64], \
                'itns' : [100,9,3,3,7,0], \
                'ro' : 1, \
                'reducmn' : 2, \
                'filter' : 'W' \
            }

            fmapla = prjlib.filename_freqs(freqs,t='la',ntype=ntype)
            fmapsa = prjlib.filename_freqs(freqs,t='sa',ntype=ntype)
 
            cinv(2,p.rlz,'co',2048,ntype,fmapla,p.fcmb.alms['o'],p.lcl,freqs=freqs,fmapsa=fmapsa,ntype=ntype,**cinv_params,**kwargs_ov)
            wn = prjlib.wfac(telescope,binary=True)
            aps(p.rlz,2048,p.fcmb,wn[0],stype=['o'],mtype=['E','B'],**kwargs_ov)



    if 'iso_noise' in run:  # compute diagonal wiener-filtered alms for isotropic noise
    
        if telescope in ['la','co']:

            # for isotropic noise spectrum and diagonal wiener filtering
            pc = prjlib.analysis_init(t=telescope,freq='coadd',snmin=snmin,snmax=snmax,ntype=ntype)
            ocl = prjlib.loadocl(pc.fcmb.scl['o'],lTmin=pc.lTmin,lTmax=pc.lTmax)
            ncl = prjlib.loadocl(pc.fcmb.scl['n'],lTmin=pc.lTmin,lTmax=pc.lTmax)

            # setup filenames for input and output
            inp = prjlib.analysis_init(t='id',ntype='cv',snmin=snmin,snmax=snmax)
            out = prjlib.analysis_init(t='id',freq=t+'coadd',fltr='diag',snmin=snmin,snmax=snmax,ntype=ntype)

            # alm and aps
            wiener_diag(pc.rlz,inp.fcmb.alms['o'],out.fcmb.alms['o'],pc.roll,pc.lmax,pc.lcl[0:3,:],ocl[0:3,:],ncls=ncl[0:3,:],**kwargs_ov)
            aps(pc.rlz,pc.lmax,out.fcmb,1.,stype=['o'],**kwargs_ov)



