# Set up analysis parameters, filenames, and arrays
import numpy as np
import healpy as hp
import basic
import sys
import configparser
import constants
import cmb as CMB
import quad_class

# Define parameters
# params associated with telescope
# params for analysis
class params:

    def __init__(self,t='',freq='',ntype='base'):

        #//// load config file ////#
        config = configparser.ConfigParser()
        print('reading '+sys.argv[1])
        config.read(sys.argv[1])
        conf = config['DEFAULT']

        #//// get parameters ////#
        self.lmin   = conf.getint('lmin',0)
        self.lmax   = conf.getint('lmax',3000)
        self.olmin  = conf.getint('olmin',2)
        self.olmax  = conf.getint('olmax',3000)
        self.bn     = conf.getint('bn',30) 
        self.binspc = conf.get('binspc','')
        self.snmin  = conf.getint('snmin',0)
        self.snmax  = conf.getint('snmax',1)
        self.doreal = conf.getboolean('doreal',False)
        self.telescope = conf.get('telescope','la')
        if t!='': self.telescope = t
        self.ascale = conf.getfloat('ascale',0.)

        self.freq   = conf.get('freq','145')
        if freq!='': self.freq = freq
        if self.telescope=='id': self.freq = '145'

        self.ntype = 'base'

        # reconstruction
        self.qtype  = conf.get('qtype','lens')
        self.nsidet = conf.getint('nsidet',2048)
        self.rlmin  = conf.getint('rlmin',500)
        self.rlmax  = conf.getint('rlmax',3000)
        self.snn0   = conf.getint('snn0',50)
        self.snrd   = conf.getint('snrd',100)
        self.snmf   = conf.getint('snmf',100)

        # delensing
        self.nsided  = 2048       #remapping nside for delensed map
        self.npixd   = 12*self.nsided**2
        self.dlmin   = conf.getint('dlmin',100)
        self.dlmax   = conf.getint('dlmax','2048')
        self.nremap  = 3          #beta iteration for remapping

        #//// derived parameters ////#
        # total number of real + sim
        self.snum  = self.snmax - self.snmin
        self.oL    = [self.olmin,self.olmax]
        self.dL    = [self.dlmin,self.dlmax]

        #definition of T+P
        self.qDO = [True,True,True,False,False,False]
        self.qMV = ['TT','TE','EE']


        #definition of qest
        #self.qlist = ['TT','TE','EE','TB','EB','MV']
        self.qlist = ['TT','TE','EE','TB','EB']
        if self.qtype=='rot':
            self.qlist = ['EB']

        self.dmethod = ['lt','dl','re']

        # Map resolution
        if self.telescope=='sa':
            self.nside = 512
        else:
            self.nside = 4096
        self.npix = 12*self.nside**2


# Define CMB file names
class cmb:

    def __init__(self,Dir,t,nside,freq,ascale,stag,otag,ids):

        #set directory
        d_inp = '/project/projectdirs/sobs/v4_sims/mbs/201901_gaussian_fg_lensed_cmb_realistic_noise/'
        d_alm = Dir+'cmb/alm/'
        d_aps = Dir+'cmb/aps/'
        d_map = Dir+'cmb/map/'
        d_msk = '/project/projectdirs/sobs/delensing/mask/'

        #mask
        self.mask  = d_msk+t+'.fits'
        self.amask = d_msk+t+'_a'+str(ascale)+'deg.fits'
        if ascale==0.: self.amask = self.mask

        #cmb signal map
        if t=='id':
            self.lcdm = [d_map+'/cmb_uKCMB_la'+freq+'_nside'+str(nside)+'_'+x+'.fits' for x in ids]
        else:
            self.lcdm = [d_map+'/cmb_uKCMB_'+t+freq+'_nside'+str(nside)+'_'+x+'.fits' for x in ids]

        #cmb noise map
        self.nois = [d_map+'/noise_uKCMB_'+t+freq+'_nside'+str(nside)+'_'+x+'.fits' for x in ids]

        #cmb alms
        self.salm = {}
        self.nalm = {}
        self.oalm = {}
        for m in constants.mtype:
            self.salm[m]  = [d_alm+'/s_'+m+'_'+stag+'_'+x+'.fits' for x in ids]
            self.nalm[m]  = [d_alm+'/n_'+m+'_'+stag+'_'+x+'.fits' for x in ids]
            self.oalm[m]  = [d_alm+'/o_'+m+'_'+stag+'_'+x+'.fits' for x in ids]

        #cmb aps
        self.scl = d_aps+'aps_sim_1d_'+stag+'.dat'
        self.scb = d_aps+'aps_sim_1d_'+stag+otag+'.dat'
        self.ocl = d_aps+'aps_'+ids[0]+'_1d_'+stag+'.dat'
        self.ocb = d_aps+'aps_'+ids[0]+'_1d_'+stag+otag+'.dat'


# filtered alms
class almfilt:

    def __init__(self,Dir,combtag,ids):

        d_alm = Dir+'cmb/alm/'
        self.walm = {}
        for m in constants.mtype:
            self.walm[m] = [d_alm+'/w_'+m+'_'+combtag+'_'+x+'.fits' for x in ids]


# * Define delensing filenames
class delensing:

    def __init__(self,Dir,dtag,otag,ids):

        #set directory
        dout = Dir+'delens/alm/'
        dder = Dir+'delens/derived/'

        #filename
        self.alm = [dout+'alm_'+dtag+'_'+x+'.pkl' for x in ids]
        self.wlm = [dout+'wlm_'+dtag+'_'+x+'.pkl' for x in ids]
        self.scl = dder+'scl_'+dtag+'.dat'
        self.ocl = dder+'ocl_'+dtag+'.dat'
        self.scb = dder+'scb_'+dtag+otag+'.dat'
        self.ocb = dder+'ocb_'+dtag+otag+'.dat'


# * Define class filename
class filename:

    def __init__(self,params):

        #//// root directories ////#
        Dir   = '../../data/sobs/'
        d_cls = '/project/projectdirs/sobs/sodelensing/cls/'

        #//// basic tags ////#
        # map
        stag = params.telescope+params.freq+'_'+params.ntype+'_a'+str(params.ascale)+'deg'

        # output multipole range
        otag = '_oL'+str(params.olmin)+'-'+str(params.olmax)+'_b'+str(params.bn)

        #//// index ////#
        ids = [str(i).zfill(4) for i in range(200)]
        # change 1st index
        if params.doreal: ids[0] = 'real'

        #//// CAMB cls ////#
        # aps of best fit cosmology
        self.ucl = d_cls+'ffp10_scalCls.dat'
        self.lcl = d_cls+'ffp10_lensedCls.dat'

        #//// Filenames ////#
        # input phi
        self.palm = ['/global/cscratch1/sd/engelen/simsS1516_v0.4/data/fullskyPhi_alm_0'+x+'.fits' for x in ids]

        # cmb map, window, alm and aps
        self.cmb = cmb(Dir,params.telescope,params.nside,params.freq,params.ascale,stag,otag,ids)

        # filtered combined map
        self.filt = {}
        for t in ['la','sa','co']:
            self.filt[t] = almfilt(Dir,t+'coadd',ids)

        # lensing reconstruction
        ltag = '_l'+str(params.rlmin)+'-'+str(params.rlmax)
        self.quad = {}
        for q in params.qlist:
            self.quad[q] = quad_class.quad_fname(params.qtype,q,Dir,stag+ltag,otag,ids)

        # mass tracer
        self.galm = ['/global/project/projectdirs/sobs/delensing/multitracer_forBBgroup/combined_phi_alms_noiselessE_mvkappa_simid_'+str(i)+'.npy' for i in range(200)]

        # delensing
        dtag = '_dl'+str(params.dlmin)+'-'+str(params.dlmax)
        self.delens = {}
        for dm in ['ideal','simple','samemask','wiener']:
            for den in ['la','co']:
                self.delens[dm,den] = delensing(Dir,dm+'_E'+den+'_'+dtag,otag,ids)


# Define arrays and functions for analysis
class recfunc:

    def __init__(self,params,filename):

        #multipole
        self.eL = np.linspace(0,params.lmax,params.lmax+1)
        self.oL = np.linspace(0,params.olmax,params.olmax+1)
        self.dL = np.linspace(0,params.dlmax,params.dlmax+1)
        self.kL = self.eL*(self.eL+1)*.5

        #binned multipole
        self.bp, self.bc = basic.aps.binning(params.bn,params.oL,params.binspc)

        #theoretical cl
        self.ucl = basic.aps.read_cambcls(filename.ucl,params.lmin,params.lmax,5)/constants.Tcmb**2
        self.lcl = basic.aps.read_cambcls(filename.lcl,params.lmin,params.lmax,4,bb=True)/constants.Tcmb**2


#initial setup
def analysis_init(t='',freq='',ntype=''):
    p = params(t=t,freq=freq,ntype=ntype)
    f = filename(p)
    r = recfunc(p,f)
    return p, f, r


def filename_init(t='',freq='',ntype=''):
    p = params(t=t,freq=freq,ntype=ntype)
    f = filename(p)
    return f


def get_beam(t,freq,lmax):
    # beam
    if t == 'la' or t == 'id':
        if freq == '93':   theta = 2.2
        if freq == '145':  theta = 1.4
        if freq == '225':  theta = 1.0
        if freq == '280':  theta = 0.9
    if t == 'sa':
        if freq == '93':   theta = 30.
        if freq == '145':  theta = 17.
        if freq == '225':  theta = 11.
        if freq == '280':  theta = 9.0

    return CMB.beam(theta,lmax)


def window(fmask,t=''):
    if t=='id':
        w, w2, w4 = 1., 1., 1.
    else:
        w  = hp.fitsfunc.read_map(fmask)
    w2 = np.average(w**2)
    w4 = np.average(w**4)
    print(w2,w4)
    return w, w2, w4


def make_qrec_filter(p,f,r):
    '''
    return C-inverse filter for reconstruction, multipled to CMB alms
    '''

    if p.telescope=='id':
        obs = r.lcl[:4,:]
        sig = r.lcl[:4,:]
    else:
        obs = np.loadtxt(f.cmb.scl,unpack=True,usecols=(1,2,3,4))
        sig = np.loadtxt(f.cmb.scl,unpack=True,usecols=(5,6,7,8))

    r.oc = np.zeros((4,p.lmax+1))
    r.fl = {}
    r.Fl = {}
    for m in constants.mtype:
        r.fl[m] = np.zeros(p.lmax+1)
        r.Fl[m] = np.zeros((p.lmax+1,p.lmax+1))

    for l in range(p.rlmin,p.rlmax+1):
        for mi, m in enumerate(constants.mtype):
            r.fl[m][l] = np.sqrt(sig[mi,l]/r.lcl[mi,l])
            r.Fl[m][l,0:l+1] = r.fl[m][l,None]/obs[mi,l,None]
            r.oc[mi,l] = obs[mi,l]/r.fl[m][l]**2
            r.oc[3,l] = obs[3,l]/(r.fl['T'][l]*r.fl['E'][l])

    del obs, sig


def Wiener_Emodes(t,lmin,lmax,fcmbscl):
    '''
    Return Wiener E-mode filter
    '''

    if t=='id':
        wle = np.ones((lmax+1,lmax+1))
    else:
        obs = np.loadtxt(fcmbscl,unpack=True,usecols=(1,2,3,4))
        sig = np.loadtxt(fcmbscl,unpack=True,usecols=(5,6,7,8))
        #E-filter
        wle = np.zeros((lmax+1,lmax+1))
        for l in range(lmin,lmax+1):
            wle[l,0:l+1] = sig[1,l,None]/obs[1,l,None]
        #Noise EE
        NE = np.loadtxt(fcmbscl,unpack=True)[10]
    
    return wle, NE


def getwElm(lmin,lmax,t='la',freq='145'):
    f = filename_init(t=t,freq=freq)
    return Wiener_Emodes(t,lmin,lmax,f.cmb.scl)


def Wiener_Lensing(p,f,r):
    #kappa filter (including kappa->phi conversion)
    r.wlk = {}
    for q in p.qlist:
        Al = np.loadtxt(f.quad[q].al,unpack=True)[1]
        r.wlk[q] = np.zeros((p.lmax+1,p.lmax+1))
        for l in range(p.dlmin,p.dlmax+1):
            r.wlk[q][l,0:l] = r.ucl[3,l]*r.kL[l]**2/(r.ucl[3,l]*r.kL[l]**2+Al[l])/r.kL[l]

