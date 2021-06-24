# Linear template delensing
import numpy as np
import healpy as hp
import pickle
import tqdm
    
# from cmblensplus
import curvedsky
import misctools

# local
import prjlib
import tools_lens
import tools_multitracer


class lensing_template:

    # Define objects containing E and kappa to be combined to form a lensing template

    def __init__(self,mobj,Eobj,olmax=2048,elmin=20,elmax=2048,klmin=20,klmax=2048,nside=2048,kfltr='none'):

        conf = misctools.load_config('DELENSING')

        #//// input E and kappa for this object ////#
        # E mode
        self.etag  = Eobj.stag
        self.fElm  = Eobj.fcmb.alms['o']['E']
        #self.etype = Eobj.telescope

        # kappa
        self.klist = mobj.klist
        self.fglm  = mobj.fcklm
        self.kfltr = kfltr # this does not work now

        #//// for lensing template ////#

        # minimum/maximum multipole of E and kappa in lensing template construction
        self.elmin = conf.getint('elmin',elmin)
        self.elmax = conf.getint('elmax',elmax)
        self.klmin = conf.getint('klmin',klmin)
        self.klmax = conf.getint('klmax',klmax)

        # output template
        #self.klist = klist # list of lensing template
        self.olmax = conf.getint('olmax',olmax)
        self.l     = np.linspace(0,self.olmax,self.olmax+1)
        self.llist = ['comb']

        # //// was used for remapping, but no longer used ////#
        # Newton method iteration number for obtaining anti-deflection angle in remapping
        self.nremap = 3
        
        # remapping Nside/Npix for lensing template construction / remapping
        self.nside  = nside
        self.npix   = 12*self.nside**2


    def fname(self,qtag,doreal): # filename for output products

        # mlist is the list of mass tracers
        # self.klist is the list of lensing template. this will be only 'comb'.
        
        # set directory
        d = prjlib.data_directory()
        ids = prjlib.rlz_index(doreal=doreal)

        # filename tag for lensing template
        ltag = 'le'+str(self.elmin)+'-'+str(self.elmax)+'_lk'+str(self.klmin)+'-'+str(self.klmax)
        ttag = ltag + '_' + self.kfltr + '_' + qtag + '_' + '-'.join(self.klist.keys()) + '_' + self.etag

        # alm of lensing template B-modes
        self.falm, self.fwlm, self.cl = {}, {}, {}
        for k in self.llist:
            self.falm[k] = [d['del']+'alm/alm_'+k+'_'+ttag+'_'+x+'.pkl' for x in ids]
            self.cl[k]   = [d['del']+'aps/rlz/cl_'+k+'_'+ttag+'_'+x+'.dat' for x in ids]

        self.gtag = '_ideal'

        # correlation coeff of templates
        self.frho = d['del'] + 'aps/rho_' + '-'.join(self.llist) + '_' + ttag


    # operation to this object
    
    def template_alm(self,rlz,wlk,glmax=2008,**kwargs_ov):

        for k in tqdm.tqdm(self.klist,ncols=100,desc='template:'):
        
            for i in tqdm.tqdm(rlz,ncols=100,desc='each rlz ('+k+')',leave=False):
        
                if misctools.check_path(self.falm[k][i],**kwargs_ov): continue
            
                # load E mode
                wElm = pickle.load(open(self.fElm[i],"rb"))[:self.elmax+1,:self.elmax+1]
                wElm[:self.elmin,:] = 0.
                
                glm = 0.*wElm
                glm[:glmax,:glmax] = pickle.load(open(self.fglm[i],"rb"))
                wplm = glm * wlk[k][:self.klmax+1,None]
                wplm[:self.klmin,:] = 0.

                # construct lensing B-mode template
                dalm = curvedsky.delens.lensingb( self.olmax, self.elmin, self.elmax, self.klmin, self.klmax, wElm, wplm )

                # save to file
                pickle.dump((dalm),open(self.falm[k][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

            
    def template_aps(self,rlz,fBlm,W,**kwargs_ov):
    
        for k in tqdm.tqdm(self.klist,ncols=100,desc='template aps'):
        
            for i in tqdm.tqdm(rlz,ncols=100,desc='each rlz ('+k+')',leave=False):
        
                if misctools.check_path(self.cl[k][i],**kwargs_ov): continue
            
                dalm = pickle.load(open(self.falm[k][i],"rb"))[0:self.olmax+1,0:self.olmax+1]
                wdlm = curvedsky.utils.mulwin_spin(0*dalm,dalm,W)[1]
            
                Balm = pickle.load(open(fBlm[i],"rb"))[:self.olmax+1,:self.olmax+1]
                wBlm = curvedsky.utils.mulwin_spin(0*Balm,Balm,W)[1]
            
                clbb = curvedsky.utils.alm2cl(self.olmax,wBlm)
                cldd = curvedsky.utils.alm2cl(self.olmax,wdlm)
                clbd = curvedsky.utils.alm2cl(self.olmax,wdlm,wBlm)
                np.savetxt(self.cl[k][i],np.array((clbb,cldd,clbd)).T)



def diag_wiener(pqf,clkk,dlmin,dlmax,kL=None,Al=None,klist=['comb']): #kappa filter (including kappa->phi conversion)

    wlk = {}

    for k in tqdm.tqdm(klist,ncols=100,desc='load diag wiener filter'):

        wlk[k] = np.zeros((dlmax+1))

        if k in ['TT','TE','EE','EB']: # this will be removed in the future version

            if Al is None:
                Nl = np.loadtxt(pqf[k].al,unpack=True)[1]
            else:
                Nl = Al[k]

            for l in range(dlmin,dlmax+1):
                wlk[k][l] = clkk[l]/(clkk[l]+Nl[l])
                if kL is not None:
                    wlk[k][l] /= kL[l]

        elif k == 'comb':

            l  = np.linspace(0,dlmax,dlmax+1)
            kL = l*(l+1.)*.5
            wlk[k][dlmin:dlmax+1] = 1./kL[dlmin:dlmax+1]

        else:

            wlk[k][dlmin:dlmax+1] = 1.

    return wlk


def init_template(qtag,mobj,Eobj,doreal,**kwargs):
    # setup parameters for lensing reconstruction (see cmblensplus/utils/quad_func.py)

    dobj = lensing_template(mobj,Eobj,**kwargs)
    dobj.fname(qtag,doreal)
    
    return dobj


def interface(run_del=[],kwargs_ov={},kwargs_cmb={},kwargs_qrec={},kwargs_mass={},kwargs_del={},klist_cmb=['TT','TE','EE','EB']):

    freq  = kwargs_cmb.pop('freq')
    etype = kwargs_del.pop('etype')

    # //// prepare E modes //// #
    if etype == 'id':
        # CMB only sim
        Eobj = prjlib.analysis_init(t=etype,freq='cocom',ntype=kwargs_cmb['ntype'])

    if etype in ['co','la']:
        # with realistic noise 
        Eobj = prjlib.analysis_init(t=etype,freq='com',fltr='cinv',ntype=kwargs_cmb['ntype'].replace('_iso',''))

    # //// prepare phi //// #
    # define object
    glob = prjlib.analysis_init( freq='com', **kwargs_cmb )
    qobj = tools_lens.init_qobj( glob.stag, glob.doreal, **kwargs_qrec )
    mobj = tools_multitracer.mass_tracer( glob, qobj, **kwargs_mass )
    dobj = init_template( glob.stag+qobj.ltag, mobj, Eobj, glob.doreal, **kwargs_del )
    
    # pre-filtering for CMB phi
    wlk = diag_wiener( qobj.f, glob.kk, dobj.klmin, dobj.klmax, kL=glob.kL, klist=dobj.klist )

    # only kcinv for TT is used
    if dobj.kfltr == 'cinv':
        print('does not support kfltr = cinv')

    # fullsky isotropic noise
    #if 'iso' in glob.ntype:
    #    for k in dobj.klist:  
    #        if k in klist_cmb:
    #            qobj.f[k].mfb = None
   
    # //// compute lensing template alm //// #
    if 'alm' in run_del:
        dobj.template_alm( glob.rlz, wlk, **kwargs_ov )

    if 'aps' in run_del:
        # prepare fullsky idealistic B modes
        kwargs_cmb['t'] = 'id'
        kwargs_cmb['ntype'] = 'cv'
        pid = prjlib.analysis_init(**kwargs_cmb)
        # use overlapped region
        Wsa = prjlib.window('sa')[0]
        Wla = prjlib.window('la',ascale=5.)[0] # here, apodized mask is used otherwise the efficiency at edges gets worse
        Wsa *= hp.pixelfunc.ud_grade(Wla,512)

        # compute lensing template spectrum projected on SATxLAT area #
        dobj.template_aps( glob.rlz, pid.fcmb.alms['o']['B'], Wsa, **kwargs_ov ) # ignore E-to-B leakage


