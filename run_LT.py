# Running delenisng code

# from external module
import numpy as np

# from this directory
import prjlib
import tools_cmb
import tools_lens
import tools_multitracer
import tools_delens


#run_qrec = ['norm','qrec','n0','mean','aps']
#run_qrec = ['norm','qrec','mean']
#run_qrec = ['n0','mean','aps']
#run_qrec = ['norm','qrec','n0','rdn0','mean','aps','kcinv']
#run_qrec = ['mean','aps']
run_qrec = []

#run_mass = ['gen_alm','comb']
run_mass = ['comb']
#run_mass = []

run_del = ['alm','aps']
#run_del = ['alm','aps','rho']
#run_del = ['rho']
#run_del = []


kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}


kwargs_cmb  = {\
    'snmin':1, \
    'snmax':100, \
    # fixed for LT
    'freq':'com', \
    # fixed for LT
    't':'la', \
    #'ntype':'base_roll50', \
    #'ntype':'base_iso_roll50', \
    'ntype':'goal_roll50', \
    #'ntype':'goal_iso_roll50', \
    'lTmin':500, \
    'lTmax':3000, \
    'fltr':'none', \
    #'fltr':'cinv', \
    'ascale':5.0, \
}


kwargs_qrec = {\
    'qlist':['TT','TE','EE','EB'], \
    'rlmin':300, \
    'rlmax':4096, \
    'nside':2048, \
    'n0min':1, \
    'n0max':int(kwargs_cmb['snmax']/2), \
    'mfmin':1, \
    'mfmax':kwargs_cmb['snmax'], \
    'rdmin':1, \
    'rdmax':kwargs_cmb['snmax'] \
}


kwargs_mass = {\
    #//// mass tracers to be combined before lensing template construction ////#
    #//// cmb tracers ////#
    #'add_cmb':['TT','TE','EE','EB'], \
    'add_cmb':[], \
    #//// galaxy/CIB tracers ////#
    #'add_gal':np.arange(6), \
    #'add_gal':[], \
    #'add_cib':True, \
    'add_cib':False, \
}


kwargs_del = {\
    #//// E-mode type for lensing template (co or la) ////#
    'etype':'co', \
    #'etype':'la', \
    #//// minimum/maximum multipole of E modes ////#
    'elmin':50, \
    'elmax':2048, \
    #//// minimum/maximum multipole of mass tracers ////#
    'klmin':20, \
    'klmax':2048, \
    #'klist':['TT','TE','EE','EB'],\ #combining after making individual templates not recommended in the future
    #//// kappa cinv filter (this does not work now) ////#
    'kfltr':'none', \
    #//// list of lensing template (should be comb) ////#
    'klist':['comb'], \
    #//// output template maximum multipole ////#
    'olmax':2048, \
}


# //// Main calculation ////#
if run_qrec:
    tools_lens.interface( run=run_qrec, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, kwargs_qrec=kwargs_qrec )

if run_mass:
    tools_multitracer.interface( run=run_mass, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, kwargs_qrec=kwargs_qrec, kwargs_mass=kwargs_mass )

if run_del:
    tools_delens.interface( run_del=run_del, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, kwargs_qrec=kwargs_qrec, kwargs_mass=kwargs_mass, kwargs_del=kwargs_del )


