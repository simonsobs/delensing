# Running delenisng code

# from external module
import numpy as np

# from this directory
import prjlib
import tools_cmb
import tools_lens
import tools_multitracer
import tools_delens


#run_cmb = ['simmap']
#run_cmb = ['simmap','calcalm']
#run_cmb = ['calcalm']
run_cmb = []

#run_qrec = ['norm','qrec','n0','mean','aps']
#run_qrec = ['n0','mean','aps']
#run_qrec = ['norm','qrec','n0','rdn0','mean','aps','kcinv']
#run_qrec = ['kcinv']
run_qrec = []

run_mass = ['gen_alm','comb']
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
    #freq: not use for simmap
    'freq':'com', \
    #'t':'co', \
    't':'la', \
    #'ntype':'base_roll50', \
    'ntype':'base_iso_roll50', \
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
    'lmin':8,\
    'lmax':2007,\
    'add_cmb':['TT','TE','EE','EB'],\
}


kwargs_del = {\
    'etype':'co',\
    #'klist':['TT','TE','EE','EB'],\
    'klist':['comb'],\
    #'kfltr':'cinv',\
    'kfltr':'none',\
    'olmax':2048,\
    'elmin':50\
}


freqs = ['93','145','225']
#freqs = ['145','225']


# //// Main calculation ////#
if run_cmb:
    tools_cmb.interface( freqs, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, run=run_cmb )

if run_qrec:
    tools_lens.interface( run=run_qrec, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, kwargs_qrec=kwargs_qrec )

if run_mass:
    tools_multitracer.interface( run=run_mass, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, kwargs_qrec=kwargs_qrec, kwargs_mass=kwargs_mass )

if run_del:
    tools_delens.interface( run_del=run_del, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, kwargs_qrec=kwargs_qrec, kwargs_mass=kwargs_mass, kwargs_del=kwargs_del )


