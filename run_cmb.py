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
run_cmb = ['calcalm']


kwargs_ov   = {\
    'overwrite':False, \
    'verbose':True \
}

kwargs_cmb  = {\
    'snmin':1, \
    'snmax':100, \
    #freq: not use for simmap
    'freq':'com', \
    #'t':'co', \
    't':'la', \
    #'t':'sa', \
    'ntype':'base_roll50', \
    #'ntype':'base_iso_roll50', \
    #'ntype':'goal_roll50', \
    #'ntype':'goal_iso_roll50', \
    'lTmin':500, \
    'lTmax':3000, \
    #'fltr':'none', \
    'fltr':'cinv', \
    'ascale':5.0, \
}


freqs = ['93','145','225']
#freqs = ['145','225']
#freqs = ['225']

# //// Main calculation ////#
tools_cmb.interface( freqs, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, run=run_cmb )

