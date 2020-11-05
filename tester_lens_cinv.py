#!/usr/bin/env python
# coding: utf-8

import numpy as np, prjlib, pickle, tools_lens

pobj = prjlib.analysis_init(t='la',freq='com',fltr='none',snmin=1,snmax=1,ntype='base_roll50')
#qobj = tools_lens.init_qobj(pobj.stag,False,rlmin=300,rlmax=4096,qlist=['TT','TE','EE','EB'])
qobj = tools_lens.init_qobj(pobj.stag,False,rlmin=300,rlmax=4096,qlist=['TT'])

#for q in qobj.qlist:  tools_lens.compute_knoise(pobj.rlz,qobj.f[q],2048)

tools_lens.compute_kcninv(qobj,pobj.rlz,pobj.kk,nside=2048,klmin=10,qlist=qobj.qlist)

