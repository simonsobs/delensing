import numpy as np, os
from quicksub import *

def create_runfile(f,r0,r1,t='co',ntype='goal_roll50'):
    
    add('import numpy as np, prjlib, tools_cmb',f,ini=True)
    add("kwargs_ov = {'overwrite':True,'verbose':True}",f)
    #add("kwargs_ov = {'overwrite':False,'verbose':True}",f)
    add("kwargs_cmb = {'snmin':"+str(r0)+",'snmax':"+str(r1)+",'t':'"+t+"','ntype':'"+ntype+"','ascale':5.0,'lTmin':500,'lTmax':3000,'fltr':'cinv','freq':'com'}",f)
    add("tools_cmb.interface( ['93','145','225'], kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, run=['calcalm'] )",f)


def jobfile(tag,r0,r1,**kwargs):

    #set run file
    f_run = 'tmp_job_run_'+tag+'.py'
    create_runfile(f_run,r0,r1,**kwargs)

    # set job file
    f_sub = 'tmp_job_sub_'+tag+'.sh'
    set_sbatch_params(f_sub,tag,mem='32G',t='1-00:00',email=True)
    add('source ~/.bashrc.ext',f_sub)
    add('py4so',f_sub)
    add('python '+f_run,f_sub)
    
    # submit
    #os.system('sbatch '+f_sub)
    os.system('sh '+f_sub)
    os.system('rm -rf '+f_run+' '+f_sub)


#rlz0 = np.arange(1,110,10)
#rlz1 = rlz0[1:]-1
#rlz1[-1] = 100

#for r0, r1 in zip(rlz0,rlz1):
#    print(r0,r1)
    #jobfile('so_cinv_la_rlz_'+str(r0)+'-'+str(r1),r0,r1,t='la',ntype='goal_roll50')
    #jobfile('so_cinv_hires_rlz_'+str(r0)+'-'+str(r1),r0,r1,t='co',ntype='base_roll50')
    #jobfile('so_cinv_hires_goal_rlz_'+str(r0)+'-'+str(r1),r0,r1,t='co',ntype='goal_roll50')

jobfile('so_cinv_la_rlz_'+str(1)+'-'+str(1),1,1,t='la',ntype='base_roll50')

