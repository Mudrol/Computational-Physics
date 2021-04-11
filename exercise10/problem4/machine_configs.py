"""
    FYS-4096 Computational Physics, Ex 10, problem 4
    modified machine_configs.py file
"""

#! /usr/bin/env python3
from nexus import job

def general_configs(machine):
    if machine=='puhti':
        jobs = get_puhti_configs()
    else:
        print('Using Puhti as defaul machine')
        jobs = get_puhti_configs()
    return jobs

def get_puhti_configs():
    # remember to load the modules used in compiling the code
    # these are needed in running the code
    scf_presub = '''
module load intel/19.0.4
module load hpcx-mpi/2.4.0
module load intel-mkl/2019.0.4
module load StdEnv
module load hdf5/1.10.4
    '''
    # what modules are needed for qmcpack
    qmc_presub = '''
    '''

    # application that performs the calculations
    qe_app='pw.x'
    conv_app='pw2qmcpack.x'
    qmc_app='qmcpack'

    # csc queue
    # https://docs.csc.fi/computing/running/batch-job-partitions/
    csc_queue = 'small' # test, small, large, ...

    # define jobs
    # 4 processes for scf, 1 for conv, 20 for vmc, 20 for optim, 20 for dmc
    scf  = job(cores=4,minutes=10,user_env=False,presub=scf_presub,app=qe_app,queue=csc_queue)
    nscf  = job(cores=4,minutes=10,user_env=False,presub=scf_presub,app=qe_app,queue=csc_queue)
    p2q = job(cores=1,minutes=10,user_env=False,presub=scf_presub,app=conv_app,queue=csc_queue)
    opt  = job(cores=20,minutes=10,user_env=False,presub=qmc_presub,app=qmc_app,queue=csc_queue)
    qmc  = job(cores=20,minutes=20,user_env=False,presub=qmc_presub,app=qmc_app,queue=csc_queue)

    # 40 processes (1 node = 40 processors at Puhti)
    #scf  = job(nodes=1,hours=1,user_env=False,presub=scf_presub,app=qe,queue=csc_queue)
    
    jobs = {'scf' : scf, 'nscf': nscf, 'p2q': p2q, 'opt': opt, 'qmc': qmc}

    return jobs
