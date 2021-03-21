#! /usr/bin/env python3
import os
from nexus import settings,job,run_project,obj
from nexus import generate_physical_system
from nexus import generate_pwscf
from machine_configs import get_puhti_configs
from numpy import *

settings(
    pseudo_dir    = './pseudopotentials',
    results       = '',
    status_only   = 0,
    generate_only = 0, 
    sleep         = 3,
    machine       = 'puhti', # define the machine (as described in nexus machines.py)
    account       = 'project_2000924', # project / account that the job is charged from
)

# get the Puhti configurations for running the job
jobs = get_puhti_configs()

cubic_box_size=[10.0]
x=1.0*cubic_box_size[0]
d_eq=1.2074 # nuclei separation in Angstrom

# Points around d_eq
ds = linspace(d_eq-0.15,d_eq+0.15,5)


# Create list of 'scf' objects to enable single workflow
scf = []
for i in range(5):

    # Generate the system for each value around d_eq
    O2 = generate_physical_system(
        units  = 'A', # Angstrom units
        axes   = [[ x,   0.0 ,  0.0   ],
                [ 0.,   x  ,  0.0   ],
                [ 0.,   0. ,   x    ]],
        elem   = ['O','O'],
        pos    = [[ x/2-ds[i]/2    ,  x/2    ,  x/2    ],
                [ x/2+ds[i]/2    ,  x/2    ,  x/2    ]],
        net_spin  = 2,
        tiling    = (1,1,1),
        kgrid     = (1,1,1), # scf kgrid given below to enable symmetries
        kshift    = (0,0,0),
        O         = 6,
    )

    # Add the scf object to the list
    scf.append( generate_pwscf(
        identifier   = 'scf',
        path         = 'scf_{}'.format(ds[i]),
        job          = jobs['scf'],
        input_type   = 'generic',
        system       = O2,
        calculation  = 'scf',
        input_dft    = 'lda', 
        ecutwfc      = 200,   
        conv_thr     = 1e-8, 
        nosym        = False,
        wf_collect   = True,
        nspin        = 2,
        tot_magnetization = 2,
        electron_maxstep = 300,
        kgrid        = (1,1,1),
        pseudos      = ['O.BFD.upf'], 
        )
    )

# run the workflow
run_project(scf)
