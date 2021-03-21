#! /usr/bin/env python3

from nexus import settings,run_project,job
from nexus import generate_physical_system
from nexus import generate_pwscf
from machine_configs import get_puhti_configs

settings(
    pseudo_dir    = './pseudopotentials',
    results       = '',
    status_only   = 0,
    generate_only = 0,
    sleep         = 3,
    machine       = 'puhti',
    account       = 'project_2000924'
    )

jobs = get_puhti_configs()

# Energy cut-offs (Ry)
ecuts = [100,150,200,250]

# kgrid
kgrid = (2,2,2)


dia16 = generate_physical_system(
    units  = 'A',
    axes   = [[ 1.785,  1.785,  0.   ],
              [ 0.   ,  1.785,  1.785],
              [ 1.785,  0.   ,  1.785]],
    elem   = ['C','C'],
    pos    = [[ 0.    ,  0.    ,  0.    ],
              [ 0.8925,  0.8925,  0.8925]],
    tiling = (2,2,2),
    kgrid  = kgrid,
    kshift = (0,0,0),
    C      = 4
    )

scf = []

# Create scf objects for energy cutoff convergence
for cutoff in ecuts:

    scf.append(generate_pwscf(
        identifier   = 'scf',
        path         = 'scf_{}'.format(cutoff),
        job          = jobs['scf'],
        input_type   = 'generic',
        calculation  = 'scf',
        input_dft    = 'lda', 
        ecutwfc      = cutoff,   
        conv_thr     = 1e-8, 
        nosym        = True,
        wf_collect   = True,
        system       = dia16,
        kgrid        = kgrid,
        pseudos      = ['C.BFD.upf'], 
        ) 
    )

# Now create scf objects for k-grid convergence, using energy cutoff of 200 Rydbergs

for i in range(4):
    grid = (i+1,i+1,i+1)

    # Create the diamond system with specified kgrid
    dia16 = generate_physical_system(
    units  = 'A',
    axes   = [[ 1.785,  1.785,  0.   ],
              [ 0.   ,  1.785,  1.785],
              [ 1.785,  0.   ,  1.785]],
    elem   = ['C','C'],
    pos    = [[ 0.    ,  0.    ,  0.    ],
              [ 0.8925,  0.8925,  0.8925]],
    tiling = (2,2,2),
    kgrid  = grid,
    kshift = (0,0,0),
    C      = 4
    )

    # Create the scf object using the diamond system
    scf.append(generate_pwscf(
        identifier   = 'scf',
        path         = 'scf_kgrid_{}'.format(i+1),
        job          = jobs['scf'],
        input_type   = 'generic',
        calculation  = 'scf',
        input_dft    = 'lda', 
        ecutwfc      = 200,   
        conv_thr     = 1e-8, 
        nosym        = True,
        wf_collect   = True,
        system       = dia16,
        kgrid        = grid,
        pseudos      = ['C.BFD.upf'], 
        ) 
    )
run_project(scf)
