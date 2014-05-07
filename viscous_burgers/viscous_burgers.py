#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Viscous Burgers demo.

Usage:
  viscous_burgers.py [-hp] [--grid=NI] [--initial-data=TYPE] [--lxf-lambda=VALUE] [--nt=COUNT]
             [--not-periodic] [--num-flux=FLUX] [--vx=XSPEED] [--vy=YSPEED] EXP DIFF


Arguments:
  EXP                    Exponent
  
  DIFF                   Diffusion


Options:
  --grid=NI              Use grid with (2*NI)*NI elements [default: 60].

  --initial-data=TYPE    Select the initial data (sin, bump) [default: sin]

  --lxf-lambda=VALUE     Parameter lambda in Lax-Friedrichs flux [default: 1].

  --nt=COUNT             Number of time steps [default: 100].

  --not-periodic         Solve with dirichlet boundary conditions on left
                         and bottom boundary.

  --num-flux=FLUX        Numerical flux to use (lax_friedrichs, engquist_osher)
                         [default: lax_friedrichs].

  -h, --help             Show this message.

  --vx=XSPEED            Speed in x-direction [default: 1].

  --vy=YSPEED            Speed in y-direction [default: 1].
'''

from __future__ import absolute_import, division, print_function

import sys
import math as m
import time
from functools import partial
import cProfile, pstats, io

import numpy as np
from docopt import docopt

import pymor.core as core
core.logger.MAX_HIERACHY_LEVEL = 2
from analyticalproblems.viscous_burgers import ViscousBurgersProblem
from discretizers.advection_diffusion import discretize_nonlinear_instationary_advection_diffusion_fv
from pymor.domaindiscretizers import discretize_domain_default
from pymor.grids import RectGrid


core.getLogger('pymor.algorithms').setLevel('INFO')
core.getLogger('pymor.discretizations').setLevel('INFO')


def viscous_burgers_demo(args):
    args['--grid'] = int(args['--grid'])
    args['--initial-data'] = args['--initial-data'].lower()
    assert args['--initial-data'] in ('sin', 'bump')
    args['--lxf-lambda'] = float(args['--lxf-lambda'])
    args['--nt'] = int(args['--nt'])
    args['--not-periodic'] = bool(args['--not-periodic'])
    args['--num-flux'] = args['--num-flux'].lower()
    assert args['--num-flux'] in ('lax_friedrichs', 'engquist_osher')
    args['--vx'] = float(args['--vx'])
    args['--vy'] = float(args['--vy'])
    args['EXP'] = float(args['EXP'])
    args['DIFF'] = float(args['DIFF'])


    print('Setup Problem ...')
    domain_discretizer = partial(discretize_domain_default, grid_type= RectGrid)
    problem = ViscousBurgersProblem(vx=args['--vx'], vy=args['--vy'], initial_data=args['--initial-data'],
                             parameter_range=(0, 1e42), torus=not args['--not-periodic'])

    print('Discretize ...')
    discretizer = discretize_nonlinear_instationary_advection_diffusion_fv
    discretization, _ = discretizer(problem, diameter=m.sqrt(2) / args['--grid'],
                                    num_flux=args['--num-flux'], lxf_lambda=args['--lxf-lambda'],
                                    nt=args['--nt'], domain_discretizer=domain_discretizer)
    
    print(discretization.explicit_operator.grid)
    #print(discretization.implicit_operator.grid)
 
    print('The parameter type is {}'.format(discretization.parameter_type))
 
    mu = {'exponent': args['EXP'], 'diffusion': [args['DIFF']]}
    # U = discretization.solve(0)
    print('Solving for exponent = {} ... '.format(mu))
    sys.stdout.flush()
    # pr = cProfile.Profile()
    # pr.enable()
    tic = time.time()
    U = discretization.solve(mu)
    # pr.disable()
    print('Solving took {}s'.format(time.time() - tic))
    # pr.dump_stats('bla')
    discretization.visualize(U)
    
    
    # plot with matplotlib on three timesteps: t = 0, t = nt/2, t = nt
    import pylab as plt
    
    U1 = U.copy(0)
    U2 = U.copy(len(U)/2)
    U3 = U.copy(len(U)-1)
    
    U1 = np.flipud(U1._array.reshape(args['--grid'], args['--grid']*2))
    U2 = np.flipud(U2._array.reshape(args['--grid'], args['--grid']*2))
    U3 = np.flipud(U3._array.reshape(args['--grid'], args['--grid']*2))
    
    data = list([U1, U2, U3])
    
    fig, axes = plt.subplots(nrows=1, ncols=3)
    
    for u,ax in zip(data,axes.flat):
        im = ax.imshow(u, cmap='jet', extent=[0, 2, 0, 1], vmin=0, vmax=1)
    
    cax = fig.add_axes([0.1, 0.3, 0.8, 0.03])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.show()

if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    viscous_burgers_demo(args)
