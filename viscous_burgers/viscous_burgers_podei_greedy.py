#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Viscous Burges with PODEI-Greedy demo.

Usage:
  viscous_burgers_podei_greedy.py [options] EXP_MIN EXP_MAX DIFF_MIN DIFF_MAX RBSIZE EXP_SNAPSHOTS DIFF_SNAPSHOTS INIT_EISIZE EXP_INIT_SNAPSHOTS DIFF_INIT_SNAPSHOTS


Arguments:
  EXP_MIN       Minimal exponent

  EXP_MAX       Maximal exponent
  
  DIFF_MIN      Minimal Diffusion
  
  DIFF_MAX      Maximal Diffusion.

  RBSIZE        Size of the reduced basis

  EXP_SNAPSHOTS     Number of snapshots for the exponent in basis generation and EI.
  
  DIFF_SNAPSHOTS     Number of snapshots for the exponent in basis generation and EI.
  
  INIT_EISIZE     Number of inital interpolation DOFs.
  
  EXP_INIT_SNAPSHOTS     Number of snapshots for the exponent in initial EI.
  
  DIFF_INIT_SNAPSHOTS    Number of snapshots for the diffusion in initial EI.


Options:
  --grid=NI              Use grid with (2*NI)*NI elements [default: 60].

  --initial-data=TYPE    Select the initial data (sin, bump) [default: sin]

  --lxf-lambda=VALUE     Parameter lambda in Lax-Friedrichs flux [default: 1].

  --not-periodic         Solve with dirichlet boundary conditions on left
                         and bottom boundary.

  --nt=COUNT             Number of time steps [default: 100].

  --num-flux=FLUX        Numerical flux to use (lax_friedrichs, engquist_osher)
                         [default: lax_friedrichs].

  -h, --help             Show this message.

  -p, --plot-err         Plot error.

  --plot-ei-err          Plot empirical interpolation error.

  --plot-solutions       Plot some example solutions.
  
  --plot-error-landscape          Calculate and show plot of reduction error vs. basis sizes.
  
  --plot-error-landscape-N=COUNT  Number of basis sizes to test [default: 10]

  --plot-error-landscape-M=COUNT  Number of collateral basis sizes to test [default: 10]
  
  --plot-error-correlation

  --test=COUNT           Use COUNT snapshots for stochastic error estimation
                         [default: 10].

  --vx=XSPEED            Speed in x-direction [default: 1].

  --vy=YSPEED            Speed in y-direction [default: 1].
'''

from __future__ import absolute_import, division, print_function

import sys
import math as m
import time
from functools import partial

import numpy as np
from docopt import docopt

from analyticalproblems.viscous_burgers import ViscousBurgersProblem
from discretizers.advection_diffusion import discretize_nonlinear_instationary_advection_diffusion_fv
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.rect import RectGrid
from algorithms.ei_greedy import ei_rb_greedy
from pymor.reductors.basic import reduce_to_subbasis
from pymor.algorithms.basisextension import pod_basis_extension
from pymor.algorithms.ei import interpolate_operators
from pymor.vectorarrays.numpy import NumpyVectorArray


def burgers_demo(args):
    args['--grid'] = int(args['--grid'])
    args['--initial-data'] = args['--initial-data'].lower()
    assert args['--initial-data'] in ('sin', 'bump')
    args['--lxf-lambda'] = float(args['--lxf-lambda'])
    args['--nt'] = int(args['--nt'])
    args['--not-periodic'] = bool(args['--not-periodic'])
    args['--num-flux'] = args['--num-flux'].lower()
    assert args['--num-flux'] in ('lax_friedrichs', 'engquist_osher')
    args['--plot-error-landscape-N'] = int(args['--plot-error-landscape-N'])
    args['--plot-error-landscape-M'] = int(args['--plot-error-landscape-M'])
    args['--test'] = int(args['--test'])
    args['--vx'] = float(args['--vx'])
    args['--vy'] = float(args['--vy'])
    args['EXP_MIN'] = int(args['EXP_MIN'])
    args['EXP_MAX'] = int(args['EXP_MAX'])
    args['DIFF_MIN'] = float(args['DIFF_MIN'])
    args['DIFF_MAX'] = float(args['DIFF_MAX'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['EXP_SNAPSHOTS'] = int(args['EXP_SNAPSHOTS'])
    args['DIFF_SNAPSHOTS'] = int(args['DIFF_SNAPSHOTS'])
    args['INIT_EISIZE'] = int(args['INIT_EISIZE'])
    args['EXP_INIT_SNAPSHOTS'] = int(args['EXP_INIT_SNAPSHOTS'])
    args['DIFF_INIT_SNAPSHOTS'] = int(args['DIFF_INIT_SNAPSHOTS'])

    print('Setup Problem ...')
    domain_discretizer = partial(discretize_domain_default, grid_type=RectGrid)
    problem = ViscousBurgersProblem(vx=args['--vx'], vy=args['--vy'], initial_data=args['--initial-data'],
                             parameter_range={'exponent': (args['EXP_MIN'], args['EXP_MAX']), 'diffusion': (args['DIFF_MIN'], args['DIFF_MAX'])}, 
                             torus=not args['--not-periodic'])

    print('Discretize ...')
    discretizer = discretize_nonlinear_instationary_advection_diffusion_fv
    discretization, _ = discretizer(problem, diameter=m.sqrt(2) / args['--grid'],
                                    num_flux=args['--num-flux'], lxf_lambda=args['--lxf-lambda'],
                                    nt=args['--nt'], domain_discretizer=domain_discretizer)

    print(discretization.explicit_operator.grid)

    print('The parameter type is {}'.format(discretization.parameter_type))

    if args['--plot-solutions']:
        print('Showing some solutions')
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for exponent = \n{} ... '.format(mu['exponent']))
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            U = discretization.solve(mu)
            discretization.visualize(U)

    print('CB an RB generation ...')
 
    extension_algorithm = partial(pod_basis_extension)
    
    if args['EXP_MIN'] == args['EXP_MAX']:
        init_samples = discretization.parameter_space.sample_uniformly({'exponent': 1, 'diffusion': args['DIFF_INIT_SNAPSHOTS']})
        samples = discretization.parameter_space.sample_uniformly({'exponent': 1, 'diffusion': args['DIFF_SNAPSHOTS']})
    elif args['DIFF_MIN'] == args['DIFF_MAX']:
        init_samples = discretization.parameter_space.sample_uniformly({'exponent': args['EXP_INIT_SNAPSHOTS'], 'diffusion': 1})
        samples = discretization.parameter_space.sample_uniformly({'exponent': args['EXP_SNAPSHOTS'], 'diffusion': 1})
    else:
        init_samples = discretization.parameter_space.sample_uniformly({'exponent': args['EXP_INIT_SNAPSHOTS'], 'diffusion': args['DIFF_INIT_SNAPSHOTS']})
        samples = discretization.parameter_space.sample_uniformly({'exponent': args['EXP_SNAPSHOTS'], 'diffusion': args['DIFF_SNAPSHOTS']})
    
    _, ei_initial_data = interpolate_operators(discretization, ['explicit_operator'],
                                            init_samples,
                                            error_norm=discretization.l2_norm, target_error=1e-10,
                                            max_interpolation_dofs=args['INIT_EISIZE'], projection='orthogonal',
                                            product=discretization.l2_product)
    
    rb_initial_data = discretization.initial_data.as_vector()
    rb_initial_data *= 1 / rb_initial_data.l2_norm()[0]

    ei_greedy_data = ei_rb_greedy(discretization, 
                               operator_names=['explicit_operator'], samples=samples,
                               error_norm=discretization.l2_norm,
                               target_error=1e-10, rb_initial_data=rb_initial_data, ei_initial_data=ei_initial_data,
                               max_extensions=args['RBSIZE'], use_estimator=False,
                               extension_algorithm=extension_algorithm)

    rb_discretization, reconstructor = ei_greedy_data['reduced_discretization'], ei_greedy_data['reconstructor']
    ei_discretization, ei_data = ei_greedy_data['ei_discretization'], ei_greedy_data['ei_data']
    
    if args['--plot-ei-err']:
        print('Showing some EI errors')
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for exponent = \n{} ... '.format(mu['exponent']))
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            U = discretization.solve(mu)
            U_EI = ei_discretization.solve(mu)
            ERR = U - U_EI
            print('Error: {}'.format(np.max(discretization.l2_norm(ERR))))
            discretization.visualize(ERR)
 
        print('Showing interpolation DOFs ...')
        U = np.zeros(U.dim)
        dofs = ei_discretization.operator.interpolation_dofs
        U[dofs] = np.arange(1, len(dofs) + 1)
        U[ei_discretization.operator.source_dofs] += int(len(dofs)/2)
        discretization.visualize(NumpyVectorArray(U))

    print('\nSearching for maximum error on random snapshots ...')

    tic = time.time()
    
    mus = list(discretization.parameter_space.sample_randomly(args['--test']))
    #mus.append({'diffusion': np.array([0.0]), 'exponent': np.array(2.0)})

    def error_analysis(N, M):
        print('N = {}, M = {}: '.format(N, M), end='')
        rd, rc, _ = reduce_to_subbasis(rb_discretization, N, reconstructor)
        rd = rd.with_(explicit_operator=rd.explicit_operator.projected_to_subbasis(dim_collateral=M))
        l2_err_max = -1
        mumax = None
        for mu in mus:
            print('.', end='')
            sys.stdout.flush()
            u = rd.solve(mu)
            URB = rc.reconstruct(u)
            U = discretization.solve(mu)
            l2_err = np.max(discretization.l2_norm(U - URB))
            l2_err = np.inf if not np.isfinite(l2_err) else l2_err
            if l2_err > l2_err_max:
                l2_err_max = l2_err
                mumax = mu
        print()
        return l2_err_max, mumax
    error_analysis = np.frompyfunc(error_analysis, 2, 2)
    
    real_rb_size = len(ei_greedy_data['data'])
    real_cb_size = len(ei_data['basis'])
    
    if args['--plot-error-landscape']:
        N_count = min(real_rb_size - 1, args['--plot-error-landscape-N'])
        M_count = min(real_cb_size - 1, args['--plot-error-landscape-M'])
        Ns = np.linspace(1, real_rb_size, N_count).astype(np.int)
        Ms = np.linspace(1, real_cb_size, M_count).astype(np.int)
    else:
        Ns = np.array([real_rb_size])
        Ms = np.array([real_cb_size])

    N_grid, M_grid = np.meshgrid(Ns, Ms)

    errs, err_mus = error_analysis(N_grid, M_grid)
    errs = errs.astype(np.float)

    l2_err_max = errs[-1, -1]
    mumax = err_mus[-1, -1]
    toc = time.time()
    t_est = toc - tic

    print('''
    *** RESULTS ***

    Problem:
       parameter range:                    ({args[EXP_MIN]}, {args[EXP_MAX]}), ({args[DIFF_MIN]}, {args[DIFF_MAX]})
       h:                                  sqrt(2)/{args[--grid]}
       initial-data:                       {args[--initial-data]}
       lxf-lambda:                         {args[--lxf-lambda]}
       nt:                                 {args[--nt]}
       not-periodic:                       {args[--not-periodic]}
       num-flux:                           {args[--num-flux]}
       (vx, vy):                           ({args[--vx]}, {args[--vy]})

    Greedy basis generation:
       number of initial snapshots:        ({args[EXP_INIT_SNAPSHOTS]}, {args[DIFF_INIT_SNAPSHOTS]})
       initial collateral basis size:      {args[INIT_EISIZE]}
       actual collateral basis size:       {real_cb_size}
       number of snapshots:                ({args[EXP_SNAPSHOTS]}, {args[DIFF_SNAPSHOTS]})
       prescribed basis size:              {args[RBSIZE]}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {ei_greedy_data[time]}

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal L2-error:                   {l2_err_max}  (mu = {mumax})
       
       maximum errors:                     {errs}
       N_grid (RB size):                   {N_grid}
       M_grid (CB size):                   {M_grid}
       
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()
    if args['--plot-error-landscape']:
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # we have to rescale the errors since matplotlib does not support logarithmic scales on 3d plots
        # https://github.com/matplotlib/matplotlib/issues/209
        surf = ax.plot_surface(M_grid, N_grid, np.log(np.minimum(errs, 1)) / np.log(10),
                               rstride=1, cstride=1, cmap='jet')
        plt.show()
        
    if args['--plot-error-correlation']:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        errs = []
        Ns = ei_greedy_data['N_M_correlation'][0]
        Ms = ei_greedy_data['N_M_correlation'][1]
        corr = []
        for i in np.arange(real_rb_size):
            N = Ns[i]
            M = Ms[i]
            if M > real_cb_size:
                M = real_cb_size
            corr.append(M/N)
            print('N = {}, M = {}: '.format(N, M), end='')
            rd, rc, _ = reduce_to_subbasis(rb_discretization, N, reconstructor)
            rd = rd.with_(explicit_operator=rd.explicit_operator.projected_to_subbasis(dim_collateral=M))
            l2_err_max = -1
            mumax = None
            for mu in mus:
                print('.', end='')
                sys.stdout.flush()
                u = rd.solve(mu)
                URB = rc.reconstruct(u)
                U = discretization.solve(mu)
                l2_err = np.max(discretization.l2_norm(U - URB))
                l2_err = np.inf if not np.isfinite(l2_err) else l2_err
                if l2_err > l2_err_max:
                    l2_err_max = l2_err
                    mumax = mu
            errs.append(l2_err_max)
            print(mumax)
            print()
            
        def tick_function(x, pos):
            N = int(x)
            if not N == 0:
                if N > real_rb_size:
                    value = (N, Ms[-1])
                else:
                    value = (N, Ms[N-1])
            else:
                value = (1, Ms[0])
            return str(value)
        
        plt.semilogy(Ns, errs)
        plt.xlabel('(RB size, CRB size)')
        plt.ylabel('max. L2-error')
        ax = plt.axes()
        ax.xaxis.set_major_formatter( FuncFormatter(tick_function))
        plt.show()
        
        plt.plot(Ns, corr)
        plt.xlabel('(RB size, CRB size)')
        plt.ylabel('M/N correlation')
        ax = plt.axes()
        ax.xaxis.set_major_formatter( FuncFormatter(tick_function))
        plt.show()
    
    if args['--plot-err']:
        discretization.visualize(U - URB)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    burgers_demo(args)
