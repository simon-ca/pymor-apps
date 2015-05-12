Viscous Burgers demo applications
=================================

This project contains three Python scripts which solve the viscous burgers equation and perform a model order reduction
using the model order reduction library pyMOR.
In the following we will give a short overview of the underlying mathematical methods we use. If you just want to know
what the scripts do just read the last section.

The viscous Burgers equation
----------------------------

The problem is to solve

    ∂_t u(x, t, μ)  +  ∇ ⋅ (v * u(x, t, μ)^μ_1)  -  ∇ ⋅ (μ_2 * ∇ u(x, t, μ) = 0
                                                                 u(x, 0, μ) = u_0(x)

for u with t in [0, 0.3], x in [0, 2] x [0, 1].

We employ finite volumes to discretize in space and for discretization in time we use the explicit Euler method for
convection and the implicit Euler method for diffusion. You can find the implementation for this sort of time stepping
in algorithms/timestepping.

Model order reduction
---------------------

Since the advection operator is not efficiently online computable in a reduced basis context we use empirical operator
interpolation to enable efficient online computation. pyMOR already provides algorithms to do that. Additionally one can
find the so called PODEI-Greedy algorithm (see Bibliography) in algorithms/ei_greedy. This algorithms basically combines
the greedy algorithm for construction of the collateral basis used for empirical operator interpolation and the greedy
algorithm for construction of the reduced basis. In particular it chooses the size of the collateral basis appropriate
to the size of reduced basis. Thus causes the error to decrease smoothly during basis generation.

The demos explained
-------------------

1. *viscous_burgers demo*: Executing the script, for example with `./viscous_burgers.py 2 0.01`, solves the equation for
                           the given parameters (in our example with exponent *μ_1 = 2* and diffusion *μ_2 = 0.01*)
                           and plots the solution.

2. *viscous_burgers_ei demo*: This demo now employs model order reduction via the reduced basis method after
                              interpolating the convection operator via empirical operator interpolation. For example
                              you can execute this demo with
                              `./viscous_burgers_ei.py 1 2 0 0.1 10 5 200 10 5 100 --plot-error-landscape`.
                              The first four numbers `1 2 0 0.1` determine the parameter ranges for the exponent *μ_1*
                              and diffusion *μ_2*. The next three numbers `10 5 200` state the number of snapshots for
                              exponent and diffusion and basis size for empirical interpolation while the last three
                              `10 5 100` serve the same purpose for the reduced basis method. The optional argument
                              `--plot-error-landscape` creates a plot at the end of the script that shows the maximum
                              approximation error depending on reduced basis size and collateral basis size.

3. *viscous_burgers_podei_greedy demo*: This demo executes a model order reduction process similar to the demo before.
                                        The major difference is that it makes use of the PODEI-Greedy algorithm
                                        (see Bibliography). The implementation we use here can be found in
                                        algorithms/ei_greedy.
                                        You can start the demo with
                                        `./viscous_burgers_podei_greedy 1 2 0 0.1 100 10 5 20 2 2 --plot-error-landscape --plot-error-correlation`
                                        for example. The first four numbers `1 2 0 0.1` again determine the parameter
                                        ranges. The following three numbers `100 10 5` are the size of the reduced basis
                                        and number of snapshots. The last three numbers `20 2 2` determine the initial
                                        collateral basis size and number of snapshots for the initialization of the
                                        PODEI-Greedy algorithm. The optional argument `-plot-error-landscape` creates an
                                        error plot like the one of the viscous_burgers_ei demo while
                                        `--plot-error-correlation` plots the maximum  approximation error versus the
                                        size of reduced and collateral basis computed by the PODEI-Greedy algorithm.

Bibliography
------------

1. M. Drohmann, B. Haasdonk and M. Ohlberger: Reduced basis approximation for nonlinear parametrized evolution
   equations based on empirical operator interpolation. SIAM J. Sci. Comput., 34.2, 937-969, 2012.
