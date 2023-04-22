from typing import Tuple

import jax.numpy as jnp
from jax import random as rand
from jax import jit


from tqdm import trange

from constants import *

initial_random_key = rand.PRNGKey(678912390)

from visualization import animate_particles

def rotation_noise(rand_key, num_particles: int, rotationDiffusion: float, dt: float) -> Tuple[jnp.ndarray,jnp.array]:
    r"""
    Computes `\delta`-correlated noise used to cause drift in `\theta(t)`.
    Becaus we work in discrete-time, we must ensure that 

    `\int_0^T \langle \xi(t)\xi(t')\rangle = 2D_R`

    We will draw our `\xi` values as IID zero-mean Gaussians, so two values of
    `\xi(t)`, `\xi(t')` will never be correlated, except when `t=t'`. But 
    to ensure that the integral takes on the correct value, we must recall that
    if `\xi` is Norm(0, `\sigma`) distributed, then because it takes this value
    for our discrete time `dt`, we have
    
    `\int_0^T \langle \xi^2(t)\rangle dt = dt\sigma^2 = 2D_R`

    so we must draw our noise terms from a Norm(0,`\sqrt{2D_R/dt}`) distribution.
    """

    key, new_key = rand.split(rand_key)

    return new_key, rand.normal(key, (num_particles,), float) * jnp.sqrt(2*rotationDiffusion / dt)
rotation_noise = jit(rotation_noise,static_argnums=(1,2,3))

def translation_noise(rand_key, num_particles: int, translationDiffusion: float, dt: float) -> Tuple[jnp.array,jnp.array]: 
    r"""
    Computes `\delta`-correlated noise used to cause drift in `r(t)`. 
    Because we work in discrete-time, we must ensure that 

    `\int_0^T \langle\zeta(t)\zeta(t')\rangle = 2D_T`

    We will draw our `\zeta` values as IID zero-mean 2D Gaussians, so two components
    `\zeta_i(t)`, `\zeta_j(t')` will never be correltated unless `t=t'` and `i=j`. 
    But to ensure that the integral takes on the correct value, we must recall that
    if our `\zeta_i` is Norm(0,`\sigma`) distributed, then for discrete time 

    `\int_0^T\langle \xi^2_i(t)\rangle dt = dt\sigma^1 = 2D_T`

    so we must draw our noise terms from a Norm(0,`\sqrt{2D_T}/dt`) distribution. 
    """

    key, new_key = rand.split(rand_key)

    return new_key, rand.normal(key, (num_particles,1,2), float) * jnp.sqrt(2*translationDiffusion / dt)
translation_noise = jit(translation_noise,static_argnums=(1,2,3))

def get_derivatives(
        r: jnp.ndarray, 
        theta: jnp.ndarray,

        rand_key: jnp.ndarray,
        sim_params: dict = {},
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_particles = r.shape[0]

    dt =                                sim_params.get("dt",DEFAULT_DT)
    v0 =                                sim_params.get("v0",DEFAULT_V0)
    translationGamma =                  sim_params.get("translationGamma",DEFAULT_TRANSLATION_GAMMA)
    translationDiffusion =              sim_params.get("translationDiffusion",DEFAULT_TRANSLATION_DIFFUSION)
    rotationGamma =                     sim_params.get("rotationGamma",DEFAULT_ROTATION_GAMMA)
    rotationDiffusion =                 sim_params.get("rotationDiffusion",DEFAULT_ROTATION_DIFFUSION)
    omega =                             sim_params.get("omega",DEFAULT_OMEGA)

    heading_vector = jnp.array([[jnp.cos(theta),jnp.sin(theta)]]).transpose([2,0,1])
    rand_key, zeta = translation_noise(rand_key,num_particles,translationDiffusion,dt)
    r_dot = v0 * heading_vector + zeta/translationGamma # should have shape (n,1,2).

    rand_key, xi = rotation_noise(rand_key, num_particles, rotationDiffusion, dt)
    theta_dot = omega + xi/rotationGamma

    return rand_key, r_dot, theta_dot
get_derivatives = jit(get_derivatives)

def run_sim(
        initial_positions: jnp.ndarray, 
        initial_heading_angles: jnp.ndarray,
        sim_params: dict = {},
        ) -> jnp.ndarray: 
    r"""
    We work with two-dimensional Active Brownian Particles (ABPs). 
    These have a preferred direction of motion, parametrized by a 
    heading variable `\theta(t)` and a constant preferred velocity v0. 

    These particles work in the overdamped limit, where forces translate
    directly into velocities, scaled by the drag coefficient `gamma`. 
    The translational equation of motion for our ABPs is given by 

    `\gamma_T (\dot{r} - v_0n(t)) = \zeta(t)`

    where `\zeta(t)` is a zero-mean, `\delta`-correlated noise term: 

    `\langle \zeta_i(t)\zeta_j(t')\rangle = 2D_T \delta(t-t')\delta_{ij}`. 

    and `n(t) = [\cos(\theta(t)),\sin(\theta(t))]` is the heading vector
    for the ABP. The particle also experiences drift in its heading, which
    we assume is primarily diffusive: 

    `\gamma_R(\dot{\theta} -\omega) = \xi(t)`

    where `\xi(t)` is another `\delta`-correlated zero mean rotation term: 

    `\langle \xi(t)\xi(t')\rangle = 2D_R \delta(t-t')`

    and `\omega` is a natural rotation rate. These particles are also assumed
    to completely re-assign their heading on a Unif[0,2`\pi`] distribution with
    Poisson-like dynamics, parametrized by `p` the rate of transition per unit
    time. 


    Our particles are taken to be spheres of radius `R = 1`. They will interact
    with walls parametrized by a sequence of points. 
    """

    dt =                                sim_params.get("dt",DEFAULT_DT)
    total_time =                        sim_params.get("total_time",DEFAULT_TOTAL_TIME)
    poissonAngleReassignmentRate =      sim_params.get("poissonAngleReassignmentRate",DEFAULT_POISSON_ANGLE_REASSIGNMENT_RATE)

    r_history = []
    theta_history = []

    rand_key = initial_random_key

    num_particles = initial_heading_angles.shape[0]
    num_steps = int(total_time / dt)

    r = initial_positions.copy().reshape((num_particles,1,2))
    theta = initial_heading_angles.copy()

    # We know that angle reassignment is done as a Poisson process, so the time
    # between events is distributed as Expo(poissonAngleReassignmentRate).
    
    key, rand_key = rand.split(rand_key)
    time_until_angle_reassignment = rand.exponential(key,(num_particles,),float) / poissonAngleReassignmentRate
    next_reassignment_all_particles = (time_until_angle_reassignment/dt).astype(jnp.int32)
    next_reassignment_event = jnp.min(next_reassignment_all_particles)

    
    # Suppose we have W walls. Walls are parametrized as two (W,2) arrays 
    # describing their start and end points, with the line between these points
    # being the "wall". At any given timestep, we determine if a particle is
    # nearby wall `w_i` parametrized by points `s_i, e_i` using two vectors:
    # a vector `f_i = (e_i - s_i)/\ell_i` where `\ell_i` is the length of the
    # wall and given by ||e_i-s_i||, and a normal unit vector `n_i` to the
    # wall. Suppose a particle is at position r. We describe its position in 
    # the basis of "along the wall" and "normal from the wall". In particular, 
    # we compute  `f(r) = (r - s_i) @ f_i` as our distance along the wall, 
    # normalized so `f(s_i) = 0` and `f(e_i) = ||e_i-s_i|| = \ell_i`. We 
    # additionally compute `n(r) = (r-s_i) @ n_i` to give our distance from 
    # the wall. 
    # 
    # Suppose our particle has radius R. We consider it as "touching" the wall
    # when -R < f(r) < \ell_i + R and |n(r)| < R. We model the wall as exerting
    # a normal force by saying 


    wall_starts = jnp.array([
        [1.,0.],
        [1.,0.],
        ])
    wall_ends = jnp.array([
        [0.,1.],
        [0.,-1.],
        ])

    wall_diffs = wall_ends - wall_starts
    diff_mags = jnp.apply_along_axis(jnp.linalg.norm,1,wall_diffs)
    fraction_along_wall_vec = wall_diffs / diff_mags**2
    rot90_arr = jnp.array([[0,-1],[1,0]])
    distance_from_wall_vec = rot90_arr @ (wall_diffs / diff_mags)

    for step in trange(num_steps):    
        rand_key, r_dot, theta_dot = get_derivatives(r,theta,rand_key,sim_params)
        r = r + r_dot * dt
        theta = theta + theta_dot * dt

        r_history.append(r.squeeze())
        theta_history.append(theta.squeeze())

        if step == next_reassignment_event:
            print("FAILED")
            reassign_which_particles = (next_reassignment_all_particles==step)
            num_reassignments = jnp.count_nonzero(reassign_which_particles)
            
            key, rand_key = rand.split(rand_key)
            new_thetas = rand.uniform(key,(num_reassignments,),float,0.,2*jnp.pi)
            theta = theta.at[reassign_which_particles].set(new_thetas)
            
            key, rand_key = rand.split(rand_key)
            next_reassignment_of_reassigned_particles = step + (rand.exponential(key,(num_reassignments,),float) / poissonAngleReassignmentRate / dt).astype(jnp.int32)
            next_reassignment_all_particles = next_reassignment_all_particles.at[reassign_which_particles].set(next_reassignment_of_reassigned_particles)

            next_reassignment_event = jnp.min(next_reassignment_all_particles)

    animate_particles(r_history,theta_history,10.,10.,show_arrows=True)

    return r, theta

sim_params = {"total_time": 10.}

resulting_r,resulting_theta = run_sim(jnp.zeros((10,2)),jnp.ones(10),sim_params)

print(resulting_r)
print(resulting_theta)