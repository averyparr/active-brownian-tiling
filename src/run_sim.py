import jax.numpy as jnp
from jax import random as rand
from jax import jit

from typing import Tuple

from tqdm import trange

from constants import *

initial_random_key = rand.PRNGKey(678912390)

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

    return new_key, rand.normal(key, (num_particles,2), float) * jnp.sqrt(2*translationDiffusion / dt)
translation_noise = jit(translation_noise,static_argnums=(1,2,3))

def get_derivatives(
        r: jnp.ndarray, 
        theta: jnp.ndarray,

        rand_key: jnp.ndarray,
        dt: float = DEFAULT_DT,
        v0: float = DEFAULT_V0,
        translationGamma: float = DEFAULT_TRANSLATION_GAMMA,
        translationDiffusion: float = DEFAULT_TRANSLATION_DIFFUSION,
        rotationGamma: float = DEFAULT_ROTATION_GAMMA,
        rotationDiffusion: float = DEFAULT_ROTATION_DIFFUSION,
        omega: float = DEFAULT_OMEGA,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_particles = r.shape[0]

    heading_vector = jnp.array([jnp.cos(theta),jnp.sin(theta)]).transpose()
    rand_key, zeta = translation_noise(rand_key,num_particles,translationDiffusion,dt)
    r_dot = v0 * heading_vector + zeta/translationGamma

    rand_key, xi = rotation_noise(rand_key, num_particles, rotationDiffusion, dt)
    theta_dot = omega + xi/rotationGamma

    return rand_key, r_dot, theta_dot
get_derivatives = jit(get_derivatives, static_argnums=(3,4,5,6,7,8,9))

def run_sim(
        initial_positions: jnp.ndarray, 
        initial_heading_angles: jnp.ndarray,
        dt: float = DEFAULT_DT,
        total_time: float = DEFAULT_TOTAL_TIME,
        v0: float = DEFAULT_V0,
        translationGamma: float = DEFAULT_TRANSLATION_GAMMA,
        translationDiffusion: float = DEFAULT_TRANSLATION_DIFFUSION,
        rotationGamma: float = DEFAULT_ROTATION_GAMMA,
        rotationDiffusion: float = DEFAULT_ROTATION_DIFFUSION,
        omega: float = DEFAULT_OMEGA,
        poissonAngleReassignmentRate: float = DEFAULT_POISSON_ANGLE_REASSIGNMENT_RATE,
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

    rand_key = initial_random_key
    num_particles = initial_heading_angles.shape[0]
    num_steps = int(total_time / dt)
    r = initial_positions.copy()
    theta = initial_heading_angles.copy()

    for _ in trange(num_steps):    
        rand_key, r_dot, theta_dot = get_derivatives(r,theta,rand_key,dt,v0,translationGamma,translationDiffusion,rotationGamma,rotationDiffusion,omega)
        r = r + r_dot * dt
        theta = theta + theta_dot * dt
    
    return r, theta

print(run_sim(jnp.zeros((1000,2)),jnp.ones(1000)))

    
run_sim = jit(run_sim,static_argnums=(2,3,4,5,6,7,8,9,10))