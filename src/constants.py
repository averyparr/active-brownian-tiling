import jax.numpy as jnp

DEFAULT_DT = 0.01
DEFAULT_TOTAL_TIME = 10.
DEFAULT_V0 = 2.
DEFAULT_TRANSLATION_GAMMA = 1.
DEFAULT_TRANSLATION_DIFFUSION = 1e-1
DEFAULT_ROTATION_GAMMA = 0.01
DEFAULT_ROTATION_DIFFUSION = 1e-4
DEFAULT_OMEGA = 0.
DEFAULT_POISSON_ANGLE_REASSIGNMENT_RATE = 0.00001

DEFAULT_GIF_FPS = 30.
DEFAULT_DO_ANIMATION = True

DEFAULT_PERIODIC_BOUNDARY_SIZE = None

BOUNDING_BOX_STARTS = 40.*jnp.array([
    [1.,1.],
    [-1.,1.],
    [-1.,-1.],
    [1.,-1.],
])
BOUNDING_BOX_ENDS = 40.*jnp.array([
    [-1.,1.],
    [-1.,-1.],
    [1.,-1.],
    [1.,1.],
])