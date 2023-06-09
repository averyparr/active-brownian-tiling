from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from PIL import Image
import io
from constants import *

def animate_particles(
        r: jnp.ndarray, 
        theta: jnp.ndarray, 
        poly_history: List[jnp.ndarray], 
        box_size: float,
        show_arrows: bool=False,
        gif_filename: str=f"{PROJECT_DIR}/plots/particles.gif",
        title=""
        ):
    """
    Create an animated GIF of particles with their positions in every frame and optionally display arrows
    representing their headings at each frame.

    Parameters
    ----------
    r : jnp.ndarray
        A 3D array of particle positions with shape (n_frames, n_particles, 2).
    theta : jnp.ndarray
        A 2D array of particle headings with shape (n_frames, n_particles).
    poly_history : jnp.ndarray
        A (n_walls,) list of (n_frames, W_i,2) Arrays of polygon vertices, where W_i
        is the number of vertices in polygon i. 
    box_size : float
        The width of the region to be animated. 
    show_arrows : bool, optional
        If True, arrows representing particle headings will be displayed in the animation. Default is False.
    gif_filename : str, optional
        The filename for the output animated GIF. Default is "particles.gif".

    Returns
    -------
    None
        The function saves an animated GIF file with the specified filename.
    """
    r = np.array(r)
    theta = np.array(theta)
    poly_history = [np.array(single_history) for single_history in poly_history]
    
    n_frames, n_particles, _ = r.shape

    # Create an array to store the frames
    frames = []

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-box_size*1.2/2, box_size*1.2/2)
    ax.set_ylim(-box_size*1.2/2, box_size*1.2/2)

    for frame in (pbar := tqdm(range(n_frames))):
        pbar.set_description(f"Processing {frame}")
        
        # Clear the axis
        ax.clear()
        ax.set_title(title)
        ax.set_xlim(-box_size*1.2/2, box_size*1.2/2)
        ax.set_ylim(-box_size*1.2/2, box_size*1.2/2)

        # Plot the particle positions
        positions = r[frame]
        ax.scatter(positions[:, 0], positions[:, 1],s=1)

        if show_arrows:
            # Calculate the heading vectors
            headings = np.column_stack((np.cos(theta[frame]), np.sin(theta[frame])))
            # Plot the arrows using quiver
            ax.quiver(positions[:, 0], positions[:, 1], headings[:, 0], headings[:, 1], color='red', angles='xy', scale_units='xy', scale=1)
        
        for single_history,c in zip(poly_history,("r","b","g","m","r","b","g","m")):
            vertices = single_history[frame]
            ax.fill(*vertices.transpose(),c=c, facecolor="none", linewidth=1)
        
        bounding_box_vertices = BOUNDING_BOX_VERTICES / DEFAULT_BOX_SIZE * box_size
        ax.fill(*bounding_box_vertices.transpose(),c="k",facecolor="none",linewidth=1)


        # Save the frame to the buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        frames.append(buf.getvalue())
        buf.close()

    # Save the frames as an animated GIF
    images = [Image.open(io.BytesIO(frame)) for frame in frames]
    images[0].save(gif_filename, save_all=True, append_images=images[1:], loop=0, duration=1000/DEFAULT_GIF_FPS)

def get_parameter_report(scan_param):
    excluded_params = [6,7]
    param_names = ["n",r"$v_0$",r"$\gamma_T$",r"$D_T$",r"$\gamma_R$",r"$D_R$",r"$\omega$",r"$\lambda$"]
    excluded_params.append(param_names.index(scan_param))
    param_vals = [
        DEFAULT_NUM_PARTICLES,
        DEFAULT_V0,
        DEFAULT_TRANSLATION_GAMMA,
        DEFAULT_TRANSLATION_DIFFUSION,
        DEFAULT_ROTATION_GAMMA,
        DEFAULT_ROTATION_DIFFUSION,
        DEFAULT_OMEGA,
        DEFAULT_TUMBLE_RATE,
        ]
    return " | ".join([param_names[i] + " = " + str(param_vals[i]) for i in range(len(param_names)) if i not in excluded_params])
