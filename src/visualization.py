from typing import Tuple
from tqdm import trange

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from PIL import Image
import io

def animate_particles(r: jnp.ndarray, theta: jnp.ndarray, width: float, height: float, show_arrows: bool=False, gif_filename: str="particles.gif"):
    """
    Create an animated GIF of particles with their positions in every frame and optionally display arrows
    representing their headings at each frame.

    Parameters
    ----------
    r : jnp.ndarray
        A 3D array of particle positions with shape (n_frames, n_particles, 2).
    theta : jnp.ndarray
        A 2D array of particle headings with shape (n_frames, n_particles).
    width : float
        The width of the region to be animated.
    height : float
        The height of the region to be animated.
    show_arrows : bool, optional
        If True, arrows representing particle headings will be displayed in the animation. Default is False.
    gif_filename : str, optional
        The filename for the output animated GIF. Default is "particles.gif".

    Returns
    -------
    None
        The function saves an animated GIF file with the specified filename.
    """
    r = np.asarray(r)
    theta = np.asarray(theta)
    
    n_frames, n_particles, _ = r.shape

    # Create an array to store the frames
    frames = []

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-width/2, width/2)
    ax.set_ylim(-height/2, height/2)

    for frame in (pbar := trange(n_frames)):
        pbar.set_description(f"Processing {frame}")
        
        # Clear the axis
        ax.clear()
        ax.set_xlim(-width/2, width/2)
        ax.set_ylim(-height/2, height/2)

        # Plot the particle positions
        positions = r[frame]
        ax.scatter(positions[:, 0], positions[:, 1])

        if show_arrows:
            # Calculate the heading vectors
            headings = np.column_stack((np.cos(theta[frame]), np.sin(theta[frame])))
            # Plot the arrows using quiver
            ax.quiver(positions[:, 0], positions[:, 1], headings[:, 0], headings[:, 1], color='red', angles='xy', scale_units='xy', scale=1)

        # Save the frame to the buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        frames.append(buf.getvalue())
        buf.close()

    # Save the frames as an animated GIF
    images = [Image.open(io.BytesIO(frame)) for frame in frames]
    images[0].save(gif_filename, save_all=True, append_images=images[1:], loop=0, duration=100)