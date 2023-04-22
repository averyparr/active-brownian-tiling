from typing import Tuple

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
    r : numpy.ndarray
        A 3D array of particle positions with shape (n_frames, n_particles, 2).
    theta : numpy.ndarray
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
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    for frame in range(n_frames):
        # Clear the axis
        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

        for particle in range(n_particles):
            # Plot the particle position
            position = r[frame, particle]
            heading = theta[frame, particle]
            ax.plot(position[0], position[1], 'o')

            if show_arrows:
                # Add an arrow to represent the heading
                dx = np.cos(heading)
                dy = np.sin(heading)
                arrow = FancyArrow(position[0], position[1], dx, dy, width=0.1, length_includes_head=True, color='red')
                ax.add_patch(arrow)

        # Save the frame to the buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        frames.append(buf.getvalue())
        buf.close()

    # Save the frames as an animated GIF
    images = [Image.open(io.BytesIO(frame)) for frame in frames]
    images[0].save(gif_filename, save_all=True, append_images=images[1:], loop=0, duration=100)