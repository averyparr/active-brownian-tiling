import jax.numpy as jnp
from jax import random as rand

## creating a base file to scaffold on for when we actually write things

rand_key = rand.PRNGKey(678912390)

print(rand.randint(rand_key,(50,),minval=0,maxval=10))
print(rand.uniform(rand_key,(50,),minval=0,maxval=10))