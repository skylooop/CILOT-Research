from functools import partial

import jax
import optax
import jax.numpy as jnp
from agent.iql.common import Model, MLP
from absl import flags
from flax.training import train_state
import jax.random as random

FLAGS = flags.FLAGS


def create_encoder(agent_state_shape: int, expert_state_shape: int, lr=1e-4):
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, dummy_inp_rng, model_rng = jax.random.split(rng, 3)
    encoder = MLP((64, 64, expert_state_shape))
    dummy = jax.random.normal(dummy_inp_rng, (1, agent_state_shape))
    params = encoder.init(model_rng, dummy, training=True)

    optimizer = optax.adam(learning_rate=lr)
    encoder_state = train_state.TrainState.create(
        apply_fn=encoder.apply,
        tx=optimizer,
        params=params
    )
    return encoder_state

# test fourier features, delete
# @jax.jit
# def fourier_features(input, mapping_size: int = 256):
#     rng = jax.random.PRNGKey(FLAGS.seed)
#     B_gauss = random.normal(rng, (input.shape[1], mapping_size))
#     x_proj = (2.*jnp.pi*input) @ B_gauss
#
#     return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
    
@jax.jit
def embed(encoder: train_state.TrainState, agent_observations, agent_next_observations):
    return encoder.apply_fn(encoder.params, jnp.asarray(agent_observations)), encoder.apply_fn(encoder.params, jnp.asarray(agent_next_observations))

@jax.jit
def update_encoder(encoder: train_state.TrainState, sampled_agent_observations, sampled_agent_next_observations,
                   best_expert_traj, transport_matrix, cost_fn):
    
    def OT_loss(params):
        embeded_sampled_agent_observations = encoder.apply_fn(params, sampled_agent_observations)
        embeded_sampled_agent_next_observations = encoder.apply_fn(params, sampled_agent_next_observations)
        agent_embeded_states_pair = jnp.concatenate((embeded_sampled_agent_observations, embeded_sampled_agent_next_observations), axis=1)

        cost_matrix = cost_fn.all_pairs(best_expert_traj, agent_embeded_states_pair)
        loss = jnp.sum(transport_matrix * cost_matrix)

        return loss

    gradient_fn = jax.value_and_grad(OT_loss)
    loss, grads = gradient_fn(encoder.params)
    new_encoder = encoder.apply_gradients(grads=grads)

    return new_encoder, loss


