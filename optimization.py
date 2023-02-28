import flax
import jax
import optax
import jax.numpy as jnp
from agent.iql.common import Model, MLP
from absl import flags
from flax.training import train_state

import wandb

FLAGS = flags.FLAGS


def create_encoder(agent_state_shape: int, expert_state_shape: int):
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, dummy_inp_rng, model_rng = jax.random.split(rng, 3)
    #encoder = Encoder_JAX(num_hidden=64, expert_state_shape=expert_state_shape)
    encoder = MLP((64, expert_state_shape))
    dummy = jax.random.normal(dummy_inp_rng, (1, agent_state_shape))
    params = encoder.init(model_rng, dummy, training=True)

    optimizer = optax.adamw(learning_rate=3e-5)
    encoder_state = train_state.TrainState.create(
        apply_fn=encoder.apply,
        tx=optimizer,
        params=params
    )

    return encoder_state


@jax.jit
def embed(encoder: train_state.TrainState, obs, next_obs):
    return encoder.apply_fn(encoder.params, jnp.asarray(obs)), encoder.apply_fn(encoder.params, jnp.asarray(next_obs))


@jax.jit
def uptade_encoder(encoder: train_state.TrainState, obs, next_obs, expert_states_pair, transport_matrix):
    def cost_matrix_fn(states_pair, expert_states_pair):
        dist = jnp.sqrt(jnp.sum(jnp.power(

            states_pair[:, None]
            - expert_states_pair[
                None,
            ]
            , jnp.array(2)), axis=-1))

        return dist

    def OT_loss(params):
        embed_obs = encoder.apply_fn(params, obs)
        next_embed_obs = encoder.apply_fn(params, next_obs)
        states_pair = jnp.concatenate([embed_obs, next_embed_obs], axis=1)
        cost_matrix = cost_matrix_fn(states_pair, expert_states_pair)
        loss = jnp.sum(transport_matrix * cost_matrix)

        return loss

    gradient_fn = jax.value_and_grad(OT_loss)
    loss, grads = gradient_fn(encoder.params)
    new_encoder = encoder.apply_gradients(grads=grads)

    return new_encoder, loss


