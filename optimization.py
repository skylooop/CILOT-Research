import jax
import optax
import jax.numpy as jnp
from agent.iql.common import Model, MLP
from absl import flags
from flax.training import train_state

FLAGS = flags.FLAGS


def create_encoder(agent_state_shape: int, expert_state_shape: int):
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, dummy_inp_rng, model_rng = jax.random.split(rng, 3)
    encoder = MLP((64, 64, expert_state_shape))
    dummy = jax.random.normal(dummy_inp_rng, (1, agent_state_shape))
    params = encoder.init(model_rng, dummy, training=True)

    optimizer = optax.adamw(learning_rate=3e-4)
    encoder_state = train_state.TrainState.create(
        apply_fn=encoder.apply,
        tx=optimizer,
        params=params
    )

    return encoder_state

@jax.jit
def embed(encoder: train_state.TrainState, agent_observations, agent_next_observations):
    return encoder.apply_fn(encoder.params, jnp.asarray(agent_observations)), encoder.apply_fn(encoder.params, jnp.asarray(agent_next_observations))

@jax.jit
def update_encoder(encoder: train_state.TrainState, sampled_agent_observations, sampled_agent_next_observations,
                   best_expert_traj_pairs, transport_matrix, cost_fn):
    
    #def cost_matrix_fn(states_pair, expert_states_pair):
    #    dist = jnp.sqrt(jnp.sum(jnp.power(states_pair[:, None] - expert_states_pair[None, ], jnp.array(2)), axis=-1))
    #    return dist

    def OT_loss(params):
        embeded_sampled_agent_observations = encoder.apply_fn(params, sampled_agent_observations)
        embeded_sampled_agent_next_observations = encoder.apply_fn(params, sampled_agent_next_observations)
        agent_embeded_states_pair = jnp.concatenate((embeded_sampled_agent_observations, embeded_sampled_agent_next_observations), axis=1)
        cost_matrix = cost_fn.all_pairs(agent_embeded_states_pair, best_expert_traj_pairs)
        loss = jnp.sum(jax.vmap(lambda x, y: jnp.multiply(x, y), in_axes=(0, None))(transport_matrix, cost_matrix))

        return loss

    gradient_fn = jax.value_and_grad(OT_loss)
    loss, grads = gradient_fn(encoder.params)
    new_encoder = encoder.apply_gradients(grads=grads)

    return new_encoder, loss


