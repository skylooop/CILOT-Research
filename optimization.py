import flax
import jax
import optax


import jax.numpy as jnp
from agent.iql.common import Model
from encoder_jax import Encoder_JAX
from absl import flags
from flax.training import train_state


FLAGS = flags.FLAGS


class OptimizeLoop_JAX:
    def __init__(self, agent_state_shape, expert_obs_shape):
        self.encoder, self.encoder_params, self.model_rng = self._prepare_encoder(agent_state_shape, expert_obs_shape)
        self.optimizer = optax.adamw(learning_rate=3e-4)
        self.encoder_state = train_state.TrainState.create(
                                        apply_fn = self.encoder.apply,
                                        tx=self.optimizer,
                                        params=self.encoder_params
                                    )
    
    @staticmethod
    def cost_matrix_fn(states_pair, expert_states_pair):
        dist = jnp.sqrt(jnp.sum(jnp.power(
            
                states_pair[:, None]
                - expert_states_pair[
                    None,
                ]
            ,jnp.array(2)), axis=-1))
        
        return dist
    
    def OT_loss(self, params, observations, next_observations, expert_states_pair, transport_matrix):
        embed_obs, obs_updated_params = self.encoder.apply(params, observations, mutable=['batch_stats'])
        next_embed_obs, next_updated_params = self.encoder.apply(params, next_observations, mutable=['batch_stats'])
        states_pair = jnp.concatenate([embed_obs, next_embed_obs], axis=1)
        cost_matrix = self.cost_matrix_fn(states_pair, expert_states_pair)
        loss = jnp.sum(transport_matrix * cost_matrix)
        
        return loss
        
        
    def optimize(self, obs, next_obs, expert_states_pair, transport_matrix):
        gradient_fn = jax.value_and_grad(self.OT_loss)
        loss, grads = gradient_fn(self.encoder_state.params, obs, next_obs, expert_states_pair, transport_matrix)
        self.encoder_state = self.encoder_state.apply_gradients(grads=grads)
        
        
    @staticmethod
    def _prepare_encoder(agent_state_shape: int, expert_state_shape: int):
        
        rng = jax.random.PRNGKey(FLAGS.seed)
        rng, dummy_inp_rng, model_rng = jax.random.split(rng, 3)
        encoder = Encoder_JAX(num_hidden=64, expert_state_shape=expert_state_shape, train=True)
        dummy = jax.random.normal(dummy_inp_rng, (1, agent_state_shape))
        params = encoder.init(model_rng, dummy)

        return encoder, params, model_rng
