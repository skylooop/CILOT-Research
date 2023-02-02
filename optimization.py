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
        self.optimizer = optax.adamw(learning_rate=3e-4).init(self.encoder_params)
        self.encoder_state = train_state.TrainState.create(
                                        apply_fn = self.encoder.apply,
                                        tx=self.optimizer,
                                        params=self.encoder_params
                                    )
    @staticmethod
    def prepare_loss(optimal_map, cost_matrix):
        def OT_loss():
            loss = (optimal_map * cost_matrix).sum()
            return loss
        return jax.jit(OT_loss())
    
    @staticmethod
    def embedding(self, observations, next_observations):
        embed_obs, obs_updated_params = self.encoder_class.encoder.apply(self.encoder_params, observations, mutable=['batch_stats'])
        next_embed_obs, next_updated_params = self.encoder_class.encoder.apply(self.encoder_params, next_observations, mutable=['batch_stats'])
        
        return embed_obs, next_embed_obs
    
    def optimize(self, embed_obs, next_embed_obs, transport_matrix, cost_matrix):
        loss_fn = self.prepare_loss(transport_matrix, cost_matrix)
        gradient_fn = jax.value_and_grad(loss_fn)
        loss, grads = gradient_fn(self.encoder_state.params)
        self.encoder_state = self.encoder_state.apply_gradients(grads=grads)
        
        
    @staticmethod
    def _prepare_encoder(agent_state_shape: int, expert_state_shape: int):
        
        rng = jax.random.PRNGKey(FLAGS.seed)
        rng, dummy_inp_rng, model_rng = jax.random.split(rng, 4)
        encoder = Encoder_JAX(num_hidden=64, expert_state_shape=expert_state_shape, train=True)
        dummy = jax.random.normal(dummy_inp_rng, (1, agent_state_shape))
        params = encoder.init(model_rng, dummy)
        
        
        return encoder, params, model_rng
