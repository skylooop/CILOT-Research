import jax
import flax.linen as nn

import typing as tp

# Used in train_iql_ot.py on 114 line
class Encoder_JAX(nn.Module):
    '''
    Input: Agent state
    Output: Expert state
    '''
    num_hidden: int
    expert_state_shape: int
    train: bool
    
    @nn.compact
    def __call__(self, *args, **kwargs) -> tp.Any:
        embedding_agent_state = nn.Dense(features=self.num_hidden)(args[0])
        embedding_agent_state = nn.tanh(embedding_agent_state)
        embedding_agent_state = nn.BatchNorm(use_running_average=not self.train)(embedding_agent_state)
        
        embedding_agent_state = nn.Dense(features=self.num_hidden*2)(embedding_agent_state)
        embedding_agent_state = nn.tanh(embedding_agent_state)
        embedding_agent_state = nn.BatchNorm(use_running_average=not self.train)(embedding_agent_state)
        
        embedding_agent_state = nn.Dense(features=self.num_hidden)(embedding_agent_state)
        embedding_agent_state = nn.tanh(embedding_agent_state)
        embedding_agent_state = nn.BatchNorm(use_running_average=not self.train)(embedding_agent_state)
        
        embedding_agent_state = nn.Dense(features=self.expert_state_shape)(embedding_agent_state)
        embedding_agent_state = nn.tanh(embedding_agent_state)
        
        return embedding_agent_state
        
        
        
if __name__ == "__main__":
    encoder = Encoder_JAX(17, 8, train=True)
    rng = jax.random.PRNGKey(1)
    rng, inp, rng_model = jax.random.split(rng, 3)
    inp = jax.random.normal(inp, (1, 17))
    params = encoder.init(rng_model, inp)
    
    encoder.apply(params, inp, mutable=['batch_stats'])
    
    print(encoder)