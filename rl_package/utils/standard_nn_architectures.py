import numpy as np
import gym

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from collections import OrderedDict

from rl_package.utils.set_seed import set_seed


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def initialize_weights(mod, initialization_type, scale=1.4142):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            nn.init.normal_(p.data, mean=0., std=0.1)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")

def orthogonal_init(tensor, gain=1):
    '''
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

def get_activation(activation):
    if activation=='relu':
        return nn.ReLU()
    elif activation=='tanh':
        return nn.Tanh()
    elif activation=='sigmoid':
        return nn.Sigmoid()
    else:
        raise Exception('invalid activation key. should be one of these: relu, tanh or sigmoid')

def get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, NN_INIT, network_key):
    module_list = {}
    for layer, activation, i in zip( MLP_LAYERS, MLP_ACTIVATIONS, range(len(MLP_LAYERS)) ):
        if i==0:
            module_list['layer '+str(i)] = nn.Linear(num_inputs, layer)
            initialize_weights( module_list['layer '+str(i)] , NN_INIT)
            module_list['layer '+str(i)+' act'] = get_activation(activation)
            last_layer = layer
        else:
            module_list['layer '+str(i)] = nn.Linear(last_layer, layer)
            initialize_weights( module_list['layer '+str(i)] , NN_INIT)
            module_list['layer '+str(i)+' act'] = get_activation(activation)
            last_layer = layer
    if network_key=='actor':
        module_list['layer '+str(i+1)] = nn.Linear(last_layer, num_outputs)
        initialize_weights( module_list['layer '+str(i+1)] , NN_INIT, scale=1.0)
    elif network_key=='critic':
        module_list['layer '+str(i+1)] = nn.Linear(last_layer, 1)
        initialize_weights( module_list['layer '+str(i+1)] , NN_INIT, scale=1.0)
    else:
        raise Exception('invalid network key. should be one of these: actor or critic')
    
    if network_key=='actor':
        if ACTOR_FINAL_ACTIVATION is not None:
            module_list['final layer act']=get_activation(ACTOR_FINAL_ACTIVATION)

    return module_list


class ActorCriticDense(nn.Module):
    def __init__(self, env, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, NN_INIT, ACTOR_DIST_LOG_STD =0.0, seed=1):
        super(ActorCriticDense, self).__init__()
        '''
        env: OpenAI gym formatted RL environemnt
        MLP_LAYERS : array of neuron layers for ex. [8 , 8] for two layers with 8 neurons, 
        MLP_ACTIVATIONS : array of activation function for ex. ['relu','relu'] for relu activations for two ayers, 
        ACTOR_FINAL_ACTIVATION : activation function for final layer of actor network, 
        ACTOR_DIST_LOG_STD : log standard distribution for action distribution , 
        NN_INIT = initialization for network for ex.: 'orthogonal', 'xavier'
        '''
        set_seed(seed) # for reproducibility
        num_inputs  = env.observation_space.shape[0]
        if type(env.action_space)==gym.spaces.box.Box: # continous action
            self.action_type = 'continous'
            num_outputs = env.observation_space.shape[0]
        if type(env.action_space)==gym.spaces.discrete.Discrete: # discrete action
            self.action_type = 'discrete'
            num_outputs = env.action_space.n

        self.critic = nn.Sequential ( OrderedDict (get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, NN_INIT, network_key= 'critic') )  )
        if self.action_type == 'continous':
            self.actor = nn.Sequential ( OrderedDict (get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, NN_INIT, network_key='actor') ) )
            self.log_std = nn.Parameter(torch.ones(1, num_outputs) *  ACTOR_DIST_LOG_STD )
        if self.action_type == 'discrete':
            self.actor = nn.Sequential ( OrderedDict (get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION='sigmoid', NN_INIT=NN_INIT, network_key='actor') ) )
        
        # self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        if self.action_type=='continous':
            std   = self.log_std.exp().expand_as(mu)
            dist  = Normal(mu, std )
        if self.action_type=='discrete':
            dist = Categorical(probs)
        return dist, value

class QNetworkDense(nn.Module):
    def __init__(self, env, MLP_LAYERS, MLP_ACTIVATIONS, NN_INIT, ACTOR_FINAL_ACTIVATION=None, std=0.0, seed=1):
        '''
        mlp_layers : list of neurons in each hodden layer of the DQN network 
        mlp_activations : list of activation functions in each hodden layer of the DQN network
        nn_init : initialization for neural letwork: orthogonal, xavier etc. 
        '''
        super(QNetworkDense, self).__init__()
        set_seed(seed)
        num_inputs = env.observation_space.shape[0]
        num_outputs = env.action_space.n

        self.actor = nn.Sequential ( OrderedDict (get_moduledict(num_inputs, num_outputs, MLP_LAYERS, MLP_ACTIVATIONS, ACTOR_FINAL_ACTIVATION, NN_INIT, 'actor') )  )
        
    def forward(self, x):
        value = self.actor(x)
        return value
