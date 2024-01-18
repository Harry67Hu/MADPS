import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gym.spaces import flatdim
 
class MADPSNet(nn.Module):
    def _init_layer(self, m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0) 
        return m

    def _make_fc(self, dims, activation=nn.ReLU, final_activation=None):
        mods = []
        input_size = dims[0]
        h_sizes = dims[1:]
        mods = [nn.Linear(input_size, h_sizes[0])]

        for i in range(len(h_sizes) - 1):
            mods.append(activation())
            mods.append(self._init_layer(nn.Linear(h_sizes[i], h_sizes[i + 1])))

        if final_activation:
            mods.append(final_activation())

        return nn.Sequential(*mods)

    def __init__(self, input_sizes, shallow_dims, deep_dims, base_seed=100):
            super().__init__()

            assert shallow_dims[-1] == deep_dims[0]
            self.laac_size = len(input_sizes)
            self.shallow_nets = nn.ModuleList()
            self.deep_nets = nn.ModuleList()

            for idx, size in enumerate(input_sizes):
                torch.manual_seed(base_seed) 
                
                # Create shallow network
                shallow_net_dims = [size] + shallow_dims
                self.shallow_nets.append(self._make_fc(shallow_net_dims,final_activation=nn.ReLU))
                
                # Create deep network
                deep_net_dims = deep_dims
                self.deep_nets.append(self._make_fc(deep_net_dims))

    def forward(self, inputs, laac_shallow, laac_deep):

        inputs = torch.stack(inputs)
        
        # Data flow of the shallow network
        shallow_outs = torch.stack([net(inputs) for net in self.shallow_nets])
        if inputs[0].dim() == 3:
            laac_shallow = laac_shallow.T.unsqueeze(0).unsqueeze(-1).unsqueeze(2)
            laac_shallow = laac_shallow.expand(1, *shallow_outs.shape[1:]).to(inputs.device)
        else:
            laac_shallow = laac_shallow.T.unsqueeze(0).unsqueeze(-1).expand(1, *shallow_outs.shape[1:]).to(inputs.device)
        shallow_outs = shallow_outs.gather(0, laac_shallow).split(1, dim=1)
        shallow_outs = [x.squeeze(0).squeeze(0) for x in shallow_outs]
        deep_inputs = torch.stack(shallow_outs)

        # Data flow of the deep network
        deep_outs = torch.stack([net(deep_inputs) for net in self.deep_nets])
        if inputs[0].dim() == 3:
            laac_deep = laac_deep.T.unsqueeze(0).unsqueeze(-1).unsqueeze(2)
            laac_deep = laac_deep.expand(1, *deep_outs.shape[1:]).to(inputs.device)
        else:
            laac_deep = laac_deep.T.unsqueeze(0).unsqueeze(-1).expand(1, *deep_outs.shape[1:]).to(inputs.device)
        deep_outs = deep_outs.gather(0, laac_deep).split(1, dim=1)
        out = [x.squeeze(0).squeeze(0) for x in deep_outs]

        return out
    
    def copy_parameters(self, i, j):
        '''
        Copy parameters from the i-th network to the j-th network.
        '''
        # Ensure valid indices
        if i < 0 or i >= self.laac_size or j < 0 or j >= self.laac_size:
            raise ValueError("Invalid indices provided for copy_parameters.")

        # Copy parameters for shallow nets
        for src_param, tgt_param in zip(self.shallow_nets[i].parameters(), self.shallow_nets[j].parameters()):
            tgt_param.data.copy_(src_param.data)


class MultiCategorical:
    def __init__(self, categoricals):
        self.categoricals = categoricals

    def __getitem__(self, key):
        return self.categoricals[key]

    def sample(self):
        return [c.sample().unsqueeze(-1) for c in self.categoricals]

    def log_probs(self, actions):

        return [
            c.log_prob(a.squeeze(-1)).unsqueeze(-1)
            for c, a in zip(self.categoricals, actions)
        ]

    def mode(self):
        return [c.mode for c in self.categoricals]

    def entropy(self):
        return [c.entropy() for c in self.categoricals]

class MultiAgentFCNetwork(nn.Module):
    '''
    A simple version of MADPS-Net (without the shallow network)
    '''
    def _init_layer(self, m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0) 
        return m

    def _make_fc(self, dims, activation=nn.ReLU, final_activation=None):
        mods = []
        input_size = dims[0]
        h_sizes = dims[1:]
        mods = [nn.Linear(input_size, h_sizes[0])]

        for i in range(len(h_sizes) - 1):
            mods.append(activation())
            mods.append(self._init_layer(nn.Linear(h_sizes[i], h_sizes[i + 1])))

        if final_activation:
            mods.append(final_activation())

        return nn.Sequential(*mods)

    def __init__(self, input_sizes, idims, base_seed=100):
        super().__init__()

        self.laac_size = len(input_sizes)
        self.independent = nn.ModuleList()

        for idx, size in enumerate(input_sizes):
            torch.manual_seed(base_seed)  
            dims = [size] + idims
            self.independent.append(self._make_fc(dims))

    def forward(self, inputs, laac_indices):

        inputs = torch.stack(inputs)
        out = torch.stack([net(inputs) for net in self.independent])

        if inputs[0].dim() == 3:
            laac_indices = laac_indices.T.unsqueeze(0).unsqueeze(-1).unsqueeze(2)
            laac_indices = laac_indices.expand(1, *out.shape[1:])
        else:
            laac_indices = laac_indices.T.unsqueeze(0).unsqueeze(-1).expand(1, *out.shape[1:])

        laac_indices = laac_indices.to(out.device)
        out = out.gather(0, laac_indices).split(1, dim=1)
        out = [x.squeeze(0).squeeze(0) for x in out]

        return out

class Policy(nn.Module):
    def __init__(self, obs_space, action_space, architecture, laac_size, state_size, initial_as_the_same=True, obs_shape=None, act_shape=None):
        super(Policy, self).__init__()

        if obs_space is None or action_space is None:
            assert obs_shape and act_shape 
            self.n_agents = len(obs_shape)
            obs_shape = obs_shape
            action_shape = act_shape

        else:
            self.n_agents = len(obs_space)
            self.laac_size = laac_size

            obs_space = obs_space[:laac_size]
            action_space = action_space[:laac_size]

            obs_shape = [flatdim(o) for o in obs_space]
            action_shape = [flatdim(a) for a in action_space]

        # self.actor = MultiAgentFCNetwork(
        #     obs_shape, architecture["actor"] + [action_shape[0]]
        # )

        self.actor = MADPSNet(obs_shape, architecture["actor"], architecture["actor"] + [action_shape[0]])

        if initial_as_the_same:
            for layers in self.actor.deep_nets:
                nn.init.orthogonal_(layers[-1].weight.data, gain=0.01)
        else:
            for idx, layers in enumerate(self.actor.shallow_nets):
                torch.manual_seed(100 + idx * 1000) 
                for layer in layers:
                    if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                        nn.init.orthogonal_(layer.weight.data, gain=0.5)
                        layer.weight.data += torch.randn_like(layer.weight.data) * 0.2  
                        layer.bias.data += torch.randn_like(layer.bias.data) * 0.2
            for idx, layers in enumerate(self.actor.deep_nets):
                torch.manual_seed(100 + idx * 1000) 
                for layer in layers:
                    if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                        nn.init.orthogonal_(layer.weight.data, gain=0.5)
                        layer.weight.data += torch.randn_like(layer.weight.data) * 0.2
                        layer.bias.data += torch.randn_like(layer.bias.data) * 0.2

        if state_size:
            state_size = len(obs_space) * [state_size]
        else:
            state_size = obs_shape

        self.critic = MultiAgentFCNetwork(
            state_size,
            architecture["critic"] + [1],
        )

        self.laac_params = nn.Parameter(torch.ones(self.n_agents-1, laac_size))
        print(self)

    def sample_laac(self, batch_size):
        sample = Categorical(logits=self.laac_params).sample([batch_size])
        self.laac_shallow = torch.cat((torch.zeros(batch_size,1).int().to(sample.device), sample), dim=1)
        self.laac_deep = torch.cat((torch.zeros(batch_size,1).int().to(sample.device), sample), dim=1)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError
    
    def copy_shallow_parameters(self, i, j):
        self.actor.copy_parameters(i, j)

    def act(self, inputs, action_mask=None):

        actor_features = self.actor(inputs, self.laac_shallow, self.laac_deep)
        
        if action_mask is not None:
        
            actor_features_with_mask = []
            for actor_feature, mask in zip(actor_features, action_mask):
                mask_values = -1e5 * (1 - mask) 
                mask_values = mask_values.to(actor_feature.device)
                masked_feature = actor_feature + mask_values
                actor_features_with_mask.append(masked_feature)

            act_probs = [F.softmax(actor_feature_masked, dim=-1) for actor_feature_masked in actor_features_with_mask]
            dist = MultiCategorical([Categorical(probs=probs) for probs in act_probs])
            action = dist.sample()
        else:
            act_probs = [F.softmax(actor_feature, dim=-1) for actor_feature in actor_features]
            dist = MultiCategorical([Categorical(probs=probs) for probs in act_probs])
            action = dist.sample()

        return action, act_probs

    def get_act_probs(self, inputs):
        actor_features = self.actor(inputs, self.laac_shallow, self.laac_deep)
        act_probs = [F.softmax(logits, dim=-1) for logits in actor_features]
        return act_probs
    

    def get_value(self, inputs):
        return torch.cat(self.critic(inputs, self.laac_deep), dim=-1)

    def evaluate_actions(self, inputs, action, action_mask=None, state=None):
        if not state:
            state = inputs

        value = self.get_value(state)
        actor_features = self.actor(inputs, self.laac_shallow, self.laac_deep)

        action_mask = [-9999999 * (1 - a) for a in action_mask] if action_mask else len(actor_features) * [0]
        action_mask = [tensor.to(actor_features[0].device) for tensor in action_mask]
        actor_features_with_mask = [x + s for x, s in zip(actor_features, action_mask)]
        act_probs = [F.softmax(logits, dim=-1) for logits in actor_features_with_mask]
        dist = MultiCategorical([Categorical(probs=probs) for probs in act_probs])

        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = dist.entropy()
        dist_entropy = sum([d.mean() for d in dist_entropy])

        return (
            value,
            action_log_probs,
            dist_entropy,
            act_probs,
        )
    
class ConditionalVAE(nn.Module):
    def __init__(self, features, input_size, condition_size, reconstruct_size, hidden_size=32):
        super(ConditionalVAE, self).__init__()
        HIDDEN = hidden_size
        self.features = features
        self.condition_size = condition_size

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size + condition_size, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=2 * features)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=features + condition_size, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=reconstruct_size),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def encode(self, x, c):
        x = torch.cat([x, c], dim=-1)
        x = self.encoder(x)
        mu = x[..., :self.features]
        log_var = x[..., self.features:]
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def forward(self, x, c):
        # encoding
        z, mu, log_var = self.encode(x, c)
        
        # decoding
        dec_input = torch.cat([z, c], dim=-1)
        reconstruction = self.decoder(dec_input)

        return reconstruction, mu, log_var



