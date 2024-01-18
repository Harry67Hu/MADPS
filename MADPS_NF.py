import numpy as np
import torch
from sacred import Ingredient
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model_NF import ConditionalVAE

madps_ingredient = Ingredient("madps")

@madps_ingredient.config
def config():

    # parameters of MADPS
    delay = 0
    reparameter_steps = 2000  # reparameter_steps should >= max_rb_steps
    max_rb_steps = 100
    reparameter_times = 200000 
    pretraining_times_seps = 1
    initial_as_the_same = False

    # parameters of MAPD
    z_features = 10
    kl_weight = 0.01
    kl_weight_seps = 0.0001
    customized_representation_learning = False
    customized_feature_size = 1

    # parameters of Actor-Critic
    batch_size = 128
    lr = 3e-4
    epochs = 10
    model_count = 15 # in most cases should be equal to agent_count!
    human_selected_idx = None # like [0, 0, 0, 0, 1, 1, 1, 1] or None 

    # ----------------------------------------------------
    # items in CVAE
    encoder_in = ["act_probs"]
    encoder_condition = ["hidden"]
    decoder_in = ["hidden"] # and actually a sampled "z"
    # reconstruct = ["act_probs"]
    # reconstruct = ["act_probs_with_mask", "customized_feature"] if customized_representation_learning else ["act_probs_with_mask"]
    reconstruct = ["customized_feature"] if customized_representation_learning else ["act_probs_with_mask"]

    # seps
    encoder_in_seps = ["agent"]
    decoder_in_seps = ["obs", "act"] # + "z"
    reconstruct_seps = ["next_obs", "rew"]

    policy_input = ["hidden"]
    policy_mask = ["act_mask"]

class rbDataSet(Dataset):
    @madps_ingredient.capture
    def __init__(self, rb, encoder_in, decoder_in, reconstruct, policy_mask):
        self.rb = rb
        self.data = []
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in encoder_in], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in decoder_in], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in reconstruct], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in policy_mask], dim=1))
        # print([x.shape for x in self.data])
    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, idx):
        return [x[idx, :] for x in self.data]
    
class rbDataSet_seps(Dataset):
    @madps_ingredient.capture
    def __init__(self, rb, encoder_in_seps, decoder_in_seps, reconstruct_seps):
        self.rb = rb
        self.data = []
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in encoder_in_seps], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in decoder_in_seps], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in reconstruct_seps], dim=1))
        
        print([x.shape for x in self.data])
    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, idx):
        return [x[idx, :] for x in self.data]

@madps_ingredient.capture
def compute_fusions(rb, agent_count, policy_model, only_measure, batch_size, lr, epochs, z_features, kl_weight, _log):
    
    # ---------------------------1. Train Conditional VAE------------------------------
    device = next(policy_model.parameters()).device

    dataset = rbDataSet(rb)

    encoder_input_size = dataset.data[0].shape[-1]
    encoder_condition_size = dataset.data[1].shape[-1]
    reconstruct_size = dataset.data[2].shape[-1]
    assert encoder_input_size == dataset.data[3].shape[-1]
   
    VAE_model = ConditionalVAE(z_features, encoder_input_size, encoder_condition_size, reconstruct_size)
    # print(VAE_model)
    VAE_model.to(device)
    optimizer = torch.optim.Adam(VAE_model.parameters(), lr=lr)

    criterion = nn.MSELoss(reduction="sum")

    def final_loss(bce_loss, mu, logvar):
        '''
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        '''
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + kl_weight*KLD
    
    def fit(model, dataloader):
        model.train()
        running_loss = 0.0
        for i, (encoder_in, encoder_condition, y, _) in enumerate(dataloader):
            (encoder_in, encoder_condition, y) = (encoder_in.to(device), encoder_condition.to(device), y.to(device))
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(encoder_in, encoder_condition)
            bce_loss = criterion(reconstruction, y)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(dataloader.dataset)
        return train_loss
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_loss = []

    for epoch in tqdm(range(epochs)):
        train_epoch_loss = fit(VAE_model, dataloader)
        train_loss.append(train_epoch_loss)

    print(f"Train Loss: {train_epoch_loss:.6f}")

    # ------------------------------2. Use VAE and agents' models to calculate dij------------------------------

    len1 = 0
    len2 = None

    BD_storage = None
    Hellinger_storage = None
    WD_storage = None
    batch_count = 1

    for i, (_, policy_input, _, _) in enumerate(dataloader):

        len1 = len(policy_input)
        if len2 is not None and len1 != len2: 
            batch_count = i
            break
        len2 = len1
        batch_count = i + 1

        # A. make hidden samples and get policy inputs
        hidden_sample_batch = policy_input.to(device) # [B,H]

        N = agent_count                         # agents_count
        E = policy_model.laac_shallow.shape[0]  # envs_count

        hidden_sample_expanded = hidden_sample_batch.unsqueeze(1).expand(-1, E, -1) # [B,E,H]
        hidden_sample_list = [hidden_sample_expanded.clone() for _ in range(N)]     # (N)[B,E,H]

        # B. generate VAE inputs
        with torch.no_grad():
            act_probs = policy_model.get_act_probs(hidden_sample_list) # (N)[B,E,A]
    
        VAE_input_batch = torch.stack(act_probs, dim=0)                # [N, B, E, A]
        VAE_condition_batch = torch.stack(hidden_sample_list, dim=0)   # [N, B, E, H]

        # C. compute Dij (only one env is enough)
        with torch.no_grad():
            z, mus, sigmas = VAE_model.encode(VAE_input_batch[:,:,0,:], VAE_condition_batch[:,:,0,:])     # [N, B, D]

        N, B, D = mus.shape

        if BD_storage is None: BD_storage = torch.zeros([N,N]).to(device)
        if Hellinger_storage is None: Hellinger_storage = torch.zeros([N,N]).to(device)
        if WD_storage is None: WD_storage = torch.zeros([N,N]).to(device)

        BD = calculate_N_Gaussians_BD(mus.transpose(0,1),sigmas.transpose(0,1)) # [B,N,D] --> [N,N]
        Hellinger = calculate_N_Gaussians_Hellinger_through_BD(BD)
        WD = calculate_N_Gaussians_WD(mus.transpose(0,1),sigmas.transpose(0,1))

        BD_storage += BD
        Hellinger_storage += Hellinger
        WD_storage += WD
    
    (BD, Hellinger, WD) = (BD_storage/batch_count, Hellinger_storage/batch_count, WD_storage/batch_count)

    # ------------------------------3. Use dij to automatically adjusting parameter sharing ------------------------------
    if not only_measure:
        N,N = WD.shape                          # agents_count
        E = policy_model.laac_shallow.shape[0]  # envs_count
        (laac_s, laac_d) = (policy_model.laac_shallow[0].cpu().numpy(), policy_model.laac_deep[0].cpu().numpy())
        epsilon1 = 0.5
        epsilon2 = 2 * epsilon1
        epsilon0 = 0.5 * epsilon1

        for i in range(N):
            for j in range(i+1,N):
                if laac_s[i] == laac_s[j]:
                    # shallow-net division
                    if WD[i,j] > epsilon2:
                        laac_s[j] = j
                        policy_model.copy_shallow_parameters(i,j)
                        break
                    # deep-net fusion
                    elif WD[i,j] < epsilon0:
                        if WD[laac_d[i],j] < epsilon0:
                            laac_d[j] = laac_d[i]
                            break
                        else:
                            laac_d[j] = i
                            break
                else:
                    if WD[i,j] < epsilon1:
                        # shallow-net fusion(normal case)
                        if WD[laac_s[i],j] < epsilon1:
                            laac_s[j] = laac_s[i]
                            break
                        # shallow-net fusion(forward case)
                        else:
                            laac_s[j] = i
                            break

        expanded_laac_s = np.tile(laac_s, (E, 1))
        expanded_laac_d = np.tile(laac_d, (E, 1))
        policy_model.laac_shallow = torch.tensor(expanded_laac_s).to(device)
        policy_model.laac_deep = torch.tensor(expanded_laac_d).to(device)

        return BD, Hellinger, WD, laac_s, laac_d
    else:
        return BD, Hellinger, WD, policy_model.laac_shallow, policy_model.laac_deep


def calculate_N_Gaussians_BD(mus, log_vars):
    '''
    Calculate the Bhattacharyya distance for N multi-dimensional Gaussian distributions using GPU.
    Assuming the input has shape mu.shape = [B, N, D], 
        where N is the number of multi-dimensional distributions and D is the dimension of each distribution.
    Assuming the input has shape sigma.shape = [B, N, D], 
        where the covariance matrix of the multi-dimensional distribution is a diagonal matrix, and sigma stores log_var.
    The output is a matrix of size [N, N].
    '''
    assert mus.dim() == 3
    assert mus.shape == log_vars.shape
    B, N, D = mus.shape

    mus = mus.transpose(0, 1).reshape(N, B * D)             # [N, B*D]
    log_vars = log_vars.transpose(0, 1).reshape(N, B * D)   # [N, B*D]

    # basic term
    mus1 = mus.unsqueeze(1).expand(-1, N, -1)
    mus2 = mus1.transpose(0, 1)
    log_vars1 = log_vars.unsqueeze(1).expand(-1, N, -1)
    log_vars2 = log_vars1.transpose(0, 1)

    sigmas1 = torch.exp(log_vars1)
    sigmas2 = torch.exp(log_vars2)
    mean_sigmas = (sigmas1 + sigmas2) / 2

    mu1_mu2_square = (mus1 - mus2) ** 2
    term3_frac = (sigmas1 + sigmas2)

    # main term
    term1 = (0.5 * torch.log(mean_sigmas).sum(dim=-1)) / (B * D)
    term2 = (-0.25 * (log_vars1 + log_vars2).sum(dim=-1)) / (B * D)
    term3 = ((0.25 * (mu1_mu2_square / term3_frac)).sum(dim=-1)) / (B * D)

    return (term1 + term2 + term3)


def calculate_N_Gaussians_Hellinger_through_BD(BD, max_value=15.0):
    '''
    Calculate the Hellinger distance through Bhattacharyya distance for N multi-dimensional Gaussian distributions.
    Clipping BD to avoid numerical instability.
    '''
    BD_clipped = torch.clamp(BD, min=-max_value, max=max_value)

    term1 = torch.exp(-BD_clipped)
    result = torch.sqrt(1 - term1)

    return result


def calculate_N_Gaussians_WD(mus, log_vars):
    '''
    Calculate the Wasserstein distance for N multi-dimensional Gaussian distributions using GPU.
    Assuming the input has shape mu.shape = [B, N, D], 
        where B is the batch dimension for averaging, N is the number of Gaussian distributions, and D is the dimension of each Gaussian distribution.
    Assuming the input has shape log_vars.shape = [B, N, D], 
        where the covariance matrix of the Gaussian distributions is a diagonal matrix, and log_vars stores log_var.
    The output is a matrix of size [N, N].
    '''
    assert mus.dim() == 3
    assert mus.shape == log_vars.shape
    B, N, D = mus.shape

    mus1 = mus.unsqueeze(1).expand(-1, N, -1, -1)
    mus2 = mus1.transpose(1, 2)

    vars1 = torch.exp(log_vars).unsqueeze(1).expand(-1, N, -1, -1)
    vars2 = vars1.transpose(1, 2)

    mean_diff = (mus1 - mus2) ** 2

    term1 = mean_diff.sum(dim=-1)   # [B, N, N]
    term2 = (((vars1**0.5) - (vars2**0.5)) ** 2).sum(dim=-1)

    output = (term1 + term2) ** 0.5

    return torch.mean(output, dim=0)   # [N, N]
