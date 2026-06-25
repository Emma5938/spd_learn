import torch.nn.functional as F 
import torch 
from collections import defaultdict
from spd_learn.modules import BiMap

# POWEREIG LAYER
def get_phi_values(model):
    phi_dict = {}

    for domain, block in model.domains_block.items():
        for layer_name in ["activation1"]:                                  #, "activation2"
            layer = block[layer_name]

            if hasattr(layer, "phi"):                                       # Checks whether layer has the ‘phi’ attribute
                # phi_value = F.softplus(layer.phi).cpu()                   # Removes the tensor from the autograd, places the tensor on the CPU, and `item()` converts a tensor to a scalar
                phi_value = layer.phi.detach().cpu()
                key = f"{domain}_{layer_name}"
                phi_dict[key] = phi_value

    return phi_dict



# TANHEIG LAYER
def get_a_values(model):
    a_dict = {}

    for domain, block in model.domains_block.items():
        for layer_name in ["activation1"]:                                  #, "activation2"
            layer = block[layer_name]

            if hasattr(layer, "a"):
                # a_value = F.softplus(layer.a).cpu()                   
                a_value = layer.a.detach().cpu()
                key = f"{domain}_{layer_name}"
                a_dict[key] = a_value

    return a_dict

def get_b_values(model):
    b_dict = {}

    for domain, block in model.domains_block.items():
        for layer_name in ["activation1"]:                                  #, "activation2"
            layer = block[layer_name]

            if hasattr(layer, "b"):
                b_value = F.softplus(layer.b).cpu()
                b_value = layer.b.detach().cpu()
                key = f"{domain}_{layer_name}"
                b_dict[key] = b_value

    return b_dict



# SPAEIG LAYER 
def get_mu_values(model):
    mu_dict = {}

    for domain, block in model.domains_block.items():
        for layer_name in ["activation1"]:                                  #, "activation2"
            layer = block[layer_name]

            if hasattr(layer, "mu"):
                mu_value = torch.sigmoid(layer.mu).cpu()
                # mu_value = layer.mu.detach().cpu()
                key = f"{domain}_{layer_name}"
                mu_dict[key] = mu_value

    return mu_dict

def get_sigma_values(model):
    sigma_dict = {}

    for domain, block in model.domains_block.items():
        for layer_name in ["activation1"]:                                  #, "activation2"
            layer = block[layer_name]

            if hasattr(layer, "sigma"):
                sigma_value = F.softplus(layer.sigma).cpu()  + 1e-4
                #sigma_value = layer.sigma.detach().cpu()
                key = f"{domain}_{layer_name}"
                sigma_dict[key] = sigma_value

    return sigma_dict

def get_A_values(model):
    A_dict = {}

    for domain, block in model.domains_block.items():
        for layer_name in ["activation1"]:                                  #, "activation2"
            layer = block[layer_name]

            if hasattr(layer, "A"):
                A_value = F.softplus(layer.A).cpu()
                #A_value = layer.A.detach().cpu()
                key = f"{domain}_{layer_name}"
                A_dict[key] = A_value

    return A_dict



# CoshP - ExpT - SinP
def get_alpha_values(model):
    alpha_dict = {}

    for domain, block in model.domains_block.items():
        for layer_name in ["activation1"]:                                  #, "activation2"
            layer = block[layer_name]

            if hasattr(layer, "alpha"):                                   
                A_value = F.softplus(layer.alpha).cpu()                   
                #A_value = layer.A.detach().cpu()
                key = f"{domain}_{layer_name}"
                alpha_dict[key] = A_value

    return alpha_dict




# Monitoring of BiMap 
class EigenvalueMonitor : 
    def __init__(self):
        self.storage = defaultdict(list)                                    # Create an empty list 
        self.hooks = []                                                     # Contains the layer hooks

    def hook(self, name):
        def fn(module, input, output):

            with torch.no_grad():
                eigvals = torch.linalg.eigvalsh(output)                     # Eigenvalue decomposition of the output 
                self.storage[name].append(eigvals.detach().cpu())           # Storage

        return fn

    def attach(self, model):                                                # Add the hooks 

        for domain, block in model.domains_block.items():                   # through domains
            for layer_name, layer in block.items():                         # through layers
                if isinstance(layer, BiMap):                                # hook only on BiMaps
                    h = layer.register_forward_hook(self.hook(f"{domain}_{layer_name}"))
                    self.hooks.append(h)

    def reset(self):
        self.storage = defaultdict(list)                                    # Reset the list at each epoch
        
    def summarize(self):

        summary = {}
        for layer_name, eigvals_list in self.storage.items():
            eigvals = torch.cat(eigvals_list, dim=0)

            summary[layer_name] = {
                "min": eigvals.min().item(),
                "max": eigvals.max().item(),
                "median": eigvals.median().item(),
                "mean": eigvals.mean().item(),
            }

        return summary

    def remove(self):                                                       # Remove the hook 

        for h in self.hooks:
            h.remove()


# MONITORING DEs ALPHA de polynomial
def get_w_values(model):
    w_dict = {}

    for domain, block in model.domains_block.items():
        for layer_name in ["activation1"]:                                  #, "activation2"
            layer = block[layer_name]

            if hasattr(layer, "w"):                                  
                w_value = F.softplus(layer.w).cpu()           
                key = f"{domain}_{layer_name}"
                w_dict[key] = w_value

    return w_dict

# MONITORING DE ALPHA DE expP
def get_alphaE_values(model):
    a_dict = {}

    for domain, block in model.domains_block.items():
        for layer_name in ["activation1"]: #, "activation2"
            layer = block[layer_name]

            if hasattr(layer, "alphaE"):                                  
                a_value = F.softplus(layer.alphaE).cpu()           
                key = f"{domain}_{layer_name}"
                a_dict[key] = a_value

    return a_dict







