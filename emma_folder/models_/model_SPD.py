
import torch
import torch.nn as nn
from warnings import warn
from spd_learn.functional import covariance
from spd_learn.modules import BiMap, CovLayer, LogEig, SPDBatchNormMeanVar, ReEig

from emma_folder.activation.spectral import PowerEig, SpAEig, TanhEig
from emma_folder.activation.elementwise import activationSPD, coshP, polynomialActivation, sinhP, expT, expP

class modelSPDNet(nn.Module): 

    def __init__(self, activation ="reeig", subspacedim1 = None, subspacedim2 = None, subspacedim3 = None, subspacedim4 = None, threshold = 1e-4, n_chans = None, domains = None , upper = True,  n_outputs = None):
        super().__init__()
        
        if n_chans is None : 
            raise ValueError("n_chans must be provided")
        if domains is None :                                                
            raise ValueError("domains must be provided")
        
        self.activation_type = activation
        self.threshold = threshold
 
        # -------------------------------------
        # Number of dimensions based on n_chans 
        # -------------------------------------
        if n_chans < 20:
            if subspacedim1 is None : subspacedim1 = int(n_chans * 0.5)
            #if subspacedim2 is None : subspacedim2 = 6

        elif 20 <= n_chans < 35:
            if subspacedim1 is None : subspacedim1 = int(n_chans * 0.5)
            #if subspacedim2 is None : subspacedim2 = 6

        elif 35 <= n_chans < 80:
            if subspacedim1 is None : subspacedim1 = int(n_chans * 0.5)
            if subspacedim2 is None : subspacedim2 = int(subspacedim1 * 0.5)
            #if subspacedim3 is None : subspacedim3 = 6

        elif 80 <= n_chans < 130:
            if subspacedim1 is None: subspacedim1 = int(n_chans * 0.5)
            if subspacedim2 is None: subspacedim2 = int(subspacedim1 * 0.5)
            #if subspacedim3 is None: subspacedim3 = int(subspacedim2 * 0.5)
            #if subspacedim4 is None: subspacedim4 = 6

 
        # -------------------------------------
        # domain dependent architecture
        # -------------------------------------
        self.domains_block = nn.ModuleDict()                                   # Domain specific 
        for domain in domains : 
            layers = {
                #"batchnorm" : SPDBatchNormMeanVar(n_chans),                    # num_features = size of input matrices
                #"activation1" : self._make_activation(n = n_chans),
                "bimap1" : BiMap(n_chans, subspacedim1),
                #"reeig1" : ReEig(self.threshold),
                "activation1": self._make_activation(n = subspacedim1),
                #"bimap2" : BiMap(subspacedim1, subspacedim2),
                #"reeig2" : ReEig(self.threshold),
                #"activation2": self._make_activation(n = subspacedim2),
            }
            
            last_dim = subspacedim1

            # 3rd BiMap if defined
            if subspacedim3 is not None:
                layers["bimap3"] = BiMap(subspacedim2, subspacedim3)
                layers["activation3"] = self._make_activation(n = subspacedim3)
                #layers["reeig3"] = ReEig(self.threshold)
                last_dim = subspacedim3

            # 4th BiMap if defined
            if subspacedim4 is not None:
                layers["bimap4"] = BiMap(last_dim, subspacedim4)
                #layers["activation4"] = self._make_activation()
                layers["reeig4"] = ReEig(self.threshold)
                last_dim = subspacedim4

            self.domains_block[domain] = nn.ModuleDict(layers)

 
        # -------------------------------------
        # Logeig and linear layer 
        # -------------------------------------
        self.logeig = LogEig(upper = upper)                                     # if Upper = True : vech                    
        self.len_last_layer = (
            last_dim * (last_dim + 1) // 2 if upper else last_dim**2
        )

        self.classifier = nn.Linear(self.len_last_layer, n_outputs)             # Linear layer


    def _make_activation(self, n):

        if self.activation_type == "reeig":
            return ReEig(self.threshold)

        elif self.activation_type == "powereig":
            return PowerEig(threshold=self.threshold)

        elif self.activation_type == "tanheig":
            return TanhEig(threshold=self.threshold)
        
        elif self.activation_type == "spaeig":
            return SpAEig(threshold=self.threshold)

        elif self.activation_type == "sinh":
            return activationSPD(mode="sinh")

        elif self.activation_type == "cosh":
            return activationSPD(mode="cosh")
        
        elif self.activation_type == "coshP":
            return coshP()

        elif self.activation_type == "sinhP":
            return sinhP()
        
        elif self.activation_type == "polyact":
            return polynomialActivation()

        elif self.activation_type == "expT":
            return expT()

        elif self.activation_type == "expP":
            return expP()

        else:
            raise ValueError("Unknown activation")


    def forward(self, X: torch.Tensor, domain) -> torch.Tensor:
        """Forward pass of the SPDNet model.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor. 

        Returns
        -------
        torch.Tensor
            Output of the classifier, with shape `(batch_size, n_outputs)`.
        """

        block = self.domains_block[domain]
        for layer in block.values():
            X = layer(X)
        X = self.logeig(X)
        X = self.classifier(X)

        return X 
