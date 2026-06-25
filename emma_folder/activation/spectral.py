import torch 
import torch.nn as nn 
import torch.nn.functional as f 

from spd_learn.functional.numerical import get_epsilon
#from spd.learn.functional.core import clamp_eigvals
from spd_learn.functional.autograd import clamp_eigvals_func, modeig_forward
from emma_folder.activation.utils_spectral import power, tanh, SpectralAttention


"""
This script contains the spectral activation functions. Their custom backward are in utils_spectral.py 
"""

class PowerEig(nn.Module):

    """ Activation Layer based on the power "phi" of eigenvalues. 

    max(epsilon, eigenvalue to the power "phi")

    Parameters 
    -------------
    threshold : float 
        The threshold allows the tuning up of the small positive eigenvalues 

    autograd : bool, default=False
        Whether to use the autograd backend.

    phi : float
        Power to raise the eigenvalues to.

    """

    threshold_: torch.Tensor                                                                    # Type Annotation : threshold will be a torch.Tensor

    def __init__(self, threshold = None, autograd = False, phi_init = 1.0, device = None, dtype = None ):
        super().__init__()
        
        self._use_dynamic_threshold = threshold is None                                         # Automatic threshold if threshold = None 
        # will be computed dynamically based on input dtype
        if threshold is None :
            self.register_buffer(
                "threshold_", torch.tensor(0.0, device = device, dtype = dtype)
            )
        else : 
            self.register_buffer(
                "threshold_", torch.tensor(threshold, device = device, dtype = dtype)
            )

        self.register_parameter(                                                                
            "phi", 
            nn.Parameter(
                torch.tensor(phi_init, device = device, dtype = dtype),                         
            requires_grad = True 
            ),
        )

        self.autograd_ = autograd
        

    def _get_threshold(self, X : torch.Tensor) : 
        """Get the threshold for eigenvalue clamping.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor (used for dtype).

        Returns
        -------
        torch.Tensor
            The threshold value.
        """
        
        if self._use_dynamic_threshold : 
            threshold_value = get_epsilon(X.dtype, "eigvalclamp")
            return torch.tensor(threshold_value, device = X.device, dtype = X.dtype)

        return self.threshold_.to(device = X.device, dtype = X.dtype)

    def forward(self, X : torch.Tensor) -> torch.Tensor :
        """Forward pass of the PowerEig layer.

        Parameters
        ----------
        X : torch.Tensor
            Input symmetric matrix.

        Returns
        -------
        torch.Tensor
            The output matrix with rectified eigenvalues.
        """
        threshold = self._get_threshold(X)
        phi_sp = f.softplus(self.phi)                                                                  # Softplus function to ensure the positiveness of phi 

        if self.autograd_ : 
            def applied_fct(eigvals, phi_sp):
                return eigvals.pow(phi_sp).clamp(min = threshold)                    

            output, s, U, s_modified = modeig_forward(X, applied_fct, phi_sp)
            return output

        else :
            output = power.apply(X, phi_sp, threshold)                                                

        output = (output + output.mT) / 2                                                             # Ensures that the output is symmetric 
        return output
 


class SpAEig(nn.Module):
    """ Activation Layer based on spectral attention mechanism. 
    
    This layer introduces non-linearity to the model, while preserving the SPD property. 

    Maths 
    -----

    lambda_tilde = lambda * (1 + softplus(teta_a)*exp(-(i-n*sigmoid(teta_mu))^2 / (2*softplus(teta_sigma)^2))

    Parameters 
    -------------

    autograd : bool, default=False
        Whether to use the autograd backend.

    threshold : float 
        The threshold allows the tuning up of the small positive eigenvalues 

    mu : float
        Mean of the gaussian

    sigma : float 
        Variance of the gaussian

    A : float
        Amplitude of the Gaussian

    """
    threshold_: torch.Tensor                                                                    

    def __init__(self, autograd = False, mu_init = 0.2, sigma_init= 8.0, A_init = 0.55 , threshold = None, device = None, dtype = None ):
        super().__init__()                                                                      

        self._use_dynamic_threshold = threshold is None                                         
        # will be computed dynamically based on input dtype
        if threshold is None :
            self.register_buffer(
                "threshold_", torch.tensor(0.0, device = device, dtype = dtype)
            )
        else : 
            self.register_buffer(
                "threshold_", torch.tensor(threshold, device = device, dtype = dtype)
            )
        
        self.register_parameter(                                                                
            "mu", 
            nn.Parameter(
                torch.tensor(mu_init, device = device, dtype = dtype),                          
            requires_grad = True 
            ),
        )

        self.register_parameter(                                                                
            "sigma", 
            nn.Parameter(
                torch.tensor(sigma_init, device = device, dtype = dtype),                       
            requires_grad = True 
            ),
        )

        self.register_parameter(                                                                
            "A", 
            nn.Parameter(
                torch.tensor(A_init, device = device, dtype = dtype),                           
            requires_grad = True 
            ),
        )

        self.autograd_ = autograd

    def _get_threshold(self, X : torch.Tensor) : 
        """Get the threshold for eigenvalue clamping.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor (used for dtype).

        Returns
        -------
        torch.Tensor
            The threshold value.
        """
        
        if self._use_dynamic_threshold : 
            threshold_value = get_epsilon(X.dtype, "eigvalclamp")
            return torch.tensor(threshold_value, device = X.device, dtype = X.dtype)

        return self.threshold_.to(device = X.device, dtype = X.dtype)

    def forward(self, X : torch.Tensor) -> torch.Tensor :
        """Forward pass of the SpAEig layer.

        Parameters
        ----------
        X : torch.Tensor
            Input symmetric matrix.

        Returns
        -------
        torch.Tensor
            The output matrix with rectified eigenvalues.
        """
        threshold = self._get_threshold(X)
        n = X.shape[-1]
        A_sp = f.softplus(self.A)                                                                
        mu_sig = 1 + (n-1) * torch.sigmoid(self.mu)
        sigma_sp = f.softplus(self.sigma) + 1e-4
        
        if self.autograd_ : 
            def applied_fct(eigvals, A_sp, mu_sig, sigma_sp, threshold):
                
                # Indexes i                                                                     # Construction of indexes i and we put them on the same device as Eigvals
                n = eigvals.shape[-1]                                                           # Number of eigenvalues
                i = torch.arange(1, n+1, device=eigvals.device, dtype=eigvals.dtype)
                i = i.unsqueeze(0).expand_as(eigvals)                                           # unsqueeze(0) adds a dimension i(,3) -> i(1,3) and expand repeat it for each element of the batch

                # Normalized gaussian 
                g = 1.0 + A_sp * torch.exp(- (i - mu_sig)**2 / (2 * sigma_sp**2))
                gn = g / g.mean(dim=1, keepdim=True)

                # Function
                eig_tilde = eigvals * gn
                
                return eig_tilde  # .clamp(min = threshold)                   

            output, s, U, s_modified = modeig_forward(X, applied_fct, A_sp, mu_sig, sigma_sp, threshold)
            return output

        else :
            output = SpectralAttention.apply(X, A_sp, mu_sig, sigma_sp, threshold)                                                

        output = (output + output.mT) / 2
        return output



class TanhEig(nn.Module):
    """ Activation Layer based on the hyperbolic tangente of the eigenvalues. 
    
    This layer introduces non-linearity to the model, while preserving the SPD property. 

    Maths 
    -----
    In the log-space, max(log(epsilon), b tanh(x/a)). 
    In the eigenvalue space, max(epsilon, exp(vu f(log(x)))) avec f(t) = a tanh(t/a)

    Parameters 
    -------------
    threshold : float 
        The threshold allows the tuning up of the small positive eigenvalues 

    autograd : bool, default=False
        Whether to use the autograd backend.

    a : float 
    b : float
    
    """

    threshold_: torch.Tensor                                                                    

    def __init__(self, threshold = None, autograd = False, a_init = 9.0, b_init = 9.2103, device = None, dtype = None ):
        super().__init__()
        
        self._use_dynamic_threshold = threshold is None
        # will be computed dynamically based on input dtype
        if threshold is None :
            self.register_buffer(
                "threshold_", torch.tensor(0.0, device = device, dtype = dtype)
            )
        else : 
            self.register_buffer(
                "threshold_", torch.tensor(threshold, device = device, dtype = dtype)
            )


        self.register_parameter(
            "a", 
            nn.Parameter(
                torch.tensor(a_init, device = device, dtype = dtype),
            requires_grad = True 
            ),
        )

        self.register_parameter(
            "b", 
            nn.Parameter(
                torch.tensor(b_init, device = device, dtype = dtype),
            requires_grad = True 
            ),
        )

        self.autograd_ = autograd
        

    def _get_threshold(self, X : torch.Tensor) : 
        """Get the threshold for eigenvalue clamping.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor (used for dtype).

        Returns
        -------
        torch.Tensor
            The threshold value.
        """
        
        if self._use_dynamic_threshold : 
            threshold_value = get_epsilon(X.dtype, "eigvalclamp")
            return torch.tensor(threshold_value, device = X.device, dtype = X.dtype)

        return self.threshold_.to(device = X.device, dtype = X.dtype)

    def forward(self, X : torch.Tensor) -> torch.Tensor :
        """Forward pass of the TanhEig layer.

        Parameters
        ----------
        X : torch.Tensor
            Input symmetric matrix.

        Returns
        -------
        torch.Tensor
            The output matrix with rectified eigenvalues.
        """
        threshold = self._get_threshold(X)
        b_sp = f.softplus(self.b)
        a_sp = f.softplus(self.a)

        if self.autograd_ : 
            def applied_fct(eigvals, a_sp, b_sp, threshold):
                f = torch.exp(b_sp * torch.tanh(torch.log(eigvals.clamp(min=1e-12))/a_sp))
                return f.clamp(min = threshold)                    

            output, s, U, s_modified = modeig_forward(X, applied_fct, a_sp, b_sp, threshold)
            return output

        else :
            output = tanh.apply(X, a_sp, b_sp, threshold)

        output = (output + output.transpose(-1, -2)) / 2 
        return output
 