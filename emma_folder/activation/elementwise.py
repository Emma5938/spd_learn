import torch 
import torch.nn as nn 
import torch.nn.functional as f 
from emma_folder.activation.utils_elementwise import cosh_parametric, sinh_parametric, truncated_exponential, exp_parametric, polynomialFunction


"""
The following activations are element-wise functions. Their custom backwards are in utils_elementwise.py 
"""


class activationSPD(torch.nn.Module):
   
    """ Activation Layer based on the hyperbolic cosine of the entries of the SPD matrix. 
    Reference :  "Non-Linear Activation Functions for Deep Riemannian Neural Networks", Lucas Heck dos Santos et al, 2026
    
    Parameters 
    -------------
    mode : either cosh for the hyperbolic cosine or sinh for hyperbolic sine

    """ 

    def __init__(self, mode = "cosh"):
        super().__init__()
        self.mode = mode

    def forward(self, X): 
        if self.mode == "cosh":
            Y = torch.cosh(X)
        elif self.mode == "sinh" : 
            Y = torch.sinh(X)

        Y = (Y + Y.mT)/2               # To ensure symmetry

        return Y



# Parametric Hyperbolic Cosine - 
class coshP(torch.nn.Module):

    """ Activation Layer based on the parametric hyperbolic cosine of the entries of the SPD matrix. 
    
    Maths 
    -------------
        cosh(alpha * X)

    Parameters 
    -------------
    alpha_init : float
        initial value of the parameter alpha

    """ 

    def __init__(self, autograd = False, alpha_init = -0.43, device = None, dtype = None):
        super().__init__()

        self.register_parameter(                            
            "alpha", 
            nn.Parameter(
                torch.tensor(alpha_init, device = device, dtype = dtype),
            requires_grad = True 
            ),
        )

        self.autograd = autograd 

    def forward(self, X):
        alpha_sp = f.softplus(self.alpha)                                               # Softplus function to ensure the positivity of alpha, initialized to 0.5

        if self.autograd == True : 
            output = torch.cosh(alpha_sp * X)
        else :
            output = cosh_parametric.apply(X, alpha_sp)
        
        return output



# Parametric Hyperbolic Sine - sinh(alpha * X)
class sinhP(torch.nn.Module): 

    """ Activation Layer based on the parametric hyperbolic sine of the entries of the SPD matrix. 
    
    Maths 
    -------------
        sinh(alpha * X)

    Parameters 
    -------------
    alpha_init : float
        initial value of the parameter alpha

    """

    def __init__(self, autograd = False, alpha_init = -0.43, device = None, dtype = None):
        super().__init__()

        self.register_parameter(                            
            "alpha", 
            nn.Parameter(
                torch.tensor(alpha_init, device = device, dtype = dtype),
            requires_grad = True 
            ),
        )

        self.autograd = autograd 

    def forward(self, X):
        alpha_sp = f.softplus(self.alpha)

        if self.autograd == True : 
            output = torch.sinh(self.alpha_sp * X)
        else :
            output = sinh_parametric.apply(X, alpha_sp)

        return output



# Polynomial activation with learnable coefficients 
class polynomialActivation(torch.nn.Module):
    
    """ Activation Layer based on a polynomial with learnable coefficients. 
    
    Maths 
    -------------
        a_0 + a_1*X + a_2 *X^2 + a_3 * X^3 + a_4 * X^4  

    Parameters 
    -------------
    K : int
        order of the polynomial 

    """
    
    def __init__(self, autograd = False, K = 4, device = None, dtype = None):
        super().__init__()

        self.K = K 

        self.w = nn.Parameter(
            torch.zeros(K + 1, device=device, dtype=dtype)
        )

        self.autograd = autograd

    def forward(self, X):
        coeffs = f.softplus(self.w)

        if self.autograd == True : 
            Y = coeffs[0] * torch.ones_like(X)                                              # On créé le premier terme qui est constant 

            H = torch.ones_like(X)                                                          # X^°0

            for k in range(1, self.K + 1):                                                  # on calcule X^1, X^2, X^3, X^4
                H = H * X                                                                   # exclut K +1 
                Y = Y + coeffs[k] * H

        else : 
            Y = polynomialFunction.apply(X, coeffs)

        return Y



# Truncated Exponential - 
class expT(nn.Module):

    """ Activation Layer based on the truncated Taylor expansion of the exponential function. 
    
    Maths 
    -------------
         a_0 + a_1*X + a_2 *X^2 + a_3 * X^3 + a_4 * X^4  with a_k = a^k/k!

    Parameters 
    -------------
    K : int
        order of the polynomial
    
    alpha_init : float 
        initial value of alpha

    """

    def __init__(self, autograd = False, alpha_init = -0.43, K=4, device=None, dtype=None):
        super().__init__()

        self.K = K

        # paramètre libre -> transformé en alpha > 0
        self.alpha = nn.Parameter(torch.tensor(alpha_init, device=device, dtype=dtype))         # -0.43 pour avoir 0.5 après le softplus

        self.autograd = autograd

    def forward(self, X):
        alpha = f.softplus(self.alpha)                                                          # Initialisé à 0.5 

        # Construire les coefficients alpha^k / k!
        coeffs = []
        ak = 1.0

        for k in range(self.K + 1):
            if k > 0:
                ak = ak * alpha / k   
            coeffs.append(ak)

        if self.autograd == True : 
            Y = coeffs[0] * torch.ones_like(X)

            H = torch.ones_like(X)

            for k in range(1, self.K + 1):
                H = H * X
                Y = Y + coeffs[k] * H

        else : 
            Y = truncated_exponential.apply(X, alpha, self.K)

        return Y


# Parametric exponential - exp(alpha *X)
class expP(nn.Module):
    
    """ Activation Layer based on the parametric exponential function. 
    
    Maths 
    -------------
        exp(alpha * X )

    Parameters 
    -------------    
    alphaE_init : float 
        initial value of alpha

    """

    def __init__(self, autograd = False, alphaE_init = -0.43, device=None, dtype=None):
        super().__init__()

        self.alphaE = nn.Parameter(
            torch.tensor(alphaE_init, device=device, dtype=dtype)
        )

        self.autograd = autograd

    def forward(self, X):
        alpha = f.softplus(self.alphaE)
        
        if self.autograd == True : 
            output = torch.exp(alpha * X)
        else :
            output = exp_parametric.apply(X, alpha)
        
        return output
