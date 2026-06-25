import torch
from torch.autograd import Function
from spd_learn.functional.autograd import modeig_backward, modeig_forward
from spd_learn.functional.numerical import get_epsilon, numerical_config



class power(Function):
    """ Raise the eigenvalues ​​to the power of exponent (phi).

    This fonction computes the power of the eigenvalues vie Eigen Value Decomposition (EVD).

    Parameters 
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    Exponent : 
        Exponent to raise the eigenvalues to. 

    threshold : float
        Threshold for numerical stability.

    Returns
    -------
    torch.Tensor
        The matrix with powered eigenvalues
        
    """

    @staticmethod
    def applied_fct(s, exponent, threshold):
        return s.pow(exponent).clamp(min = threshold)                       # raise to the power phi, then clamp with a threshold 

    @staticmethod
    def derivative(s, exponent, threshold):                                 
        s_pow = s.pow(exponent)

        s_deriv = exponent * s.pow(exponent - 1.0) 
        # pick subgradient 0 for clamped eigenvalues 
        s_deriv[s_pow <= threshold] = 0
        return s_deriv 

    @staticmethod
    def forward(ctx, X, exponent, threshold):
        output, s, U, s_modified = modeig_forward(X, power.applied_fct, exponent, threshold)
        ctx.save_for_backward(s, U, s_modified)
        ctx.exponent = exponent
        ctx.threshold = threshold
        return output

    @staticmethod
    def backward(ctx, grad_output):                                                                 # Must return a gradient for each argument of the forward (except ctx) : dL/dX et dL/dphi, and none for the threshold
        s, U, s_modified = ctx.saved_tensors
        exponent = ctx.exponent
        threshold = ctx.threshold

        # Gradient wrt X
        grad_X = modeig_backward(grad_output, s, U, s_modified, power.derivative, exponent, threshold)         
        
        # Gradient wtr the exponent
        G = U.transpose(-1,-2) @ grad_output @ U 
        s_pow = s.pow(exponent)

        mask = s_pow > threshold
        grad_exponent = torch.sum(G.diagonal(dim1 = -2, dim2= -1) * (s.pow(exponent) * torch.log(s.clamp(min=1e-12))) * mask)

        return grad_X , grad_exponent, None



class tanh(Function):
    """Activation Layer based on the hyperbolic tangente of the eigenvalues. 
    
     Eigenvalues are mapped to the log-domain, passed through a scaled tanh non-linearity, then mapped back via the exponential.

    Parameters 
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    a : 
        Non-linearity form

    b : 
        global intensity

    threshold : float
        Threshold for numerical stability.

    Returns
    -------
    torch.Tensor
        The matrix with powered eigenvalues
        
    """

    @staticmethod
    def applied_fct(s, a, b, threshold):                                   # log-domain, passed through a scaled tanh non-linearity, then mapped back via the exponential
        f = torch.exp(b * torch.tanh(torch.log(s.clamp(min=1e-12))/a))
        return f.clamp(min = threshold)                       

    @staticmethod
    def derivative(s, a, b, threshold):                                 
        p1 = b/(s.clamp(min=1e-12)*a)
        p2 = 1/(torch.cosh(torch.log(s.clamp(min=1e-12))/a)**2)
        p3 = torch.exp(b * torch.tanh(torch.log(s.clamp(min=1e-12))/a)) 
        s_deriv = p1*p2*p3
        # pick subgradient 0 for clamped eigenvalues 
        s_deriv[p3 <= threshold] = 0
        return s_deriv 

    @staticmethod
    def forward(ctx, X, a, b, threshold):
        output, s, U, s_modified = modeig_forward(X, tanh.applied_fct, a, b, threshold)
        ctx.save_for_backward(s, U, s_modified)
        ctx.a = a
        ctx.b = b
        ctx.threshold = threshold
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        a = ctx.a
        b = ctx.b
        threshold = ctx.threshold

        # Calcul optimization 
        log_s = torch.log(s.clamp(min=1e-12))
        u = log_s / a
        t = torch.tanh(u)
        ft = torch.exp(b * t)

        mask = s_modified > threshold                                                                   # Boolean                                  

        G = U.transpose(-1,-2) @ grad_output @ U 

        # Gradient wrt X
        grad_X = modeig_backward(grad_output, s, U, s_modified, tanh.derivative, a, b, threshold)


        # Gradient wrt a 
        a_g = - b * (1/(torch.cosh(u)**2)) * (log_s/(a**2)) * ft
        grad_a = torch.sum(G.diagonal(dim1 = -2, dim2= -1) * a_g *mask)

        # Gradient wrt b 
        b_g = t * ft 
        grad_b = torch.sum(G.diagonal(dim1 = -2, dim2= -1) * b_g*mask)

        return grad_X ,grad_a, grad_b, None




class SpectralAttention(Function):
    """ Raise the eigenvalues ​​to the power of exponent (phi).

     Eigenvalues are mapped to the log-domain, moduled with a gaussian, then mapped back via the exponential.

    Parameters 
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    A : 
        Amplitude of the Gaussian
    
    mu : 
        Mean of the gaussian

    sigma : float
        Variance of the Gaussian

    Returns
    -------
    torch.Tensor
        The matrix with rectified eigenvalues
        
    """

    @staticmethod
    def applied_fct(s, A, mu, sigma, threshold):
        n = s.shape[-1]
        i = torch.arange(1, n+1, device= s.device, dtype= s.dtype)
        i = i.unsqueeze(0).expand_as(s)

        # Normalized Gaussian 
        g = 1.0 + A * torch.exp(- ((i - mu)**2) / (2 * sigma**2))
        gn = g / g.mean(dim=1, keepdim=True)

        # Function 
        s_tilde = s * gn

        return s_tilde #.clamp(min=threshold)                   

    @staticmethod
    def derivative(s, A, mu, sigma, threshold):
        # Indexes
        n = s.shape[-1]
        i = torch.arange(1, n+1, device= s.device, dtype= s.dtype)
        i = i.unsqueeze(0).expand_as(s)                                                  

        # Normalized gaussian
        g = 1.0 + A * torch.exp(- ((i - mu)**2) / (2 * sigma**2))
        gn = g / g.mean(dim=1, keepdim=True)  

        # Derivative
        s_deriv = gn 

        return s_deriv 

    @staticmethod
    def forward(ctx, X, A, mu, sigma, threshold):
        output, s, U, s_modified = modeig_forward(X, SpectralAttention.applied_fct, A, mu, sigma, threshold)
        ctx.save_for_backward(s, U, s_modified)
        ctx.A = A
        ctx.mu = mu
        ctx.sigma = sigma
        ctx.threshold = threshold
       
        return output

    @staticmethod
    def backward(ctx, grad_output):                                                         
        s, U, s_modified = ctx.saved_tensors
        A, mu, sigma = ctx.A, ctx.mu, ctx.sigma
        threshold = ctx.threshold

        # Indexes
        n = s.shape[-1]
        i = torch.arange(1, n+1, device= s.device, dtype= s.dtype)
        i = i.unsqueeze(0).expand_as(s)                                                  
        
        # Projection 
        G = U.transpose(-1,-2) @ grad_output @ U 
        diag_G = torch.diagonal(G, dim1 = -2, dim2= -1)



        # ------------- Notations ------------------------------
        # Normalized Gaussian
        Ei = torch.exp(- ((i - mu)**2) / (2 * sigma**2))
        g = 1.0 + A * Ei
        g_mean = g.mean(dim=1, keepdim=True) 
        gn = g / g_mean
        

        # Useful means
        E_mean = Ei.mean(dim=1, keepdim = True)
        E_mu = torch.mean((i-mu)*Ei, dim=1, keepdim = True)
        E_sigma = torch.mean(((i-mu)**2)*Ei, dim=1, keepdim = True)  

        #-------------- GRADIENTS -------------------------------
        # Gradient wrt X
        grad_X = modeig_backward(grad_output, s, U, s_modified, SpectralAttention.derivative, A, mu, sigma, threshold)

        # Gradient wrt A 
        A_g = s * (Ei/g_mean - (g/g_mean**2)*E_mean)
        grad_a = torch.sum( diag_G * A_g )      #* mask 

        # Gradient wrt mu
        mu_g = s * (A/sigma**2) * ((i-mu)*Ei/g_mean - g/g_mean**2 * E_mu)
        grad_mu = torch.sum(diag_G * mu_g )     #* mask

        # Gradient wrt sigma
        sigma_g = s * (A/sigma**3) * (((i-mu)**2)*Ei/g_mean - g/g_mean**2 * E_sigma )
        grad_sigma = torch.sum(diag_G * sigma_g ) #* mask

        return grad_X ,grad_a, grad_mu, grad_sigma, None

   