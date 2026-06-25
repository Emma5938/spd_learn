import torch
from torch.autograd import gradcheck
import sys 
sys.path.insert(0, r"C:\Users\andrieue\Desktop\Codes\spd_learn\emma_folder\activation")    # à changer
from utils_elementwise import exp_parametric

"""
Test backward of exp_parametric (SPD Hadamard exponential activation)

1. Générer une matrice SPD random
2. Forward pass via Function
3. Gradcheck (finite differences)
4. Vérifier gradients wrt X et alpha
"""

# -------------------------
# SPD matrix generator
# -------------------------
def random_matrix(batch, n, device="cuda", dtype=torch.float64):
    A = torch.randn(batch, n, n, device=device, dtype=dtype)
    X = A @ A.transpose(-1, -2)
    I = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
    return X + 1e-3 * I


# -------------------------
# Exponential for gradcheck
# -------------------------
def exp_layer(X, a):
    return exp_parametric.apply(X, a)


# -------------------------
# TEST
# -------------------------
def test_backward_exp_parametric():

    device = "cuda"
    dtype = torch.float64                                           # Required for gradcheck stability

    # 1. SPD input
    X = random_matrix(2, 4, device=device, dtype=dtype)
    X.requires_grad_(True)

    # 2. parameter alpha
    a = torch.tensor(0.8, device=device, dtype=dtype, requires_grad=True)

    # -------------------------
    # symmetrized version
    # -------------------------
    def symmetric_exp(X_in, a_in):
        X_sym = (X_in + X_in.transpose(-1, -2)) / 2
        return exp_parametric.apply(X_sym, a_in)

    # -------------------------
    # gradcheck
    # -------------------------
    test = gradcheck(
        symmetric_exp,
        (X, a),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3
    )

    print("Gradcheck result:", test)

    assert test, "Gradcheck FAILED"

    print("exp_parametric backward is correct (X, a)")


# -------------------------
if __name__ == "__main__":
    test_backward_exp_parametric()