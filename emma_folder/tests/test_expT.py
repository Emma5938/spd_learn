import torch
from torch.autograd import gradcheck
import sys 
sys.path.insert(0, r"C:\Users\andrieue\Desktop\Codes\spd_learn\emma_folder\activation")
from utils_elementwise import truncated_exponential

"""
Test the backward of the layer

1. Générer une matrice SPD random, puis forward
2. Générer un gradient random de la couche suivante dL/dz
3. Backward : dL/dx et dL/dalpha
4. Vérifier avec gradcheck (finite differences)
"""

# -------------------------
# SPD generator
# -------------------------
def random_matrix(batch, n, device="cuda", dtype=torch.float64):
    A = torch.randn(batch, n, n, device=device, dtype=dtype)
    X = A @ A.transpose(-1, -2)
    I = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
    return X + 1e-3 * I


# -------------------------
# function for gradcheck
# -------------------------
def expT_func(X, alpha, K=4):

    coeffs = []
    ak = 1.0

    Y = torch.zeros_like(X)
    H = torch.ones_like(X)

    for k in range(K + 1):
        if k > 0:
            ak = ak * alpha / k
        coeffs.append(ak)

    for k in range(K + 1):
        if k > 0:
            H = H * X
        Y = Y + coeffs[k] * H

    return Y


# -------------------------
# TEST
# -------------------------
def test_backward_expT():

    device = "cuda"
    dtype = torch.float64

    X = random_matrix(2, 4, device=device, dtype=dtype)
    X.requires_grad_(True)

    alpha = torch.tensor(0.8, device=device, dtype=dtype, requires_grad=True)

    K = 4

    def func(X, alpha):
        X_sym = (X + X.transpose(-1, -2)) / 2
        return expT_func(X_sym, alpha, K)

    # -------------------------
    # gradcheck
    # -------------------------
    test = gradcheck(
        func,
        (X, alpha),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3
    )

    print("Gradcheck result:", test)
    assert test, "Gradcheck FAILED"

    print("Backward correct for truncated exponential activation")


# -------------------------
if __name__ == "__main__":
    test_backward_expT()