import torch
from torch.autograd import gradcheck
import sys
sys.path.insert(0, r"C:\Users\andrieue\Desktop\Codes\spd_learn\emma_folder\activation")
from utils_elementwise import cosh_parametric

"""
Test the backward of the layer

1. Générer une matrice SPD random, puis forward
2. Générer un gradient random de la couche suivante dL/dz
3. Backward : dL/dx et dL/da
4. Vérifier avec gradcheck (finite differences)
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
# Test backward
# -------------------------
def test_backward_cosh_parametric(
    n_matrices=2,
    n_dimension=5,
    a_val=1.5,
    device="cuda",
):

    dtype = torch.float64

    # 1. SPD input
    X = random_matrix(n_matrices, n_dimension, device=device, dtype=dtype)
    X.requires_grad_(True)

    # 2. parameter a
    a = torch.tensor(a_val, device=device, dtype=dtype, requires_grad=True)

    # -------------------------
    # function for gradcheck
    # -------------------------
    def cosh_layer(X_in, a_in):
        return cosh_parametric.apply(X_in, a_in)

    # -------------------------
    # enforce symmetry
    # -------------------------
    def symmetric_cosh(X_in, a_in):
        X_sym = (X_in + X_in.transpose(-1, -2)) / 2
        return cosh_parametric.apply(X_sym, a_in)

    # -------------------------
    # gradcheck
    # -------------------------
    test_passed = gradcheck(
        symmetric_cosh,
        (X, a),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3
    )

    if test_passed:
        print("Cosh_parametric backward correct (X, a)")
        print("Backward test réussi !")
    else:
        print("Échec du gradcheck")


# -------------------------
if __name__ == "__main__":
    test_backward_cosh_parametric()