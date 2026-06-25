import sys
sys.path.insert(0, r"C:\Users\andrieue\Desktop\PythonPackages")
import torch
import copy
import geoopt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


sys.path.insert(0, r"C:\Users\andrieue\Desktop\Codes\spd_learn")
import spd_learn 
import pickle 
import numpy as np
sys.path.insert(0, r"C:\Users\andrieue\Desktop\PythonPackages")
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import os
import random
import tkinter as t
from tkinter.filedialog import askdirectory, askopenfilename

sys.path.insert(0, r"C:\Users\andrieue\Desktop\Codes\spd_learn\emma_folder\utils")
sys.path.insert(0, r"C:\Users\andrieue\Desktop\Codes\spd_learn\emma_folder\models_")

from emma_folder.utils.eeg_data.data_scripts.get_eeg_data  import DomainBatchSampler 
from emma_folder.models_.model_SPD import modelSPDNet
from emma_folder.utils.monitoring import get_phi_values, get_a_values, get_b_values, get_A_values, get_mu_values, get_sigma_values, EigenvalueMonitor, get_alpha_values, get_w_values, get_alphaE_values


#-----------------------------------------------
# Load the data
#-----------------------------------------------

path = t.filedialog.askdirectory(title="Select the folder containing the data")            # Path to the folder containing the data
print("Path of selected folder : ", path)

list_files = [ file for file in os.listdir(path) if file.endswith(".pkl")]                 # List of .pkl file in the folder


acc_avg = []                                                                               # To stock accuracy of each fold


for i,file in enumerate(list_files) :                                                      # Each fold
    path_data = os.path.join(path, file)
    print(f"Fichier : {file}")

    with open(path_data, 'rb') as f : 
        data = pickle.load(f)

    print("Type des données : ", type(data))                                        # Type of data : dict
    print(sorted(data))                                                             # 3 keys : 'test', 'train', 'val'

    # ---------- TEST --------------
    test_data = data['test']
    X_test, Y_test, D_test = test_data
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.long)
    D_test = torch.tensor(D_test)

    # ---------- TRAIN --------------
    train_data = data['train']

    # 3 elements (tuple): 
    # Indice 0 : (1439, 22, 22)             - SPD Matrices with 22 EEG channels 1439 trials
    # Indice 1 : (1439), valeurs : 0 ou 1   - Classes
    # Indice 2 : (1439), valeur de 0 à 17   - Domains

    X_train, Y_train, D_train = train_data

    unique_domains = np.unique(D_train)
    domains = [f"domain {d}" for d in unique_domains]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    D_train = torch.tensor(D_train)

    # ---------- VAL --------------
    val_data = data['val']                                                          
    X_val, Y_val, D_val = val_data
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.long)
    D_val = torch.tensor(D_val)

    # ---------- Dataloader & Dataset --------------
    batch_size = 32
    
    sampler_train = DomainBatchSampler(D_train, batch_size=batch_size, shuffle=True)                        # Creates batches with unique domains 
    sampler_test = DomainBatchSampler(D_test, batch_size=batch_size, shuffle=False)
    sampler_val = DomainBatchSampler(D_val, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(TensorDataset(X_train, Y_train, D_train), batch_sampler=sampler_train)
    val_loader = DataLoader(TensorDataset(X_val, Y_val, D_val), batch_sampler=sampler_val)
    test_loader = DataLoader(TensorDataset(X_test, Y_test, D_test), batch_sampler=sampler_test)

    #-----------------------------------------------
    # Model  - Architecture 
    #-----------------------------------------------

    n_chans = X_train.shape[1]                                                      # nbr channels
    n_outputs = len(torch.unique(Y_train))                                          # nbr of classes 

    print("\n" + "=" * 60)
    print("Model Architectures ")
    print("=" * 60)

    spdnet = modelSPDNet(
        activation ="polyact",                                                      # !! ACTIVATION !!
        n_chans = n_chans,
        n_outputs = n_outputs,
        threshold = 1e-4,                                      
        domains = domains
    )

    print("\nSPDNet :")
    print(f"Parameters : {sum(p.numel() for p in spdnet.parameters()) :,}")
    print("Architecture : Bimap -> Activation -> LogEig -> Linear")
    print(f"Number of channels = {n_chans}")
    print(f"NUmber of classes = {n_outputs}")
    print("Final feature size:", spdnet.len_last_layer)


    #-----------------------------------------------
    # Training configuration - Hyperparameters 
    #-----------------------------------------------
    max_epochs = 75
    learning_rate = 0.005

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 60}")
    print("Training configuration")
    print("=" * 60)
    print(f"Device : {device}")
    print(f"Batch size : {batch_size}")
    print(f"Max epochs : {max_epochs}")
    print(f"Learning rate : {learning_rate}")

    spdnet.to(device)

    #-----------------------------------------------
    # Loss Function and Optimization
    #-----------------------------------------------
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(spdnet.parameters(), lr = 0.005)
    optimizer = geoopt.optim.RiemannianAdam(spdnet.parameters(), lr = 0.005)


    #-----------------------------------------------
    # Hook ReEig - nombre de valeurs clampées
    #-----------------------------------------------
    ENABLE_CLAMP_MONITORING = spdnet.activation_type == "reeig"                                             # True only if the activation is ReEig

    def reeig_hook(module, input, output):
        if not ENABLE_CLAMP_MONITORING:
            return
        X = input[0]                                                                                        #  SPD Matrices before ReEig - input is a tuple
        eps = 1e-4
        eigvals = torch.linalg.eigvalsh(X.detach())                                                         # Eigenvalue for each matrix of the batch (B, lambda)
        num_clamped = (eigvals < eps).sum(dim=-1)                                                           # Number of clamped values for each matrix of the batch // sum(dim=-1) replace booleans by values
        total = eigvals.size(-1)                                                                            # Total number of eigenvalues in a matrix 
        
        batch_num_clamped = num_clamped.sum().item()                                                        # Sum on the batch
        batch_total = total * X.size(0)                                                                     # Total on the batch

        if not hasattr(module, 'list_num_clamped'):
            module.list_num_clamped = []
            module.list_total = []

        module.list_num_clamped.append(batch_num_clamped)
        module.list_total.append(batch_total)
        
    reeig_layers = []
    if ENABLE_CLAMP_MONITORING:
        for domain_name, domain_block in spdnet.domains_block.items():
            for act_name in ["activation1"]:            #, "activation2"
                layer = domain_block[act_name]
                layer.layer_id = f"{domain_name}-{act_name}"   # Unique id 
                layer.list_num_clamped = []
                layer.list_total = []
                layer.register_forward_hook(reeig_hook)
                reeig_layers.append(layer)



    #-----------------------------------------------
    # HOOK BIMAP TO LOG EIGENVALUES
    #-----------------------------------------------
    monitor = EigenvalueMonitor()
    monitor.attach(spdnet)



    #-----------------------------------------------
    # Parameters monitoring - To store the parameters as the epochs progress
    #-----------------------------------------------
    phi_monitoring = {}                                             # PowerEig
    a_monitoring = {}                                               # TanhEig
    b_monitoring = {}                                               # TanhEig
    A_monitoring = {}                                               # SpAEig
    mu_monitoring = {}                                              # SpAEig
    sigma_monitoring = {}                                           # SpAEig 
    alpha_monitoring = {}                                           # cosh P
    w_monitoring = {}                                               # expT
    alphaE_monitoring = {}                                          # expP

    #-----------------------------------------------
    # Training the model SDPNet
    #-----------------------------------------------
    train_losses = []                                                                                   # To stock the loss 
    val_losses = []
    best_loss = float("inf")                                                                            # Early stopping parameters
    patience, wait = 10, 0
    val_acc = []

    for epoch in range(max_epochs) :                                                                    # Go through the epochs
        # ---------
        # TRAINING
        # ---------
        monitor.reset()                                                                                 # Bimap eigenvalues monitoring reset to 0
        spdnet.train()                                                                                  # The model is in training mode
        train_loss = 0                                                                                  # Variable for accumulating the loss across all batches 

        for x,y,d in train_loader :                                                                     # We iterate through the batches (x is of size (batch_size, 22, 22), y is of size (batch_size), and d is of size (batch_size))
            assert torch.all(d == d[0]), "Batch contains multiple domains!"
            x = x.to(device)                                                                            # To ensure that the data and the model are on the same device
            y = y.to(device)
            d = d.to(device)

            domain_name = f"domain {d[0].item()}"
            pred = spdnet(x, domain_name)                                                               # Forward : batch x run through SPDNet
            loss = criterion(pred, y)                                                                   # Calculating the loss between predictions and labels

            optimizer.zero_grad()                                                                       # Reset gradients to 0 (otherwise the gradients from previous batches would be added together)
            loss.backward()                                                                             # Calculate the gradient of the loss with respect to each model parameter
            
            #torch.nn.utils.clip_grad_norm_(spdnet.parameters(), 1)                                     # Set the gradient limit to 1
            optimizer.step()                                                                            # Updating model weights based on gradients 
            train_loss += loss.item()                                                                   # Accumulates the batch loss 

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
       
        # ----------
        # VALIDATION
        # ----------
        spdnet.eval()                                                                                   # The model is in evaluation mode
        val_loss = 0                                                                                    # Variable for accumulating the loss across all batches
        correct = 0                                                                                     # Number of correct predictions
        total = 0                                                                                       # Total number of examples viewed
        all_val_preds = []                                                                              # Predictions for balanced accuracy
        all_val_labels = []                                                                             # all labels for balanced accuracy

        
        # -------------------------- Monitoring -------------------------------
        phi_values = get_phi_values(spdnet)

        a_values = get_a_values(spdnet)
        b_values = get_b_values(spdnet)

        A_values = get_A_values(spdnet)
        mu_values = get_mu_values(spdnet)
        sigma_values = get_sigma_values(spdnet)

        Alpha_values = get_alpha_values(spdnet)

        w_values = get_w_values(spdnet)

        alphaE_values = get_alphaE_values(spdnet)

        for key, value in phi_values.items():
            phi_monitoring.setdefault(key, []).append(value)

        for key, value in a_values.items():
            a_monitoring.setdefault(key, []).append(value)

        for key, value in b_values.items():
            b_monitoring.setdefault(key, []).append(value)

        for key, value in A_values.items():
            A_monitoring.setdefault(key, []).append(value)

        for key, value in mu_values.items():
            mu_monitoring.setdefault(key, []).append(value*(n_chans-1)+1)

        for key, value in sigma_values.items():
            sigma_monitoring.setdefault(key, []).append(value)

        for key, value in Alpha_values.items():
            alpha_monitoring.setdefault(key, []).append(value)

        for key, value in w_values.items():
            w_monitoring.setdefault(key, []).append(value)
        
        for key, value in alphaE_values.items():
            alphaE_monitoring.setdefault(key, []).append(value)

        stats = monitor.summarize()
        

        # ------------------------- Validation -----------------------------------
        with torch.no_grad():                                                                           # Gradient calculation is disabled during validation
            for x,y,d in val_loader :                                                                   # We go through the validation batches
                x = x.to(device)
                y = y.to(device)
                d = d.to(device)

                domain_name = f"domain {d[0].item()}"
                pred = spdnet(x, domain_name)
                loss = criterion(pred, y)
                val_loss += loss.item()

                predictions = pred.argmax(dim=1)                                                        # We take the most likely class for each example
                all_val_preds.append(predictions.cpu())
                all_val_labels.append(y.cpu())
                correct += (predictions == y).sum().item()                                              # Counts the number of correct predictions for the batch (where `predictions == y` is a boolean value for each correct prediction)
                total += y.size(0)                                                                      # Number of examples in the batch

        val_loss /= len(val_loader)                                                                     # Average loss over the valid set
        val_accuracy = correct / total                                                                  # Proportion of correct predictions on validation
        val_losses.append(val_loss)

        predictions_val = torch.cat(all_val_preds)                                                      # To concatenate tensors along a dimension 
        Y_val_full = torch.cat(all_val_labels)
        val_bal_acc = balanced_accuracy_score(Y_val_full.numpy(),predictions_val.numpy())
        val_acc.append(val_bal_acc)

        
        print(f"Epoch {epoch+1}/{max_epochs} | Train loss : {train_loss:.3f} | Val loss : {val_loss:.3f} | Val accuracy : {val_accuracy:.3f}")
        

        # -----------------------------------------------------
        # Print the specific values for the Bimap layer output and the mu 
        for layer, values in stats.items():
            if not layer.startswith("domain 0"):
                continue

            print( f"Epoch {epoch+1} | {layer} | min={values['min']:.4f} | median={values['median']:.4f} | max={values['max']:.4f} | mean={values['mean']:.4f}")

        first_domain = list(spdnet.domains_block.keys())[0]

        # for key, values in mu_mag_monitoring.items():

        #     if not key.startswith(first_domain):
        #         continue

        #     # last value of mu
        #     if isinstance(values, list):
        #         mu = values[-1]

        #         if isinstance(mu, torch.Tensor):
        #             mu = mu.item()

        #     elif isinstance(values, torch.Tensor):
        #         mu = values[-1].item()

        #     else:
        #         mu = values[-1]

        #     print(f"{key} | mu = {mu:.4f}")

        # -------------------------EARLY STOPPING ------------------------------------
        min_delta = 1e-3                                                                                #minimum improvement considered to be ‘real’
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            wait = 0
            best_model_state = copy.deepcopy(spdnet.state_dict())                                        # Save the best model
            #best_model_state = spdnet.state_dict()                                                      # Save the best model
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                spdnet.load_state_dict(best_model_state)                                                # Load the best model

                break


        # ----------------------- HOOK ---------------------------
        for layer in reeig_layers:
            num_clamped_epoch = sum(layer.list_num_clamped)                                             # Sum across all batches
            total_epoch = sum(layer.list_total)
            print(f"Layer {layer.layer_id} : valeurs clampées / total = {num_clamped_epoch}/{total_epoch}")
            # We are resetting the lists for the next epoch
            layer.list_num_clamped = []
            layer.list_total = []


    # -----
    # TEST 
    # -----
    spdnet.eval()
    correct = 0
    total = 0

    all_pred = []                                                                                       # We store the predictions from all batches
    all_label = []                                                                                      # Store all labels

    with torch.no_grad() :
        for x,y,d in test_loader : 
            x = x.to(device)
            y = y.to(device)
            d = d.to(device)

            domain_name = f"domain {d[0].item()}"
            pred_test = spdnet(x, domain_name)
            predictions_test = pred_test.argmax(dim=1)

            correct += (predictions_test == y).sum().item()
            total += y.size(0)

            all_pred.append(predictions_test.cpu())
            all_label.append(y.cpu())

    predictions_test = torch.cat(all_pred)                                                              # To concatenate tensors along a dimension 
    Y_test_full = torch.cat(all_label)

    test_accuracy = correct / total
    print(f"\nTest Accuracy  {test_accuracy:.3f}")


    test_balanced_accuracy = balanced_accuracy_score(Y_test_full.numpy(),predictions_test.numpy())
    print(f"\nTest Balanced Accuracy {test_balanced_accuracy:.3f}")

    acc_avg.append(test_balanced_accuracy)





   #-----------------------------------------------
   # Visualization 
   #-----------------------------------------------

    # ----- Training & Validation Loss Curves/Accuracy -----
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label = "Train Loss")
    plt.plot(val_losses, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(val_acc, label = "Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.title("Validation Accuracy during Training")
    plt.show()

    
    
    #-------------------------------------------
    # ------ Monitoring of phi - POWEREIG-------
    #-------------------------------------------
    # tempo - ----
    # for key, values in phi_monitoring.items():
    #     print(f"Type des valeurs pour {key}: {type(values)}")
    #     if isinstance(values, list):
    #         print(f"Types des éléments dans la liste pour {key}: {[type(v) for v in values]}")

    # tempo ----
    ENABLE_PHI_MONITORING = spdnet.activation_type == "powereig" 
    if ENABLE_PHI_MONITORING : 
        plt.figure(figsize=(8,5))

        for key, values in phi_monitoring.items():
            # Cas 1 : tensor PyTorch
            #if isinstance(values, torch.Tensor):
                #y = values.detach().cpu().numpy()
            # Cas 2 : liste de tensors PyTorch
            if isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values):
                y = np.array([v.detach().cpu().numpy() for v in values])
            # Cas 3 : liste ou numpy array
            else:
                #y = np.array(values)
                raise ValueError("PB Monitoring phi")
            plt.plot(y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("phi")
        plt.title("Evolution de phi pendant l'entraînement")
        plt.legend()
        plt.grid(True)

        plt.show()



    #-------------------------------------------
    # ------ Monitoring of a - TANHEIG -------
    #-------------------------------------------
    ENABLE_A_B_MONITORING = spdnet.activation_type == "tanheig" 
    if ENABLE_A_B_MONITORING :
        plt.figure(figsize=(8,5))

        for key, values in a_monitoring.items():
            # Cas 1 : tensor PyTorch
            if isinstance(values, torch.Tensor):
                y = values.detach().cpu().numpy()
            # Cas 2 : liste de tensors PyTorch
            elif isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values):
                y = np.array([v.detach().cpu().numpy() for v in values])
            # Cas 3 : liste ou numpy array
            else:
                y = np.array(values)
    
            plt.plot(y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("a")
        plt.title("Evolution de a pendant l'entraînement")
        plt.legend()
        plt.grid(True)

        plt.show()
        
        # ------ Monitoring of b - TANHEIG-------
        plt.figure(figsize=(8,5))

        for key, values in b_monitoring.items():
            # Cas 1 : tensor PyTorch
            if isinstance(values, torch.Tensor):
                y = values.detach().cpu().numpy()
            # Cas 2 : liste de tensors PyTorch
            elif isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values):
                y = np.array([v.detach().cpu().numpy() for v in values])
            # Cas 3 : liste ou numpy array
            else:
                y = np.array(values)
    
            plt.plot(y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("b")
        plt.title("Evolution de b pendant l'entraînement")
        plt.legend()
        plt.grid(True)

        plt.show()



    #-------------------------------------------
    # ------ Monitoring of A, mu, sigma - SPAEIG -------
    #-------------------------------------------
    ENABLE_A_MU_SIGMA_MONITORING = spdnet.activation_type == "spaeig" 
    if ENABLE_A_MU_SIGMA_MONITORING :
        plt.figure(figsize=(8,5))

        for key, values in A_monitoring.items():
            # Cas 1 : tensor PyTorch
            if isinstance(values, torch.Tensor):
                y = values.detach().cpu().numpy()
            # Cas 2 : liste de tensors PyTorch
            elif isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values):
                y = np.array([v.detach().cpu().numpy() for v in values])
            # Cas 3 : liste ou numpy array
            else:
                y = np.array(values)
    
            plt.plot(y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("A")
        plt.title("Evolution de A pendant l'entraînement")
        plt.legend()
        plt.grid(True)

        plt.show()
        
        # ------ Monitoring of mu - SPAEIG-------
        plt.figure(figsize=(8,5))

        for key, values in mu_monitoring.items():
            # Cas 1 : tensor PyTorch
            if isinstance(values, torch.Tensor):
                y = values.detach().cpu().numpy()
            # Cas 2 : liste de tensors PyTorch
            elif isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values):
                y = np.array([v.detach().cpu().numpy() for v in values])
            # Cas 3 : liste ou numpy array
            else:
                y = np.array(values)
    
            plt.plot(y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Mu")
        plt.title("Evolution de mu pendant l'entraînement")
        plt.legend()
        plt.grid(True)

        plt.show()

        # ------ Monitoring of sigma - SPAEIG-------
        plt.figure(figsize=(8,5))

        for key, values in sigma_monitoring.items():
            # Cas 1 : tensor PyTorch
            if isinstance(values, torch.Tensor):
                y = values.detach().cpu().numpy()
            # Cas 2 : liste de tensors PyTorch
            elif isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values):
                y = np.array([v.detach().cpu().numpy() for v in values])
            # Cas 3 : liste ou numpy array
            else:
                y = np.array(values)
    
            plt.plot(y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Sigma")
        plt.title("Evolution de Sigma pendant l'entraînement")
        plt.legend()
        plt.grid(True)

        plt.show()



    #-------------------------------------------
    # ------ Monitoring of alpha - COSHP -------
    #-------------------------------------------
    ENABLE_ALPHA_MONITORING = spdnet.activation_type in ("coshP", "sinhP", "expT")
    if ENABLE_ALPHA_MONITORING :
        plt.figure(figsize=(8,5))

        for key, values in alpha_monitoring.items():
            # Cas 1 : tensor PyTorch
            if isinstance(values, torch.Tensor):
                y = values.detach().cpu().numpy()
            # Cas 2 : liste de tensors PyTorch
            elif isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values):
                y = np.array([v.detach().cpu().numpy() for v in values])
            # Cas 3 : liste ou numpy array
            else:
                y = np.array(values)
    
            plt.plot(y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Alpha")
        plt.title("Evolution de Alpha pendant l'entraînement")
        plt.legend()
        plt.grid(True)

        plt.show()


    #-------------------------------------------
    # ------ Monitoring of w - POLYNOMIAL ACTIVATION  -------
    #-------------------------------------------
    ENABLE_W_MONITORING = spdnet.activation_type == "polyact" 
    if ENABLE_W_MONITORING :

        plt.figure(figsize=(8,5))

        for key, values in w_monitoring.items():
            # Cas 1 : tensor PyTorch
            if isinstance(values, torch.Tensor):
                y = values.detach().cpu().numpy()
            # Cas 2 : liste de tensors PyTorch
            elif isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values):
                y = np.array([v.detach().cpu().numpy() for v in values])
            # Cas 3 : liste ou numpy array
            else:
                y = np.array(values)
    
            plt.plot(y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Alpha")
        plt.title("Evolution de Alpha pendant l'entraînement")
        plt.legend()
        plt.grid(True)

        plt.show()
        # domain_key = "domain 0_activation1"
        # w_history = np.array([t.detach().cpu().numpy() for t in w_monitoring[domain_key]])  # shape (n_epochs, n_chans)

        # plt.figure(figsize=(10, 5))
        # plt.imshow(w_history.T, aspect='auto', cmap='RdBu_r', vmin=0.5, vmax=1.5)
        # plt.colorbar(label='wᵢ')
        # plt.xlabel('Epoch')
        # plt.ylabel('X^°k')
        # plt.title('Evolution du vecteur w pendant l\'entraînement')
        # plt.show()

        # # vecteur final appris 
        # w_final = w_history[-1]  # dernière epoch

        # plt.figure(figsize=(8, 4))
        # plt.bar(range(1, len(w_final)+1), w_final)
        # plt.axhline(y=1.0, linestyle='--', color='red', label='identité (d=1)')
        # plt.xlabel('wᵢ+1')
        # plt.title(f'Coefficients de la suite appris - {domain_key}')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    #-------------------------------------------
    # ------ Monitoring of a - EXPP  -------
    #-------------------------------------------
    ENABLE_ALPHAE_MONITORING = spdnet.activation_type == "expP" 
    if ENABLE_ALPHAE_MONITORING :

        plt.figure(figsize=(8,5))

        for key, values in alphaE_monitoring.items():
            # Cas 1 : tensor PyTorch
            if isinstance(values, torch.Tensor):
                y = values.detach().cpu().numpy()
            # Cas 2 : liste de tensors PyTorch
            elif isinstance(values, list) and all(isinstance(v, torch.Tensor) for v in values):
                y = np.array([v.detach().cpu().numpy() for v in values])
            # Cas 3 : liste ou numpy array
            else:
                y = np.array(values)
    
            plt.plot(y, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Alpha")
        plt.title("Evolution de a pendant l'entraînement")
        plt.legend()
        plt.grid(True)

        plt.show()

    

mean = np.mean(acc_avg)
std = np.std(acc_avg)

print("\n" + "="*50)
print(f"FINAL RESULT: {mean:.4f} ± {std:.4f}")
print("="*50)

