import sys
sys.path.insert(0, r"C:\Users\andrieue\Desktop\PythonPackages")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, r"C:\Users\andrieue\Desktop\Codes\spd_learn")
import spd_learn 
import geoopt
import pickle 
import numpy as np
sys.path.insert(0, r"C:\Users\andrieue\Desktop\PythonPackages")
import matplotlib.pyplot as plt 
from sklearn.metrics import balanced_accuracy_score


import os
import random
import copy
import tkinter as t
from tkinter.filedialog import askdirectory, askopenfilename

sys.path.insert(0, r"C:\Users\andrieue\Desktop\Codes\spd_learn\emma_folder\utils")
sys.path.insert(0, r"C:\Users\andrieue\Desktop\Codes\spd_learn\emma_folder\models_")

from emma_folder.utils.eeg_data.data_scripts.get_eeg_data  import DomainBatchSampler 
from emma_folder.models_.model_SPD import model_SPDNet



#-----------------------------------------------
# Reproductibility - def set_seed()
#-----------------------------------------------
def set_seed(seed : int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

#-----------------------------------------------
# Path to the folds of a dataset 
#-----------------------------------------------
path = t.filedialog.askdirectory(title="Select the folder containing the data")            # Path to the folder containing the data
print("Path of selected folder : ", path)

list_files = [ file for file in os.listdir(path) if file.endswith(".pkl")]


#-----------------------------------------------
# Seed 
#-----------------------------------------------
seeds = [1,2,3,4,5]
res_seed = {}
for seed in seeds :
    print(f"\n===== SEED {seed} =====")

    res_fold = {}
    
    for i,file in enumerate(list_files) :
        
        path_data = os.path.join(path, file)
        print(f"Fichier : {file}")

        #-----------------------
        # Load data
        #-----------------------
        with open(path_data, 'rb') as f : 
            data = pickle.load(f) 

        # ---------- TEST --------------
        test_data = data['test']
        X_test, Y_test, D_test = test_data
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.long)
        D_test = torch.tensor(D_test)

        # ---------- TRAIN --------------
        train_data = data['train']
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
    
        sampler_train = DomainBatchSampler(D_train, batch_size=batch_size, shuffle=True)      # Creates batches with unique domains 
        sampler_test = DomainBatchSampler(D_test, batch_size=batch_size, shuffle=False)
        sampler_val = DomainBatchSampler(D_val, batch_size=batch_size, shuffle=False)

        train_loader = DataLoader(TensorDataset(X_train, Y_train, D_train), batch_sampler=sampler_train)
        val_loader = DataLoader(TensorDataset(X_val, Y_val, D_val), batch_sampler=sampler_val)
        test_loader = DataLoader(TensorDataset(X_test, Y_test, D_test), batch_sampler=sampler_test)

        res_couche = []

        for layer in ["reeig", "cosh", "coshP", "expT", "expP", "tanheig"]:
            set_seed(seed)
            n_chans = X_train.shape[1]
            n_outputs = len(torch.unique(Y_train))
            
            #-----------------------
            # Model 
            #-----------------------
            spdnet = modelSPDNet(
                activation = layer,
                n_chans = n_chans,
                n_outputs = n_outputs,
                threshold = 1e-4,
                domains = domains
            )

            #----------- Training configuration ------------
            max_epochs = 75
            learning_rate = 0.005
            device = "cuda" if torch.cuda.is_available() else "cpu"

            spdnet.to(device)

            #----------- Loss and Optimization -------------
            criterion = nn.CrossEntropyLoss()
            # optimizer = optim.Adam(spdnet.parameters(), lr = 0.003)
            optimizer = geoopt.optim.RiemannianAdam(spdnet.parameters(), lr = 0.005)

            #-----------------------
            # Training the model
            #-----------------------

            best_loss = float("inf")
            patience, wait = 10, 0
            
            for epoch in range(max_epochs) :                                
                #--------- TRAINING ----------
                spdnet.train()                                              
                train_loss = 0                                              #  To stock the loss 

                for x,y,d in train_loader :
                    assert torch.all(d == d[0]), "Batch contains multiple domains!"
                    x = x.to(device)
                    y = y.to(device)
                    d = d.to(device)

                    domain_name = f"domain {d[0].item()}"
                    pred = spdnet(x, domain_name)
                    loss = criterion(pred, y)

                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(spdnet.parameters(), 1)
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                #--------- VALIDATION ---------
                spdnet.eval()
                correct = 0
                total = 0
                all_val_preds = []
                all_val_labels = []
                val_loss = 0

                with torch.no_grad():
                    for x,y,d in val_loader :
                        x = x.to(device)
                        y = y.to(device)
                        d = d.to(device)

                        domain_name = f"domain {d[0].item()}"
                        pred = spdnet(x, domain_name)
                        loss = criterion(pred, y)
                        val_loss += loss.item()

                        predictions = pred.argmax(dim=1)
                        all_val_preds.append(predictions.cpu())
                        all_val_labels.append(y.cpu())
                        correct += (predictions == y).sum().item()
                        total += y.size(0)

                val_loss /= len(val_loader)
                val_accuracy = correct / total
        
                predictions_val = torch.cat(all_val_preds)
                Y_val_full = torch.cat(all_val_labels)
                val_bal_acc = balanced_accuracy_score(Y_val_full.numpy(),predictions_val.numpy())
                print(f"Epoch {epoch+1}/{max_epochs} | Train loss : {train_loss:.3f} | Val loss : {val_loss:.3f} | Val accuracy : {val_accuracy:.3f}")

                # EARLY STOPPING 
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    best_model_state = copy.deepcopy(spdnet.state_dict())   # Save the best model properly
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping!")
                        spdnet.load_state_dict(best_model_state)
                        break
            print(">>> JE SUIS APRES LA BOUCLE")
            #--------- TEST --------- 
            spdnet.eval()
            correct = 0
            total = 0
            all_pred = []
            all_label = []

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

            predictions_test = torch.cat(all_pred)                                                # To concatenate tensors along a dimension 
            Y_test_full = torch.cat(all_label)

            # Balanced_accuracy on the test
            test_balanced_accuracy = balanced_accuracy_score(Y_test_full.numpy(),predictions_test.numpy())
            print(f"\nTest Balanced Accuracy {test_balanced_accuracy:.4f}")

            res_couche.append(test_balanced_accuracy)
        res_fold[i] = res_couche
    res_seed[seed] = res_fold

#-----------------------------------------------------------
# Converting the results into a vector y = 1D
#-----------------------------------------------------------
y = []
for seed in res_seed:
    for fold in res_seed[seed]:
        y.append(res_seed[seed][fold])

y = np.array(y)

# Sauvegarder en CSV 
np.savetxt("results_15_06_Cho2017'.csv", y, delimiter=",", header="reeig, cosh, coshP, expT, expP, tanheig ", comments="")

# Sauvegarder en txt 
np.savetxt("results_15_06_Cho2017.txt", y)

