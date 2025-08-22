import pandas as pd
import numpy as np
import matplotlib as plt
import logging, time, os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch.optim.lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, classification_report, f1_score
from preprocessing import preprocess
from sklearn.utils.validation import check_is_fitted
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cpu")
logger.info(f"Utilizing CPU since GPU doesn't exit here: {device} \n")

class prepfortorch(Dataset):
    def __init__(self, x, y):

        #since x is a dense numpy array
        if hasattr(x, "toarray"):   
            X = x.toarray()
        else:
            X = x
            
        X = np.asarray(X, dtype=np.float32)
        y = y.to_numpy().astype(np.float32).reshape(-1) # so its 1-D

        self.X = torch.from_numpy(X) #create tensors from numpy arrays
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class classmodel(nn.Module):
    def __init__(self, num_feat):
        super().__init__()

        self.model = nn.Sequential( #set up the container
            nn.Linear(num_feat, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) #BCEwithLogitsLoss
        )

    def forward(self, x):
        return self.model(x)
    
def train_time(df, model, n_factors = 30, n_epochs = 20, batch_size = 50, device = "cpu", pos_weights = None, val_ds = None):

        #lets start off with AdamW optimizer and l2 regularization (weight decay)
        optimizer = optim.AdamW(model.parameters(), lr = 0.001, weight_decay= 1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = 'max',
            factor= 0.5,
            patience= 2,
            min_lr = 1e-6
        )

        criteron = nn.BCEWithLogitsLoss(pos_weight= pos_weights)
        
        #train/val Dataloader method
        data_loader = DataLoader(df, batch_size= 512, shuffle= True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, drop_last= True)

        model.to(device)

        best_loss = float('inf')
        best_model_state = None #initiliazation
        no_improve_epochs = 0
        epoch_loss = []

        history = [] #to save everything(mode, weights and parameters)

        for epoch in range(n_epochs):
            model.train()
            total_train_loss = 0
            total_samples = 0

            for batchx, batchy in data_loader:
                batchx = batchx.to(device)
                batchy = batchy.to(device).float().unsqueeze(1) 

                optimizer.zero_grad()

                prediction = model(batchx)

                loss = criteron(prediction, batchy)
                bs = batchx.size(0) #batch size to calculate total loss 

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * bs #scale by batch size
                total_samples += bs

                #validation phase
            
            model.eval()
            total_val_loss = 0
            val_sample = 0

            with torch.no_grad():   #no more gradient calculation
                for batchx, batchy in val_loader:
                    batchx = batchx.to(device)
                    batchy = batchy.to(device).float().unsqueeze(1)

                    #make the prediction
                    prediction = model(batchx)

                    loss = criteron(prediction, batchy)
                    vs = batchx.size(0)

                    total_val_loss +=loss.item() * vs
                    val_sample += vs
                
            
            avg_train_loss = total_train_loss / total_samples
            avg_val_loss = total_val_loss / val_sample

            epoch_loss.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            if (epoch + 1)%1 ==0:
                logging.info((f"Epoch {epoch+1} / {n_epochs} train Loss: {avg_train_loss:.4f}, valid loss: {avg_val_loss:.4f}"))
            
            #print metrics from validate function
            val_metrics = validate(val_ds, model, device, batch_size)
            
            logging.info(
                f"val PR-AUC: {val_metrics['pr_auc']:.5f}, "
                f"val ROC-AUC: {val_metrics['roc_auc']:.5f}, "
                f"F1: {val_metrics['f1'].max():.5f}, Tau: {val_metrics['tau']:.5f}"
            )
            
            #save now
            history.append({ "epoch": epoch+1,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                            })
            
            #early stopping check
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                no_improve_epochs = 0
                best_model_state = model.state_dict() #save the best model weights
            
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= 5:
                logging.info(f"Stopping early at epoch {epoch+1} -No improvement for 5 epoch")
                break
            
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.cpu() #move model back to the cpu

        plt.plot(epoch_loss)
        plt.title("Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

        return model, best_model_state, {"history": history}



@torch.no_grad()
def validate(valid_set, model, device, batch_size):

    #enter evaluation mode - important for batch normalization/dropout layers
    model.eval()

    valid_loader = DataLoader(valid_set, batch_size= batch_size, shuffle= False)

    all_p, all_y = [], [] #empty lists to hold
    for batchx, batchy in valid_loader:
        batchx = batchx.to(device)
        batchy = batchy.to(device).float().unsqueeze(1)

        #make prediction
        predict = model(batchx)

        probs = torch.sigmoid(predict) # or like this we did for the titanic  pred_labels = (pred >= 0.5).long().squeeze() #convert into class lables (0 -> not survived, 1-> survived)

        all_p.append(probs.cpu()); all_y.append(batchy.cpu())
    
    y = torch.cat(all_y).numpy()
    p = torch.cat(all_p).numpy()

    pr_auc = average_precision_score(y, p)
    roc_score = roc_auc_score(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    
    f1 = 2*prec*rec / (prec + rec + 1e-12)
    best = f1.argmax()
    tau = thr[max (0, best-1)]
    
    return {
    "pr_auc": pr_auc,
    "roc_auc": roc_score,
    "f1": f1,
    "tau": float(tau),
    }

def save_model(run_dir, model, best_state):
    """ Save for model depolyment"""

    Path(run_dir).mkdir(parents=True, exist_ok=True)

    #save cpu tensors
    best_state_cpu = {k: v.cpu() for k, v in best_state.items()}
    model_cpu = model.to("cpu")

    #Now save best model and weights
    torch.save(best_state, f"{run_dir}/model_state.pt")

    #save entire model (no need to redefine class)
    torch.save(model, f"{run_dir}/model_full.pt" )

    print(f"✅ Model saved to {run_dir} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

def evaluate(test_set, model, tau, batch_size):
    
    #lets really evaluate the model using the test set now!!

    model.eval()

    test_loader = DataLoader(test_set, batch_size= batch_size, shuffle= False)
    all_p, all_y = [], [] #empty lists to hold
    
    with torch.no_grad():
        for batchx, batchy in test_loader:
            batchx = batchx.to(device)
            batchy = batchy.to(device)

            predict = model(batchx)

            proba = torch.sigmoid(predict) #convert from logits to probabalities 

            all_p.append(proba.cpu()); all_y.append(batchy.cpu())
        
        y = torch.cat(all_y).numpy() #concatatenate the tensors into numpy
        p = torch.cat(all_p).numpy()
        
        pr_curve = average_precision_score(y,p)
        roc_auc = roc_auc_score(y, p)

        prec, rec, thr = precision_recall_curve(y, p)
        
        #tau from validation
        tau = float(tau)
        y_hat = (p >= tau).astype(int)

        #now calculate classification report/confusion matrix from fixed set
        cm = confusion_matrix(y,y_hat, labels=[0,1])
        report = classification_report(
            y, y_hat,
            labels=[0,1],
            target_names=["not_fraud", "fraud"],
            digits=4,
            zero_division=0
        )
        
        logger.info(f"precision_recall_curve: {pr_curve}")
        logger.info(f"roc_auc score: {roc_auc}")
        logger.info(f"Confusion Matrix: \n{cm}")
        logger.info(f"classification_report: \n{report}")
        

        
def main():
    #main method to do everything
    df1 = pd.read_parquet(r"C:\Users\mubarak.derie\OneDrive - Accenture\Documents\Python\2025_projects\torch_proj#1_classify\paysim_data_pt.parquet") #loading 6 million

    #lets take a sample of 50-100k for now
    df = df1.sample(n=100000, random_state= 42)

    #logger.info(df.head())

    #apply preprocessing, this is for classification
    x, y, preprocessor_cls = preprocess(df, task= 'classification')

    #split into train+val and test (60% , 20% and 20%)
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

    #Now split into train and val 
    x_train, x_val, y_train, y_val = train_test_split(x_temp,y_temp, test_size=0.25, stratify=y_temp, random_state = 42)


    #This part is just for saving fitted preprocessor as a pkl file for depolyment
    try:
        check_is_fitted(preprocessor_cls)
    except Exception:
        preprocessor_cls.fit(df) #fit the raw dataframe so the transformer learns the full schema/categories
    
    #joblib.dump(preprocessor_cls, "preprocessor_cls.pkl")
    #logger.info("Saved fitted -> classification preprocessor -> preprocessor_cls.pkl")

    #apply fit/transform to train set and only tranform on test set
    pipeline = Pipeline(steps=[('preprocessor', preprocessor_cls)])

    x_train_pp = pipeline.fit_transform(x_train).astype(np.float32)
    x_val_pp = pipeline.transform(x_val).astype(np.float32)
    x_test_pp = pipeline.transform(x_test).astype(np.float32)

    #into tensors for torch
    train_ds = prepfortorch(x_train_pp, y_train)
    val_ds = prepfortorch(x_val_pp, y_val)
    test_ds = prepfortorch(x_test_pp, y_test)
    
    #to help with the class imbalance
    pos_weights = torch.tensor([
        (y_temp == 0).sum() / (y_temp==1).sum() 
    ], dtype=torch.float32).to(device)


    #Model training time
    num_feat = x_train_pp.shape[1] #compress features to number of columns
    base_model = classmodel(num_feat)
    
    
    #model, best_model_state, logs = train_time(train_ds, base_model, n_factors=30, n_epochs = 25, batch_size= 512, device="cpu", pos_weights= pos_weights, val_ds = val_ds)

    #Now save the model to this folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #run_dir = os.path.join("artificats", f"paysim_classify_model_{timestamp}")
    #save_model(run_dir, model, best_model_state)
    
    #get tau from validation:
    #val = validate(val_ds, base_model, device, batch_size=512)
    #logging.info(
    #    f"val PR-AUC: {val['pr_auc']:.5f}, "
    #    f"val ROC-AUC: {val['roc_auc']:.5f}, "
    #    f"F1: {val['f1'].max():.5f}, Tau: {val['tau']:.5f}"
    #)

    #now really evaluate using test set
    #base_model.load_state_dict(torch.load(r"C:\Users\mubarak.derie\OneDrive - Accenture\Documents\Python\2025_projects\torch_proj#1_classify\artificats\paysim_classify_model_2025-08-14_15-47-07\model_state.pt", map_location="cpu")) #Load the model weights

    #evaluate(test_ds, base_model, val["tau"], batch_size=512)
    
    
if __name__=="__main__":
    main()
