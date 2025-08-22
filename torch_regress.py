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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, classification_report, f1_score
from preprocessing import preprocess
from sklearn.utils.validation import check_is_fitted
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cpu")
logger.info(f"Utilizing CPU since GPU doesn't exit here: {device} \n")

class intotorch(Dataset):
    def __init__(self,x,y):
        
        y = y.to_numpy(dtype = np.float32).reshape(-1) #convert to numpy since its a series and then reshape so its 1-D

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class regressmodel(nn.Module):
    
    def __init__(self, num_feats):
        super().__init__()

        self.regress = nn.Sequential(
            nn.Linear(num_feats, 128, bias= False), #no need for bias before BN
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128,64, bias=True), #add a bias term
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64,1) #linear output
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        #function for setting up xavier init with weights
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.regress(x).squeeze(1)
    

def train_regress(train_ds, model, n_factors = 30, n_epochs = 50, batch_size = 512, device = "cpu", val_ds = None):
    
    #utilize dataloader:
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    #Lets setup the optimizer
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode= 'min',
        factor = 0.8,
        patience=1,
        #threshold = 1e-4,
        min_lr=1e-6,
    )

    criteron = nn.MSELoss() #huberloss combines MSE+MAE

    model.to(device) #off to the cpu

    best_model_state = None
    best_loss = float('inf')
    no_improve_epochs = 0 #for early stopping
    epoch_loss = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_samples = 0

        for batchx,batchy in train_dataloader:
            batchx = batchx.to(device)
            batchy = batchy.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            predictions = model(batchx).unsqueeze(1)

            loss = criteron(predictions, batchy)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #clip gradients
            optimizer.step()

            bs = batchx.size(0)

            train_loss += loss.item() * bs
            train_samples += bs

        model.eval()
        val_loss = 0
        val_samples = 0

        with torch.no_grad():
            for batchx,batchy in val_loader:
                batchx = batchx.to(device)
                batchy = batchy.to(device).float().unsqueeze(1)

                predictions = model(batchx).unsqueeze(1)

                loss = criteron(predictions, batchy)
                vs = batchx.size(0)

                val_loss += loss.item() * vs
                val_samples += vs


        avg_train_loss = train_loss / train_samples
        avg_val_loss = val_loss/ val_samples

        epoch_loss.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        

        if (epoch + 1) %1 == 0:
            logging.info(f" Epoch {epoch + 1} / {n_epochs} Train Loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")

        val_metrics = validate_metrics(val_ds, model, batch_size)
        logging.info(f"validation mae: {val_metrics['mae']:.4f}, validation rmse: {val_metrics['rmse']:.4f}")

        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict().copy()
        
        else:
            no_improve_epochs +=1

        if no_improve_epochs >=5:
            logging.info(f"Stopping early at epoch {epoch + 1} - No improvement for 5 Epoch")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.cpu()

    plt.plot(epoch_loss)
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return model, best_model_state

def validate_metrics(valid_ds, model, batch_size):

    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    model.eval()

    all_p, all_y = [],[]

    with torch.no_grad():
        for batchx, batchy in valid_loader:
            batchx = batchx.to(device)
            batchy = batchy.to(device)

            predications = model(batchx)

            all_p.append(predications.cpu()); all_y.append(batchy.cpu())

    y = torch.cat(all_y).numpy()
    p = torch.cat(all_p).numpy()

    mae_metric = mean_absolute_error(y, p)
    rmse_metric = root_mean_squared_error(y, p)

    #return metrics as dicts
    return {
        'mae': mae_metric,
        'rmse': rmse_metric
    }

def save_regress_model(run_dir, model, best_state):
    """ save for model depolyment"""

    Path(run_dir).mkdir(parents=True, exist_ok=True)

    #save cpu tensors
    best_state_cpu = {k: v.cpu() for k,v in best_state.items()}
    model_cpu = model.to("cpu")

    #now save the best model
    torch.save(best_state, f"{run_dir}/model_state.pt")

    torch.save(model, f"{run_dir}/model_full.pt")

    print(f" Model saved to {run_dir} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

def make_y_inverse(task: str):
    #for regression, we log1p(amount) -> inverse is expm1
    if task == 'regression':
        return lambda arr: np.expm1(arr)

    return None #for classification

def evaluate(test_ds, model, batch_size, task = 'regression'):
    #evaluating the model on the test set
    
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_pred, all_y = [], []

    with torch.no_grad():
        for batchx, batchy in test_loader:
            batchx = batchx.to(device)
            batchy = batchy.to(device).float().view(-1)

            predictions = model(batchx).float().view(-1)
            
            all_pred.append(predictions);   all_y.append(batchy)
    
    y = torch.cat(all_y).numpy()
    p = torch.cat(all_pred).numpy()

    rmse = root_mean_squared_error(y, p)
    mae = mean_absolute_error(y, p)
    med_abs_error = np.median(np.abs(y - p))

    logger.info(f"Test MAE_std: {mae}, Test RMSE_std: {rmse}, Median Abs Error: {med_abs_error}")
    results = {"mae_std": mae, "rmse_std": rmse}

    #additional part: Inverse transform to show actual metrics
    y_inverse = make_y_inverse(task)
    if y_inverse is not None:
        
        #gotta ensure its 2D for sklearn metrics
        y_real = y_inverse(y.reshape(-1,1)).reshape(-1)
        p_real = y_inverse(p.reshape(-1,1)).reshape(-1)

        mae_real = mean_absolute_error(y_real, p_real)
        rmse_real = root_mean_squared_error(y_real, p_real)

        logger.info(f" [REAL] Test MAE: {mae_real:.4f}, Test RMSE: {rmse_real:.4f}")
        results.update({"mae_real": mae_real, "rmse_real": rmse_real})

    return results


def main():
    
    #main method to do everything
    df = pd.read_parquet(r"C:\Users\mubarak.derie\OneDrive - Accenture\Documents\Python\2025_projects\torch_proj#1_classify\paysim_data_pt.parquet") #loading 6 million

    #lets take a sample of 50-100k for now
    n_rows = min(100000, len(df))
    df = df.sample(n=n_rows, random_state= 42)
    
    #call preprocess method
    x,y, preprocessor_reg = preprocess(df, task = 'regression')

        
    #train/val/test set up
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

    #this additional code is just for saving the regression preprocessing pkl file for depolyment
    try:
        check_is_fitted(preprocessor_reg)
    except Exception:
        preprocessor_reg.fit(df)

    #joblib.dump(preprocessor_reg, "preprocessor_reg.pkl")
    #logger.info("Saved fitted preprocessor -> preprocessor_reg.pkl")

    #set up preprocessor pipeline
    pipeline = Pipeline(steps= [("preprocessor", preprocessor_reg)])

    #Now fit/transform preprocessor respectfully
    x_train_pp = pipeline.fit_transform(x_train).astype(np.float32)
    x_val_pp = pipeline.transform(x_val).astype(np.float32)
    x_test_pp = pipeline.transform(x_test).astype(np.float32)

    #into torch Dataset for conversion to tensors
    train_ds = intotorch(x_train_pp, y_train)
    val_ds = intotorch(x_val_pp, y_val)
    test_ds = intotorch(x_test_pp, y_test)

    input_feats = x_train_pp.shape[1] #compress features 
    base_model = regressmodel(input_feats)

    #run the model/train
    #model, best_model_state = train_regress(train_ds, base_model, n_factors= 30, n_epochs= 25, batch_size=512, device="cpu", val_ds = val_ds)

    #save the model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #run_dir = os.path.join(os.path.dirname(__file__), "regress_artificats", f"paysim_regress_model_{timestamp}")
    #save_regress_model(run_dir, model, best_model_state)

    #load best model for evaluate
    best_model_load = torch.load(r"C:\Users\mubarak.derie\OneDrive - Accenture\Documents\Python\2025_projects\torch_proj#1_classify\regress_artificats\paysim_regress_model_2025-08-18_13-14-55\model_state.pt")

    base_model.load_state_dict(best_model_load)

    #evaluate(test_ds, base_model, batch_size=512, task = 'regression')

if __name__=="__main__":
    main()
