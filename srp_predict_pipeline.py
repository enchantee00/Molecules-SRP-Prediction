# materials.smi-ted (smi-ted)
from smi_ted_light.load import load_smi_ted_srp

# Data
import torch
import torch.nn.functional as F

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np

# Chemistry
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors

def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


def eda(df_lst):
    embed_df_lst = []
    feature_df_lst = []
    combined_df_lst = []
    y_lst = []
    for df in df_lst:
        df['norm_smiles'] = df['SMILE'].apply(normalize_smiles)
        df = df.dropna()

        with torch.no_grad():
            embeddings = model_smi_ted.encode(df['norm_smiles'])
        y = df[["SRP"]]

        X_embeddings = np.array(embeddings)
        df_features = df.drop(columns=['norm_smiles', 'SMILE', 'SRP', 'Unnamed: 0'])
        X_features = np.array(df_features)

        X_combined = np.hstack([X_embeddings, X_features])

        embed_df_lst.append(X_embeddings)
        feature_df_lst.append(X_features)
        combined_df_lst.append(X_combined)
        y_lst.append(y)
    
    return embed_df_lst, feature_df_lst, combined_df_lst, y_lst


def neural_network(embed, features):
    X_embedding_tensor = torch.tensor(embed, dtype=torch.float32).to(device)
    X_other_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    # y_tensor = torch.tensor(y.values, dtype=torch.float32)
        
        
    with torch.no_grad():
        y_pred_tensor = model_smi_ted.net(X_embedding_tensor, X_other_tensor).squeeze()

    y_nn = y_pred_tensor.cpu().numpy()
    # y = y_tensor.cpu().numpy()

    return y_nn



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # nn model finetuned
    model_smi_ted = load_smi_ted_srp(
        folder='models/smi_ted_light_srp',
        ckpt_filename='smi-ted-Light-Finetune_seed0_srp_epoch=14_valloss=0.2484.pt'
    ).to(device)

    # ML model optuna
    best_svr_params = {'C': 0.22323043102745171, 'epsilon': 0.01213144105136768, 'gamma': 0.014464031322773921}
    svr_model = SVR(**best_svr_params)

    best_gb_params = {'n_estimators': 400, 'learning_rate': 0.2221299759663393, 'max_depth': 3}
    gb_model = GradientBoostingRegressor(**best_gb_params)

    meta_model = Ridge(alpha=2.264839300910111)


    df_train = pd.read_csv('data/train.csv')
    df_valid = pd.read_csv('data/valid.csv')
    df_test = pd.read_csv('data/test.csv')

    df_lst = [df_train, df_valid, df_test]

    # 전처리 완료(임베딩 + 피처 분리 + y 값 분리)
    embed_df_lst, feature_df_lst, combined_df_lst, y_lst = eda(df_lst)

    y_valid_nn = neural_network(embed_df_lst[1], feature_df_lst[1])
    y_test_nn = neural_network(embed_df_lst[2], feature_df_lst[2])
    print("NN Predidction finished")

    svr_model.fit(combined_df_lst[0], y_lst[0].values.ravel())
    y_valid_svr = svr_model.predict(combined_df_lst[1])
    y_test_svr = svr_model.predict(combined_df_lst[2])
    print("SVR Predidction finished")


    gb_model.fit(combined_df_lst[0], y_lst[0].values.ravel())
    y_valid_gb = gb_model.predict(combined_df_lst[1])
    y_test_gb = gb_model.predict(combined_df_lst[2])
    print("GB Predidction finished")

    X_meta_train = np.column_stack([y_valid_nn, y_valid_svr, y_valid_gb, df_valid['ph']])  # 훈련 데이터 예측값 필요
    X_meta_test = np.column_stack([y_test_nn, y_test_svr, y_test_gb, df_test['ph']])

    meta_model.fit(X_meta_train, y_lst[1].values.ravel())
    y_pred_stacked = meta_model.predict(X_meta_test)

    mse_stacked = mean_squared_error(y_lst[2], y_pred_stacked)
    rmse_stacked = np.sqrt(mse_stacked)
    r2_stacked = r2_score(y_lst[2], y_pred_stacked)

    print(f"Stacking Ensemble - RMSE: {rmse_stacked:.4f}, R²: {r2_stacked:.4f}")


    

