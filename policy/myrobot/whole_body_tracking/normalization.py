import torch, pandas as pd, numpy as np
m = torch.jit.load("policy.pt", map_location="cpu").eval()
amp = pd.read_csv("ampobs.csv")
obs = torch.tensor(amp.iloc[0, :89].values, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    a_pred = m(obs).squeeze(0).numpy()
a_train = amp.iloc[0, 89:110].values
print("pred:  ", np.round(a_pred, 3))
print("train: ", np.round(a_train, 3))
print("L2 diff:", np.linalg.norm(a_pred - a_train))