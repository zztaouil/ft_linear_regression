from utils import read_model_params
from train import compute_cost
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    t0, t1 = read_model_params()
    print(f"Model Precision (RMSE):\
 {compute_cost(df["km"].values, df["price"].values, t0, t1)}")
