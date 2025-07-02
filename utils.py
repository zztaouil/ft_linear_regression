import json
import numpy as np

def read_model_params():
    with open("model_params.json", "r") as f:
        obj = json.load(f)
    return obj["theta0"], obj["theta1"]


def write_model_params(theta0, theta1):
    with open("model_params.json", "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)


def predict(mileage, theta0, theta1):
    return theta0 + theta1 * mileage