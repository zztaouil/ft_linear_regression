import pandas as pd
import numpy as np
from utils import predict, write_model_params, read_model_params
import matplotlib.pyplot as plt


def normalize_features(data):
    """Normalize features to have mean 0 and std 1"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std


# theta 0 + theta 1 * mileage = price
def update_weights(mileage, price, theta0, theta1, learning_rate):
    theta0_gradient = np.float64(0)
    theta1_gradient = np.float64(0)
    m = len(mileage)

    for i in range(m):
        theta0_gradient += (theta0 + theta1 * mileage[i]) - price[i]
        theta1_gradient += ((theta0 + theta1 * mileage[i]) - price[i]) * mileage[i]

    new_theta0 = theta0 - learning_rate * (theta0_gradient / m)
    new_theta1 = theta1 - learning_rate * (theta1_gradient / m)
    return new_theta0, new_theta1


def compute_cost(mileage, price, theta0, theta1):
    m = len(mileage)
    cost = np.float64(0)

    for i in range(m):
        cost += ((theta0 + theta1 * mileage[i]) - price[i]) ** 2
    return cost / m


def train(mileage, price, theta0, theta1, learning_rate, iterations):
    for i in range(iterations):
        theta0, theta1 = update_weights(mileage, price, theta0, theta1, learning_rate)
        cost = compute_cost(mileage, price, theta0, theta1)
        if i % 1000 == 0:
            print(
                f"Iteration {i}: theta0 = {theta0:.20f}, theta1 = {theta1:.6f}, cost = {cost:.6f}"
            )
    return theta0, theta1


def plot_data(mileage, price, theta0, theta1, codename):
    plt.scatter(mileage, price)
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car Price vs Mileage")
    plt.plot(mileage, theta0 + theta1 * mileage, color="red")
    plt.savefig(f"car_price_vs_mileage_{codename}.png")

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    normalized_mileage, km_mean, km_std = normalize_features(df["km"].values)
    normalized_price, price_mean, price_std = normalize_features(df["price"].values)

    t0, t1 = train(normalized_mileage, normalized_price, 0, 0, 0.0001, 20000)

    plot_data(df["km"].values, df["price"].values, t0, t1, "original")
    t0 = t0 * price_std + price_mean
    t1 = t1 * price_std / km_std
    plot_data(normalized_mileage, normalized_price, t0, t1, "normalized")
    