import pandas as pd
from utils import (write_model_params,
                   compute_cost,
                   plot_data,
                   plot_loss_history,
                   )


def normalize_features(data):
    a = data.min()
    b = data.max()
    return (data - a) / (b - a), a, b


def denormalize_features(data, a, b):
    return data * (b - a) + a


def denormalize_model_params(t0, t1, a, b, c, d):
    dt1 = t1 * (d - c) / (b - a)
    dt0 = c + t0 * (d - c) - dt1 * a
    return dt0, dt1


# theta 1 * mileage + theta 0 = price
def update_weights(X, y, theta0, theta1, learning_rate):
    theta0_gradient = 0
    theta1_gradient = 0
    m = len(X)

    for i in range(m):
        theta0_gradient += (theta0 + theta1 * X[i]) - y[i]
        theta1_gradient += ((theta0 + theta1 * X[i]) - y[i]) * X[i]

    new_theta0 = theta0 - learning_rate * (theta0_gradient / m)
    new_theta1 = theta1 - learning_rate * (theta1_gradient / m)
    return new_theta0, new_theta1


def train(mileage, price, theta0, theta1, l_rate, iterations):
    loss_history = []

    for i in range(iterations):
        theta0, theta1 = update_weights(mileage, price, theta0, theta1, l_rate)
        cost = compute_cost(mileage, price, theta0, theta1)
        if i % 1000 == 0:
            print(f"Iteration {i}: theta0 = {theta0:.20f},\
 theta1 = {theta1:.6f}, cost = {cost:.6f}")
        loss_history.append(cost)
    return theta0, theta1, loss_history


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    norm_X, a, b = normalize_features(df["km"].values)
    norm_y, c, d = normalize_features(df["price"].values)

    t0, t1, l_h = train(norm_X, norm_y, 0, 0, 0.001, 100000)

    d_t0, d_t1 = denormalize_model_params(t0, t1, a, b, c, d)
    write_model_params(d_t0, d_t1)
    print(f"Model Precision (RMSE):\
 {compute_cost(df["km"].values, df["price"].values, d_t0, d_t1)}")
    plot_data(df["km"].values, df["price"].values, d_t0, d_t1, "original")
    plot_data(norm_X, norm_y, t0, t1, "normalized")
    plot_loss_history(l_h)
