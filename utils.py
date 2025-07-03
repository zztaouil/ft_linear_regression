import json
import matplotlib.pyplot as plt


def read_model_params():
    with open("model_params.json", "r") as f:
        obj = json.load(f)
    return obj["theta0"], obj["theta1"]


def write_model_params(theta0, theta1):
    with open("model_params.json", "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)


def compute_cost(mileage, price, theta0, theta1):
    """
     Root Mean Square Error
    """
    m = len(mileage)
    cost = 0

    for i in range(m):
        cost += ((theta0 + theta1 * mileage[i]) - price[i]) ** 2
    return (cost / (2 * m)) ** 0.5


def plot_data(mileage, price, theta0, theta1, codename):
    plt.figure()
    plt.scatter(mileage, price)
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title(f"Car Price vs Mileage {codename}")
    plt.plot(mileage, theta0 + theta1 * mileage, color="red")
    plt.savefig(f"car_price_vs_mileage_{codename}.png")
    plt.close()


def plot_loss_history(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.savefig("loss_history.png")
    plt.close()
