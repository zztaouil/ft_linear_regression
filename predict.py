import json
from utils import read_model_params, predict



if __name__ == "__main__":
    theta0, theta1 = read_model_params()

    print("Welcome to the car price prediction system\n")
    mileage = int(input("Enter the car mileage: "))
    prediction = predict(mileage, theta0, theta1)
    print(f"The predicted price is {prediction}")