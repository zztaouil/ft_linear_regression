from utils import read_model_params


if __name__ == "__main__":
    theta0, theta1 = read_model_params()

    print("Welcome to the car price prediction system\n")
    try:
        mileage = int(input("Enter the car mileage: "))
        prediction = theta1 * mileage + theta0
        print(f"The predicted price is {prediction:.3f}")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
