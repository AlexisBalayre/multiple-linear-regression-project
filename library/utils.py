import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from library.LinearRegressionCustomModel1 import LinearRegressionCustomModel1
from library.LinearRegressionCustomModel2 import LinearRegressionCustomModel2
from library.PolynomialDegree5RegressionCustomModel import (
    PolynomialDegree5RegressionCustomModel,
)
from library.QuadraticRegressionCustomModel import QuadraticRegressionCustomModel


# Function to compute the optimal train size
def computeOptimalTrainSize(r):
    # Compute optimal train size
    optimalTestSize = 1 / (np.sqrt(r) + 1)
    optimalTrainSize = 1 - optimalTestSize

    return optimalTrainSize


# Function to compute the optimal parameters depending on the model type
def computeOptimalParameters(x_train, x_test, y_train, y_test, model_type):
    # Set hyperparameters
    iterations_numbers = [
        200,
        300,
        400,
        500,
        1000,
        2000,
        3000,
        5000,
        7000,
        10000,
        20000,
        50000,
    ]  # Number of iterations
    learning_rates = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]  # Learning rate

    # Initialise an empty array to store the results
    results = []

    # Set seed for reproducibility
    np.random.seed(0)

    # Computes linear regression for each hyperparameters
    for iterations_nb in iterations_numbers:
        for learning_rate in learning_rates:
            # Initialise model
            if model_type == "linear_2":
                # Linear Model 2
                model = LinearRegressionCustomModel2(learning_rate, iterations_nb)
            elif model_type == "quadratic":
                # Quadratic model
                model = QuadraticRegressionCustomModel(learning_rate, iterations_nb)
            elif model_type == "degree_5":
                # Polynomial degree 5 model
                model = PolynomialDegree5RegressionCustomModel(
                    learning_rate, iterations_nb
                )
            else:
                # Model 1
                model = LinearRegressionCustomModel1(learning_rate, iterations_nb)
            # Fit model to training data
            theta, cost = model.fit(x_train, y_train)
            # Skip if training was not successful
            if theta is None:
                continue
            # Compute metrics
            rSquare, meanAbErr, meanSqErr, rootMeanSqErr = model.metrics(x_test, y_test)
            # Save results
            results.append(
                {
                    "iterations_nb": iterations_nb,
                    "learning_rate": learning_rate,
                    "cost": cost,
                    "rSquare": rSquare,
                    "meanAbErr": meanAbErr,
                    "meanSqErr": meanSqErr,
                    "rootMeanSqErr": rootMeanSqErr,
                }
            )

    # Convert results to dataframe
    results = pd.DataFrame(results)

    # Sort results by MSE
    results = results.sort_values("meanSqErr", ascending=True)

    # Return best hyperparameters
    return (
        int(results.iloc[0]["iterations_nb"]),
        results.iloc[0]["learning_rate"],
        results,
    )


# Function to prepare data (split data into train and test sets, and standardise data)
def prepare_data(x, y, train_size):
    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)

    # standardise data
    x_train = (x_train - x_train.mean()) / x_train.std()  # standardise train set
    x_test = (x_test - x_test.mean()) / x_test.std()  # standardise test set

    #  Return train and test sets
    return x_train, x_test, y_train, y_test
