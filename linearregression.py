# Steps for solving the linear regression algorithm
    
#     1- Initialize for the parameters slope (m), intercept (b)
#     2- Hypothesis function ypredicted = m.x + b
#     3- Computing cost function for the cost function
#     4- Computing the gradient: update for the values of m and b
#     5- repeat step 3, step 4 till reaching the number of iterations
#     6- Predict 

import numpy as np
def linearRegression(x, y, learning_rate = 0.001, iterations=1000):
    """
    Args:
    This linear regression function implements the linear regression algorithm from scratch
    x: Feature values
    y: Targeted values
    learning_rate: Step size for the gradient
    iterations: Number of iterations for the gradient itself

    Returns:
    m: slope of the line
    b: intercept of the line
    costs: List of cost function values 
    
    """

    #Initialize the parameters 
    m = 0
    b = 0
    n = len(y)
    costs = []
    for _ in range(iterations):
        y_pred = m*x + b
        cost = (1/n) * np.sum((y_pred - y) ** 2)
        costs.append(cost)

        # computing for the gradients 
        dm = (-2/n) * np.sum(x*(y- y_pred))
        db = (-2/n) * np.sum(y-y_pred)

        # updating for the parameters 
        m-= learning_rate * dm
        b-= learning_rate * db

    return m, b, costs



if __name__ == '__main__':
    x = np.array([1,2,3,4,5])

    y = np.array([2,4,6,8,10])

    m,b, costs = linearRegression(x, y, learning_rate=0.01, iterations=1000)


    print(f"Slope of (m): {m}")
    print(f"The intercept of b : is {b}")
    print(f"The final cost is {costs[-1]}")

    y_pred = m*x+ b
    print(f"The predicted values are {y_pred}")