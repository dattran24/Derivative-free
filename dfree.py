####################### New Algorithm
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from scipy.optimize import fmin
from scipy import optimize
import time
# Set random seed for reproducibility

def forward_finite_difference(f, x, epsilon):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus_epsilon = np.copy(x)
        x_plus_epsilon[i] += epsilon
        grad[i] = (f(x_plus_epsilon) - f(x)) / epsilon
    return grad
def gradient_descent(f,ftrue, initial_x, maxcount, level):
    n=len(initial_x)
    L=n;
    C=L*np.sqrt(n)/2;
    D=2*np.sqrt(n);
    x = initial_x
    fcount=0;
    ftrue_values = [ftrue(x)]
    countall=[0]
    while fcount<maxcount:
        epsilon=math.sqrt(D*level/C)
        grad = forward_finite_difference(f, x, epsilon)
        fcount=fcount+n+1;
        L_init=L;
        C_init=L*np.sqrt(n)/2;
        loop=0;
        while f(x-1/L*grad) >f(x)-0.08/L*np.dot(grad, grad):
          loop=loop+1;
          fcount=fcount+1;
          if loop%2==1:
            L=L_init*2**(-loop//2);
            C=L*np.sqrt(n)/2;
            epsilon=math.sqrt(D*level/C)
            grad = forward_finite_difference(f, x, epsilon)
            fcount=fcount+n;
          else:
            L=L_init*2**(loop//2);
            C=L*np.sqrt(n)/2;
            epsilon=math.sqrt(D*level/C)
            grad = forward_finite_difference(f, x, epsilon)
            fcount=fcount+n;
        xold=x;
        x=x-1/L*grad;
        if math.isnan(ftrue(x)):
            x=xold;
        if fcount>=maxcount:
            ftrue_values.append(ftrue(xold))
            countall.append(maxcount)
        else:
            ftrue_values.append(ftrue(x))
            countall.append(fcount) 
        
    return x, f(x), ftrue_values,fcount,countall
import matplotlib.pyplot as plt
def multipleplot(ftrue_values, L_values, L_approximate_values, Loop_values,countall, join):
    fig, axs = plt.subplots(1, 3, figsize=(25, 5))
    axs[0].plot(countall, ftrue_values)
    axs[0].set_xlabel('Function evaluations')
    axs[0].set_ylabel('Function Value')
    axs[0].set_yscale('log')
    axs[0].set_title('Function Values with Gradient Descent')
    axs[0].grid(True)
  
    axs[1].plot(range(len(L_values)), L_values, label='L-GDF ')
    if join == 'yes':
        axs[1].plot(range(len(L_approximate_values)), L_approximate_values, marker='o', label='L-TRUE')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Lipschitz Estimate')
    axs[1].set_yscale('log')
    axs[1].set_title('Lipschitz Estimate')
    axs[1].grid(True)
    axs[1].legend()
    
    axs[2].plot(range(len(Loop_values)), Loop_values)
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Loop Bisection')
    axs[2].set_title('Loop Bisection')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
def plot_many_functions(*args, labels=None, xlabel=None, ylabel=None, title=None):
    """
    Plot multiple arrays in the same plot.

    Parameters:
        *args (array-like): Arrays to be plotted.
        labels (list of str, optional): Labels for each array.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        title (str, optional): Title of the plot.
    """
    if labels is None:
        labels = [f"f_{i+1}" for i in range(len(args))]
    
    for i, arr in enumerate(args):
        plt.plot(arr, label=labels[i])
    
    plt.xlabel("Iterations" if xlabel is None else xlabel)
    plt.ylabel("Function Value" if ylabel is None else ylabel)
    
    if title:
        plt.title(title)
    
    plt.legend()
    plt.show()
