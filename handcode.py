import numpy as np
import matplotlib.pyplot as plt
import math

def fordif(f, x, epsilon):
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n):
        x_plus_epsilon = x.copy()
        x_plus_epsilon[i] = x_plus_epsilon[i] + epsilon
        grad[i] = (f(x_plus_epsilon) - f(x)) / epsilon
    return grad

def implicit(f,ftrue, maxcount, x):
    k = 1
    n = len(x)
    tmin = 1e-10
    beta = 0.1
    gamma = 0.5

    fun_all = []
    count_all = []
    count = 0
    fun_all.append(f(x))
    count_all.append(0)
    x_all = [x]

    while count < maxcount:
        g = fordif(f, x, 2 / (2 ** k))
        count += n + 1
        t = 1
        count += 2
        while f(x - t * g) > f(x) - beta * t * np.linalg.norm(g) ** 2 and t >= tmin:
            t = gamma * t
            count += 2
        if t < tmin:
            t = 0;
        x = x - t * g
        fun_all.append(ftrue(x))
        x_all.append(x)
        if count >= maxcount:
            count = maxcount
        count_all.append(count)
        k += 1
    return x, fun_all, count_all, x_all
def RG(f,ftrue, x_init, L=None, mu=None,h=None, maxcount=None):
    n=len(x_init)
    if L is None:
      L=len(x_init);
    if mu is None:
      mu = (5 / (3 * L + 12)) * np.sqrt(1e-2 / (2 * L));
    if h is None:
      h=1 / ((4 * n + 16) * L)
    x = x_init
    fun_all = [f(x)]
    count = 0
    count_all = [count]
    x_all = [x]
    k=1;
    while count < maxcount:
        u = np.random.randn(n)
        g = ((f(x + mu * u) - f(x)) / mu) * u
        count += 2
        x -= h * g
        k += 1
        fun_all.append(ftrue(x))
        count_all.append(count)
    return x_all, fun_all, count_all
def fordif(f, x, epsilon):
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n):
        x_plus_epsilon = x.copy()
        x_plus_epsilon[i] += epsilon
        grad[i] = (f(x_plus_epsilon) - f(x)) / epsilon
    return grad
def DFC(f,ftrue, x_init, maxcount, eps, L):
    n = len(x_init)
    if L==None:
      L=len(x_init);
    kappa = np.sqrt(n) / 2
    delta = eps
    r = 2
    x = x_init
    theta = 0.5
    fun_all = []
    count_all = []
    count = 0
    fun_all.append(f(x))
    count_all.append(0)
    while count < maxcount:
        if delta<1e-20:
            break
        g = fordif(f, x, delta)
        count += n + 1
        while np.linalg.norm(g) <= 2.5 * L*kappa * delta:
            delta = theta * delta
            g = fordif(f, x, delta)
            count += n
        if f(x - (1/L)*g ) <= f(x) - (0.1/L)*np.linalg.norm(g) ** 2:
            count += 1
            x = x - (1/L)*g
        else:
            count += 1
            L *= r
        fun_all.append(ftrue(x))
        if count >= maxcount:
            count = maxcount
        count_all.append(count)
    return x, fun_all,count_all
def GDada(f,ftrue, x_init, maxcount, eps, L):
    n = len(x_init)
    if L==None:
      L=len(x_init);
    kappa = np.sqrt(n) / 2
    delta = eps
    r = 2
    x = x_init
    theta = 0.5
    fun_all = []
    count_all = []
    count = 0
    fun_all.append(f(x))
    count_all.append(0)
    while count < maxcount:
        if delta<1e-20:
            break
        g = fordif(f, x, delta)
        count += n + 1
        while np.linalg.norm(g) <= 2.5 * L*kappa * delta:
            delta = theta * delta
            g = fordif(f, x, delta)
            count += n
        if f(x - (1/L)*g ) <= f(x) - (0.1/L)*np.linalg.norm(g) ** 2:
            count += 1
            x = x - (1/L)*g
            L =L/ r
        else:
            count += 1
            L *= r
        fun_all.append(ftrue(x))
        if count >= maxcount:
            count = maxcount
        count_all.append(count)
    return x, fun_all,count_all
def DFB(f,ftrue, maxcount, x, delta=None):
    if delta==None:
       delta=1e-4
    k = 1
    n = len(x)
    L = 1
    tmin = 1e-6
    mu = 2.1
    eta = 2
    theta = 0.5
    beta = 0.1
    gamma = 0.5
    kappa=np.sqrt(n)/2;
    fun_all = [f(x)]
    count_all = [0]
    count = 0

    while count < maxcount:
        g = fordif(f, x, delta)
        count += n + 1
        while np.linalg.norm(g) <= mu * kappa * delta * L and delta>=1e-30:
            delta = theta * delta
            g = fordif(f, x, min(delta, 1 / (k)))
            count += n 
        if delta<1e-30:
            break
        t = 1
        while f(x - t * g) > f(x) - beta * t * np.linalg.norm(g) ** 2 and t >= tmin:
            t = gamma * t
            count += 1
        if t < tmin:
            t = t
            L = eta * L
            tmin = tmin / 2
        x = x - t * g
        fun_all.append(ftrue(x))
        if count >= maxcount:
            count = maxcount
        count_all.append(count)
        k += 1
    #print(delta)
    return x, fun_all, count_all

def GD_FD_backtracking(f,ftrue, initial_point, maxcount,delta=None):
    alpha=0.1;
    beta=0.5;
    if delta==None:
      delta=1e-8
    x = initial_point;
    n=len(x);
    count=0;
    fun_all=[];
    if not math.isnan(ftrue(x)):
        fun_all.append(ftrue(x))
    count_all = [0]
    while count<maxcount:
        grad = fordif(f,x,delta)
        count=count+n+1;
        t = 1;
        while f(x - t *grad) > f(x) - alpha * t * np.dot(grad, grad) and t>=1e-10:
            t = beta * t
            count=count+1
        x = x - t * grad
        fun_all.append(ftrue(x))
    return x, fun_all
def GD_FD_constant(f,ftrue, initial_point, maxcount,delta,stepsize):
    if delta==None:
        delta=1e-8
    if stepsize==None:
        stepsize=1/len(initial_point);
    x = initial_point
    count=0;
    n=len(x);
    fun_all=[ftrue(x)];
    count_all = [0]
    while count<maxcount:
        grad = fordif(f,x,delta)
        count=count+n+1;
        x = x - stepsize * grad
        fun_all.append(ftrue(x))
        if count >= maxcount:
            count = maxcount
        count_all.append(count)
    return x, fun_all,count_all
def GD_FD_constantnew(f,ftrue, initial_point, maxcount,delta):
    x = initial_point
    L=len(x);
    count=0;
    n=len(x);
    fun_all=[ftrue(x)];
    count_all = [0]
    while count<maxcount:
        grad = fordif(f,x,delta)
        count=count+n+1;
        stepsize=1/L;
        if f(x - stepsize * grad)<=f(x) - 0.15*stepsize*np.linalg.norm(grad) ** 2:
            count=count+1
            x = x - stepsize * grad
        else:
            count=count+1
            L=L*2;
        fun_all.append(ftrue(x))
        if count >= maxcount:
            count = maxcount
        count_all.append(count)
    return x, fun_all,count_all