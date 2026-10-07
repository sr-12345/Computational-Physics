import numpy as np
import matplotlib as plt

#function to be differentiated (in this case cos^2(2x))

def f(x):
    return (np.cos(2*x))**2

#analytical derivative of f(x) for comparison

def df_analytic(x):
    return -4*np.cos(2*x)*np.sin(2*x)


#forward difference method for numerical differentiation, uses an interval dx to calculate the derivative at position x

def forward_difference(f, x, dx):

    forward_difference = (f(x+dx)-f(x))/(dx)

    return forward_difference


#plot the difference between the analytical derivative and the numerical implementation for varying dx sizes

xs = np.linspace(-2*np.pi,2*np.pi,100)
df_dx_1 = forward_difference(f, xs, dx=1e-5)
df_dx_2 = forward_difference(f, xs, dx=1e-8)
df_dx_analytical = df_analytic(xs)
plt.figure(figsize=(8, 4))
plt.plot(xs, df_dx_1 - df_dx_analytical, label = 'dx too large')
plt.plot(xs, df_dx_2 - df_dx_analytical, label = 'dx optimal')

df_dx_3 = forward_difference(f, xs, dx=1e-11)
plt.plot(xs, df_dx_3 - df_dx_analytical, label = 'dx too small')
plt.xlabel('x')
plt.ylabel('difference(numerical - analytical)')
plt.title('Analysis of descrepencies between analytical derivative and numerical implementation for varying dx size')
plt.legend()