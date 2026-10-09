import numpy
from matplotlib import pyplot as plt
import matplotlib.colors
from random import random


# This script illustrates the different behaviours of the gradient descent (GD) method when finding the minima of 
# Rosenbrock's Banana Function,
# 𝑓(𝑥,𝑦)≡(1−𝑥)2+100(𝑦−𝑥2)2 .


# banana function

def f(r):
    '''Function to be minimised'''
    x, y = r

    return (1-x)**2 + 100*(y-x**2)**2

# analytical derivative of banana function

def grad(r):
    '''Calculate gradient of banana function at coordinates r = (x,y)'''
    x, y = r
    
    output = (-2*(1-x)-400*x*(y-x**2) , 200*(y-x**2))
    return numpy.array(output)


# Gradient descent algorithm, returns an array of the coordinates of the trajectory of the gradient descent method

def gradientDescent(df, r0, eta, nstep):
    xy = r0
    history = numpy.empty( (nstep+1, 2) )
    
    history[0]=xy
    
    for i in range(nstep):
        xy = xy - df(history[i])*eta
        history[i+1]=xy
        
    return history



# Plotting: trajectory of the gradient descent method for different step sizes

# Generate banana function

N = 100 # Resolution of 2D image
x0 = -0.2
x1 = 1.2
y0 = 0
y1 = 1.2
xs = numpy.linspace(x0, x1, N)
ys = numpy.linspace(y0, y1, N)
dat = numpy.zeros((N, N))

for ix, x in enumerate(xs):
    for iy, y in enumerate(ys):
        r = [x,y]
        dat[iy, ix] = f(r)

plt.figure(figsize=(8,8))
im = plt.imshow(dat, extent=(x0, x1, y0, y1), origin='lower', cmap=matplotlib.cm.gray, 
                norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=200))
plt.colorbar(im, orientation='vertical', fraction=0.03925, pad=0.04)

# Now generate the trajectories:
gammas = [0.004, 0.003, 0.002]  # Gammas to try out
r0 = numpy.array([0.2, 1])  # Initial seed point

for gamma in gammas:
    
    output = gradientDescent(grad, r0, gamma, N)
    
    xs = output[: ,0]
    ys = output[: ,1]

    plt.plot(xs,ys, label=f"step-size={gamma}")

plt.legend()
plt.show()
