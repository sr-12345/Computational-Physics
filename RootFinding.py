import numpy
from matplotlib import pyplot as plt


# Three methods to find a zero of a function.
# The functions will return the list of all approximations to the zero the algorithm went through, including the initial value.

# Newton-Raphson method, stops iterating when | x_{i+1} - x_{i} | < tolerance 
# returns an array with all the x_i values. x_0 is the starting point.

def NewtonRaphson(f, df, x0, tolerance):

    history = [x0]
    
    while True:
        xcurrent = history[-1]
        xnext = xcurrent - f(xcurrent)/df(xcurrent)
        history.append(xnext)
        
        if abs(xnext-xcurrent)<tolerance:
            break
    
    return history


# Bisection method, begins with two initial values which must have opposite signs
# The method iteratively bisects the interval and chooses the subinterval in which the function changes sign.

def bisect(f,x0,x1,tolerance):
   
    if f(x0)*f(x1)>0:
        print("x0 and x1 should have opposite sign")
    history = []
    max_iterations = 10000

    for i in range(max_iterations):
        if abs(x0-x1)>tolerance:
            m = (x0+x1)/2
            history.append(m)
        
            if f(m)*f(x0)>0:
                x0=m

            else:
                x1=m
            
        else:
            break

    return history       


# Secant method, similar to Newton-Raphson but does not require the derivative of the function.
# Termination condition is | (x_{i+1}-x{i})/(x_{i+1}+x_{i}) | < tolerance

def secant(f,x0,x1,tolerance,returnPoints=False):

    xa=x0
    xb=x1
    xs=[x0, x1]
    vs=[f(x0), f(x1)]
    while True:
        if abs((xs[-1] - xs[-2])/(xs[-1] + xs[-2])) < tolerance:
            return xs
        df=(vs[-1]-vs[-2])/(xs[-1]-xs[-2])
        
        xnew = xs[-1] - vs[-1]/df
        xs.append(xnew)
        vs.append(f(xnew))


# We are now going to look at the three methods and see how they compare using the equation
# $$ x-\tanh(2x)=0$$

def f(x): 
    return x-numpy.tanh(2*x)

def df(x): 
    return 1.0-2.0/numpy.cosh(2*x)**2

xmin, xmax = -1.2, 1.2
xs = numpy.linspace(xmin, xmax, 300)
plt.plot(xs, f(xs))
plt.hlines(0, xmin, xmax);

# This section will produce the plot to show the convergence of the different methods. 
# There is nothing to do beyond evaluating it.

plt.figure(figsize=(12,6))
    
target = 0.9575040240772687

val_plot = plt.subplot(121)
err_plot = plt.subplot(122)

def makePlot(xs,label):
    ns=range(len(xs))    
    val_plot.plot(ns, xs,'o')
    reldiff = abs((numpy.array(xs) - target)/target)
    err_plot.plot(ns, reldiff, 'o-', label=label)
    
    
xs = NewtonRaphson(f, df, 0.6, 1e-13)
makePlot(xs, 'NR')
xs = secant(f, 1.0, 0.5, 1e-13)
makePlot(xs, 'sec')
xs = bisect(f, 1.0, 0.6, 1e-6)
makePlot(xs, 'bisec')

val_plot.set_xlabel('iteration nbr')
val_plot.set_ylabel('$x_i$')
err_plot.set_yscale('log')
err_plot.legend()
err_plot.set_xlabel('iteration nbr')
err_plot.set_ylabel('relative error');



# Now that we have a function to find zeros numerically we can return to the generation of the numbers 
# according to a given distribution function. We consider the function 

# $$ f(x) = \mathcal{N}\left(1 + \frac{2}{1 + x^2} + \sin(\sqrt{3 x})^2\right)$$

#with $\mathcal{N}$ such that $f$ is a probability distribution. The function can be integrated to give its primitive:

#$$ g(x) = \mathcal{N}\left(\frac{1}{12}+\frac{3 x}{2} + 2 \arctan{x} - \frac{1}{12} \cos(2 \sqrt{3x}) - \frac{\sqrt{x} \sin(2 \sqrt{3x})}{2 \sqrt{3}}\right) $$


norm = (181 + 24*numpy.arctan(10)-numpy.cos(2*numpy.sqrt(30))-2*numpy.sqrt(30)*numpy.sin(2*numpy.sqrt(30)))/12

def pdf(x):
    return  (1 + (2/(1+x**2)+ numpy.sin(numpy.sqrt(3*x))**2))/norm

def cumulative(x):
    return  (1./12.+(3*x)/2.0 + 2*numpy.arctan(x) - numpy.cos(2*numpy.sqrt(3.0*x))/12. - (numpy.sqrt(x)*numpy.sin(2*numpy.sqrt(3*x)))/(2.*numpy.sqrt(3)))/norm


xs = numpy.linspace(0, 10, 200)
cs = cumulative(xs)
plt.plot(xs, cs);
plt.xlabel('x')
plt.ylabel('g(x)')
plt.xlim(0,10)
plt.ylim(0,1);


# Function that returns values distributed according to pdf(x) given a set of values `xis`  
# uniformly distributed between $0$ and $1$


import random

def generate(xis):
    sample = []
    
    for xi in xis:
        
        def rootfunction(x):
            return cumulative(x) - xi
        
        root = bisect(rootfunction,0,10,1e-10)[-1]
        sample.append(root)

    return sample


# plot of the distribution of the generated values

numpy.random.seed(121314)
xis = numpy.random.random(10000)

xs = generate(xis)
plt.hist(xs, bins=50, density=True);

xs = numpy.linspace(0, 10, 200)
fs = pdf(xs)
plt.plot(xs, fs, 'k--');