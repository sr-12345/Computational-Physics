import matplotlib.pyplot as plt
import numpy
import random

# In this part of the script we create a function to generate values of $x$ between 0 and 10 distributed according to 
# $$ \frac{1}{\mathcal{N}} \left( 1 + \frac{2}{1+x^2}+ \sin(\sqrt{3 x})^2\right) $$
# with 
# $$ \mathcal{N} = \int\limits_0^{10} f(x) dx \;.$$

norm = (181 + 24*numpy.arctan(10)-numpy.cos(2*numpy.sqrt(30))-2*numpy.sqrt(30)*numpy.sin(2*numpy.sqrt(30)))/12

def f(x):
    return  (1 + (2/(1+x**2)+ numpy.sin(numpy.sqrt(3*x))**2))/norm


# plot of the function

xs = numpy.linspace(0, 10, 200)
fs = f(xs)
plt.plot(xs, fs);
plt.ylabel('f(x)')
plt.xlabel('x')
plt.xlim(0,10)
plt.ylim(0,0.25);


#function genSample generates a sample of npts values 𝑥 distributed according to 𝑓(𝑥)

def genSample(npts):
    sample = []
    
    fmax = 0.2
    
    while len(sample)<npts:
        xpoint = random.random()*10
        ypoint = random.random()*fmax
        
        if ypoint< f(xpoint):
            sample.append(xpoint)
    

    return numpy.array(sample)


# plot to show the number of points has been generated according to the function f(x)

plt.figure(figsize=(12,6))
npts = 100000
xs = numpy.linspace(0,10,100)
plt.plot(xs, f(xs), label = 'True Function f(x)')
plt.hist(genSample(npts), bins=50, density=True, label = 'Monte Carlo Distribution')
plt.ylabel('f(x)')
plt.xlabel('x')
plt.legend()
plt.title('Distribution According to the Function f(x)');
