import numpy as np
import matplotlib.pyplot as plt

#function to be integrated (in this case x^2cos(2x))

def f(x):
    return (x**2)*np.cos(2*x)

#analytical integral of f(x) for comparison

def g(x):
    return (1/4)*((2*x**2-1)*np.sin(2*x)+2*x*np.cos(2*x))

#analytical integration of f(x) from xmin to xmax

def integrate_analytic(xmin, xmax):

    return (g(xmax)-g(xmin))


#numerical integration of f(x) from xmin to xmax using Simpson's rule with N panels

def integrate_numeric(xmin, xmax, N):

    deltax=(xmax-xmin)/N
    
    ytotal = f(xmin)+f(xmax)
    
    for i in range (0,N):
        
        ytotal += 4*(f(0.5*((xmin + i*deltax)+(xmin + (i+1)*deltax))))
        
        ytotal += 2*(f(xmin + (deltax*i)))
    
    return (deltax/6)*(ytotal)

print(integrate_numeric(xmin=0, xmax=4, N=1))


#log-log plot of fractional error in numerical integration vs number of panels

x0, x1 = 0, 2  # Bounds to integrate f(x) over
panel_counts = [4, 8, 16, 32, 64, 128, 256, 512, 1024]  # Panel numbers to use
result_analytic = integrate_analytic(x0, x1)  # Define reference value from analytical solution

error_values = []

for N in panel_counts:
    result_numerical = integrate_numeric(x0,x1,N)
    fractional_error = abs((result_numerical-result_analytic)/result_analytic) 
    error_values.append(fractional_error)
    
plt.loglog(panel_counts, error_values,'o')
plt.xlabel('Number of Panels (N)')
plt.ylabel('Fractional Error')
plt.title('Error in Numerical Integration vs Number of Panels (Logarithmic)')
plt.show()