import numpy as np
import matplotlib.pyplot as plt

#will generate and plot the decay curve for Carbon-14 analytically and numerically.  
#14C has a half life of 5730 years, can derive the mean lifetime tau from this

def meanLifetime(halfLife):
    return -halfLife/np.log(0.5)

T_HALF = 5730
TAU = meanLifetime(T_HALF)


def f_rad(N, t):
    return -N/TAU

#analytic solution to the differential equation for radioactive decay
#N0 is the initial number of attoms at time t=0, function returns nuclei count at time t

def analytic(N0, t):
    return N0*np.exp(-t/TAU)



#numerical solution to the differential equation using Euler's method

def solve_euler(f, n0, t0, dt, n_steps):
       
    count_initial = n0 
    
    counts_at_each_timestep = [count_initial]
    
    for i in range(0, n_steps):
        
        t = t0 + dt*(i+1)
        N_at_t = counts_at_each_timestep[i]
        dn_dt=f(N_at_t,t)
        count_next = N_at_t + dt*dn_dt
        counts_at_each_timestep.append(count_next)
    
    return counts_at_each_timestep


#numerical solution to the differential equation using Runge-Kutta 4th order method

def solve_RK4(f, n0, t0, dt, n_steps):
    
    t=[t0]
    N=[n0]
    
    for i in range(0,n_steps):
        
        t_current = t[i]
        N_current = N[i]
        
        k1 = f(N_current, t_current)
        k2 = f(N_current+(dt*k1)/2, t_current+dt/2)
        k3 = f(N_current+(dt*k2)/2, t_current+dt/2)
        k4 = f(N_current+dt*k3, t_current+dt)
         
        k = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        N_next = N_current + k*dt
        t_next = t_current + dt
        
        N.append(N_next)
        t.append(t_next)
        
    return N


#plotting to show the error scaling for RK4 and Eurler's method

def RK4_error(f, n0, t0, dt, n_steps):
    
    t=t0
    sum=0
    
    for i in range(0,n_steps):
        sum+= abs(solve_RK4(f, n0, t0, dt, n_steps)[i] - analytic(n0,t))
        t += dt
    
    average_error_rk4=sum/n_steps
    
    return average_error_rk4


def Euler_error(f, n0, t0, dt, n_steps):
    
    t=t0
    sum=0
    
    for i in range(0,n_steps):
        sum+= abs(solve_euler(f, n0, t0, dt, n_steps)[i] - analytic(n0,t))
        t += dt
    
    average_error_euler=sum/n_steps
    
    return average_error_euler



#choose these parameters
n_of_steps = [10,20,40,80,160,320,640,1280,2560]
t_end = 15000
n0 = 10000
t0 = 0

RK4Errors = []
EulerErrors = []

for i in range(0,len(n_of_steps)):
    
    dt = t_end/n_of_steps[i]
    
    RK4Errors.append(RK4_error(f_rad, n0, t0, dt, n_of_steps[i]))
    EulerErrors.append(Euler_error(f_rad, n0, t0, dt, n_of_steps[i]))
    
plt.loglog(n_of_steps, RK4Errors, 'o-', label = 'RK4 Method')
plt.loglog(n_of_steps, EulerErrors, 'o-', label = 'Euler Method')
plt.xlabel('Number of Steps')
plt.ylabel('Error Between Numerical Method and Analytic Solutionn', fontsize = 8)
plt.title('Comparison of Error in Numerical Solutions to First Order ODEs Against Step Size (Logarithmic)', fontsize = 8)
plt.legend()
plt.show()