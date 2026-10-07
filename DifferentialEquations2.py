import numpy as np
from matplotlib import pyplot as plt


# Script simulates the trajectory of a cannonball in the presence of drag

# Define all constants to be used in the simulation
r_cb = 0.15  # Radius of cannonball in m
rho_iron = 7874  # Density of iron in kg/m^3
g = 9.81  # Acceleration due to gravity in m/s^2
kappa = 0.47  # Drag coefficient of a sphere
rho_air = 1.23  # Density of air in kg/m^3
v0 = 125.00  # Initial speed in m/s


# cross sectional area and mass of the cannonball

def get_area(r):

    return np.pi*r**2

def get_mass(r):

    V = 4/3*np.pi*r**3
    return V* rho_iron

area_cb = get_area(r_cb)
mass_cb = get_mass(r_cb)


#Implements the differential equation for the cannonball's motion in the presence of drag

def f(state, t):
    
    # Unpack array of the state
    x, y, vx, vy = state
    
    dx_t, dy_dt, dvx_dt, dvy_dt = 0, 0, 0, 0
    
    
    v = np.array([vx,vy])
    magnitude_v = np.linalg.norm(v)
    
    a = np.array([0,-g])
    
    F_drag = -0.5*kappa*rho_air*area_cb*magnitude_v*v
    
    a_due_to_drag = F_drag/mass_cb
    
    a_total = a + a_due_to_drag
    
    dx_dt = vx
    dy_dt = vy
    
    dvx_dt = a_total[0]
    dvy_dt = a_total[1]
    
    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])


# solves the ODE using Euler's method from state_initial to end time t1 using n_steps steps
# state initial is an array with the initial conditions of the cannonball: [x0, y0, vx0, vy0]

def solve_euler(state_initial, t1, n_steps):
    
    t0 = 0
    dt = t1/n_steps
    t_values = np.linspace(t0,t1,n_steps+1)
    
    history = np.empty((n_steps+1, 4))
    history[0]= state_initial
    
    for i in range(0,n_steps):
        
        t =  t0 + dt*(i+1)
        prior_state = history[i]
        gradient = f(prior_state, t)
        next_state = prior_state + dt*gradient
        history[i+1]=next_state
        
    return history


# finds the x value where a linear function crosses zero given two points (x1,y1) and (x2,y2) on the line

def find_zero_linear(x1, x2, y1, y2):
    if y1*y2 > 0:
        print("I expect y1 and y2 to have opposite signs!")
    
    x_zero = x1 - y1*((x2-x1)/(y2-y1))
    
    return x_zero


# uses the above function to determine the range of the cannonball

def find_range(history):
    all_xs = history[:,0]
    all_ys = history[:,1]
    negatives = np.argwhere(all_ys<0)
    if len(negatives) == 0 :
        print ("The projectile did not touch down! Returning the last known location")
        return all_xs[-1]
    (index,) = negatives[0]
    y1, y2 = all_ys[index-1], all_ys[index]
    x1, x2 = all_xs[index -1], all_xs[index]
    return find_zero_linear(x1,x2,y1,y2)



#shows trajectories for different launch angles at a velocity of 125 m/s and drag effects


plt.figure(figsize=(12,6))

n_steps = 1000
thetas = range(5, 90, 5)


for theta in thetas:
    rad_thetas = np.radians(theta)

    vx0 = 125*np.cos(rad_thetas)
    vy0 = 125*np.sin(rad_thetas)

    state_initial = np.array([0,0,vx0,vy0])
    history = solve_euler(state_initial,30,n_steps)

    x_vals = history[:,0]
    y_vals = history[:,1]

    value_below_ground = 0

    for i in range (len(y_vals)):
        if y_vals[i]<0:
            value_below_ground = i
            break

    last_value_above= value_below_ground - 1

    plottable_x= []
    plottable_y= []

    for i in range(0, last_value_above+1):
        plottable_x.append(x_vals[i])
        plottable_y.append(y_vals[i])

    x_ground = find_range(history)
    plottable_x.append(x_ground)
    plottable_y.append(0)


    plt.plot(plottable_x,plottable_y, label = f"{theta}°")

plt.xlabel("x position (m)")
plt.ylabel("y position (m)")
plt.title("Trajectory for Varying Angle of Launch")
plt.legend(title = "Launch Angle")
plt.show()



# plot to show the range for different initial velocities with and without drag at a constant launch angle 

plt.figure(figsize=(12,6))

n_steps = 1000
max_time = 300
v0s = np.linspace(50, 1000, 20)

angle_deg = 60
angle_rad = np.radians(angle_deg)


#y_vals are range
#x_vals are just the initial velocities

#WITH DRAG

ranges=[]

for velocity in v0s:
    
    vx0 = velocity*np.cos(angle_rad)
    vy0 = velocity*np.sin(angle_rad)


    state_initial = np.array([0,0,vx0,vy0])
    history = solve_euler(state_initial,max_time,n_steps)
    x_range = find_range(history)
    
    
    ranges.append(x_range)
    
    
plt.plot(v0s,ranges,'o-', label= "with air resistance")




#WITHOUT DRAG

# need to create a function of state which is independent of drag

def f_no_drag(state, t):
    
    
    # Unpack array of the state
    x, y, vx, vy = state
    
     
    dx_t, dy_dt, dvx_dt, dvy_dt = 0, 0, 0, 0
    
    dx_dt = vx
    dy_dt = vy
    
    dvx_dt = 0
    dvy_dt = -g
    
    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])

#new euler for no drag

def solve_euler_no_drag(state_initial, t1, n_steps):
    
    
    t0 = 0
    dt = t1/n_steps
    t_values = np.linspace(t0,t1,n_steps+1)
    
    history_no_drag = np.empty((n_steps+1, 4))
    history_no_drag[0]= state_initial
    
    for i in range(0,n_steps):
        
        t =  t0 + dt*(i+1)
        prior_state = history_no_drag[i]
        gradient = f_no_drag(prior_state, t)
        next_state = prior_state + dt*gradient
        history_no_drag[i+1]=next_state
        
    return history_no_drag

#plotting section

ranges_no_drag=[]

for velocity in v0s:
    
    vx0 = velocity*np.cos(angle_rad)
    vy0 = velocity*np.sin(angle_rad)


    state_initial = np.array([0,0,vx0,vy0])
    history_no_drag = solve_euler_no_drag(state_initial,max_time,n_steps)
    x_range = find_range(history_no_drag)
    
    
    ranges_no_drag.append(x_range)
    
    
plt.plot(v0s,ranges_no_drag,'o-', label = "without air resistance")


plt.xlabel("Initial Velocity (ms$^{-1}$)")
plt.ylabel("Range (m)")
plt.title("Projectile Range for Varying Launch Velocity (60° Launch Angle)")
plt.legend()
plt.show()