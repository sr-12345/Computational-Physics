import numpy as np
from matplotlib import pyplot as plt 

# We implement a walker class. When initialised a list of possible steps is populated. In one dimension it is

# [+s] , [-s] 

# where s is the step size, it defaults to 1 but can be set as an argument in the constructor. 
# In two dimensions the steps list contains

# [ +s , 0 ] , [ -s , 0 ] ,  [ 0 , +s ] , [ 0 , -s ]

# At each step the current position of the walker, saved in `self.pos`, is updated by adding one of the possible steps. 
# The function `pickStep` chooses randomly one of the possible steps. 
# This function is used to implement the `doSteps` function that performs `n` steps and returns a `(n+1) x ndim` 
# array representing the trajectory of the walker, including the starting point. 


class walker:
    def __init__(self,x0,ndim=1, step_size=1.0):
        self.pos=x0
        self.ndim=ndim
        self.possibleSteps=[]
        for i in range(ndim):
            step=numpy.zeros(ndim)
            step[i]= - step_size
            self.possibleSteps.append(numpy.array(step,dtype='f'))
            step[i]= + step_size
            self.possibleSteps.append(step.copy())
        self.npossible=len(self.possibleSteps)

    def pickStep(self):
        istep = numpy.random.choice(range(self.npossible))
        return self.possibleSteps[istep]
        
    def doSteps(self,n):
        positions=numpy.ndarray((n+1,self.ndim),dtype='f')
        
        positions[0]=self.pos
        
        for i in range(n):
            step = self.pickStep()
            self.pos = self.pos + step
            positions[i+1]= self.pos
            
            
        return positions
    

numpy.random.seed(1111)
w = walker(numpy.zeros(1))
pos_test = w.doSteps(10)
reference = [[ 0.], [-1.], [ 0.], [ 1.], [ 2.], [ 1.], [ 0.], [-1.], [-2.], [-3.], [-4.]]
assert len(pos_test)==11
# plots to help debugging
plt.plot(range(11),pos_test, label='your trajectory')
plt.plot(range(11),reference,'o', label='reference points')
plt.legend()
plt.xlabel('step number')
assert (pos_test == reference).all()

numpy.random.seed(1112)
w = walker(numpy.zeros(2), ndim=2)
pos_test = w.doSteps(10)
reference = numpy.array([[ 0.,  0.], [-1.,  0.], [-1., -1.], [-2., -1.], 
             [-2.,  0.], [-2.,  1.], [-1.,  1.], [-1.,  2.], 
             [ 0.,  2.], [ 1.,  2.], [ 0.,  2.]])
assert pos_test.shape == (11,2)
# plots to help debugging
plt.plot(pos_test[:,0], pos_test[:,1],'-o', label='your trajectory')
plt.plot(reference[:,0],reference[:,1],'o', label='reference points')
plt.legend()
assert (pos_test == reference).all()


# plot to help visualise the trajectories of 10 walkers wakjiung 100 steps

nsteps = 100
for i in range(10):
    w = walker(numpy.zeros(1))
    ys = w.doSteps(nsteps)
    plt.plot(range(nsteps+1),ys)


# plot of average position and average squared position of 100 1D walkers using 1000 steps

walkers = 100
steps = 1000

all_positions=np.zeros((walkers, steps+1))

for i in range(walkers):
    walker_current = walker(np.zeros(1))
    path = walker_current.doSteps(steps)
    all_positions[i]=path[:,0]

squaredpos=all_positions**2
avgsquared=np.mean(squaredpos, axis = 0)
avgpos=np.mean(all_positions, axis = 0)

xs = np.arange(0,steps+1,1)

plt.plot(xs, avgpos, label='Average Position')
plt.plot(xs, avgsquared, label ='Average Squared Position')
plt.xlabel('Step')
plt.ylabel('Position')
plt.title('Average Position vs Average Squared Position for 1D Walkers')
plt.legend()
plt.show()


# plot to show that the average squared distance scaling is independent of the dimension in which the walker moves

steps=100
walkers=400
dims=[1,2,3,4]

for dimensions in dims:
    all_pos_sqrd=np.zeros((walkers, steps+1))
    
    for i in range(walkers):
        walker_current = walker(np.zeros(dimensions), ndim=dimensions)
        path = walker_current.doSteps(steps)
        dis_squared = np.sum(path**2, axis=1)
        
        all_pos_sqrd[i]=dis_squared

    average_pos_sqrd=np.mean(all_pos_sqrd, axis=0)
    
    plt.plot(average_pos_sqrd, label= f"Dimension = {dimensions}")

plt.title("Average Squared Distance vs Step Number in Many Dimensions")
plt.xlabel("Steps")
plt.ylabel("Average Squared Distance")
plt.legend()
plt.show()
        


# Using 1000 walkers randomly distributed in the unit square (the positions are given in the array rand_pos) 
# to simulate the diffusion of particles with step size 0.05. 
# Plot shows the position of the walkers after 10, 100 and 500 steps. 

ndim = 2
nwalkers = 500

stepsize = 0.05
steps_toplot=[10,100,500]

rand_pos = numpy.random.uniform(size=(nwalkers, ndim))
colours = ['red','green', 'blue']
plt.figure(figsize=(18,6))

for i in range(3):
    no_of_steps=steps_toplot[i]
    new_position=np.zeros((nwalkers,ndim))
    
    for n in range(nwalkers):
        currentwalker= walker(rand_pos[n], ndim=ndim, step_size=stepsize)
        path=currentwalker.doSteps(no_of_steps)
        new_position[n]=path[-1]
    
    rand_pos = new_position
 
    plt.subplot(1, 3, i+1)
    plt.title(f"Positions after {no_of_steps} steps")
    plt.xlim((-3, 4))
    plt.ylim((-3, 4))
    plt.scatter(rand_pos[:,0], rand_pos[:,1], color = colours[i], alpha=0.5)
    
    plt.xlabel("x")
    plt.ylabel("y")
    