import numpy
import matplotlib.pyplot as plt

# The function integrate gives the Monte Carlo estimate for the volume of a unit ball in dim no. of dimensions. 
# To do so we choose random numbers in the dim-dimensional cube of sides  [−1,+1] that contains the unit ball. 
# Uses random numbers provided by numpy.random.uniform, they form a npoints dimensional array of dim coordinates in the cube.


def integrate(npoints, dim):
    # the random numbers
    rs = numpy.random.uniform(-1, 1, size=(npoints,dim)) 

    rs_squared = rs**2
    rs_squared_sum = numpy.sum(rs_squared, axis=1)
    rs_mag = rs_squared_sum**0.5

    accepted_vectors = 0
    
    for vector in rs_mag:
        
        if vector < 1:
            accepted_vectors+=1

    
    return 2**(dim) * (accepted_vectors/npoints)


# plot to show that the integration error scaling is independent of the dimension and that the error scales as  1/√𝑁
# uses known volumes of unit balls in 2,3,4,5,6 dimensions to calculate the error


ns = [2**ii for ii in range(4,20)]
dimensions = range(2,7)

useable_dimensions = numpy.array(dimensions)
volumes = [numpy.pi, 4/3*numpy.pi, 1/2*numpy.pi**2, 8/15*numpy.pi**2, 1/6*numpy.pi**3]

plt.figure(figsize = (12,6))

for dim,vols in zip(useable_dimensions,volumes):
    
    errors = []
    
    for n in ns:
        
        error = abs((vols-integrate(n, dim))/vols)
        errors.append(error)
     
    plt.loglog(ns, errors, label = f"{dim} Dimensions")

true_trend = 1/numpy.sqrt(ns)    

plt.plot(ns, true_trend, '--', label = r"$\frac{1}{\sqrt{N}}$")

plt.xlabel("Number of Points")
plt.ylabel("Error")
plt.title("Error Scaling of Monte Carlo Integration in Many Dimensions")
plt.legend()
plt.show()