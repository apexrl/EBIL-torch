import numpy as np                                                              
import seaborn as sns                                                           
from scipy import stats                                                         
import matplotlib.pyplot as plt        
import math

mu = 0
variance = 1
labels=['density', 'energy, Z=10', 'energy, Z=20', 'energy, Z=30']

sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
px = stats.norm.pdf(x, mu, sigma)

plt.plot(x, px, 'b', label='density')
z = 10
ex = -np.log(px*z)
plt.plot(x, ex, 'r', label='energy, Z=10')
z = 20
ex = -np.log(px*z)
plt.plot(x, ex, 'g', label='energy, Z=20')
z = 30
ex = -np.log(px*z)
plt.plot(x, ex, 'y', label='energy, Z=30')

plt.xlabel('x', fontsize="x-large")
plt.ylabel('y', fontsize="x-large")

plt.legend(labels, ncol=1, loc="best", fontsize="x-large", frameon = False)

plt.savefig('energy-example.pdf', bbox_inches = 'tight', pad_inches = 0)

plt.show()
