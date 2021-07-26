import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(-15, 15, 4000)

y1 = x + np.random.normal(0, 0.2, len(x)) 
y2 = -x + np.random.normal(0, 0.2, len(x)) 

data = [(x[i], y1[i]) for i in range(len(x))] + [(x[i], y2[i]) for i in range(len(x))] 
data = np.array(data)

data_dict = {'data':data}

# joblib.dump(data_dict, './demos/pm_x.pkl')

x = np.linspace(0, 15, 4000)

y1 = x[:int(len(x)/2)] + np.random.normal(0, 0.2, int(len(x)/2))
y2 = -x[int(len(x)/2):] + np.random.normal(0, 0.2, int(len(x)/2)) + 15
y3 = np.zeros(len(x)) + np.random.normal(0, 0.2, len(x))

data = [(x[i], y1[i]) for i in range(int(len(x)/2))] + [(x[i+int(len(x)/2)], y2[i]) for i in range(int(len(x)/2))]# + [(x[i], y3[i]) for i in range(len(x))] 
data = np.array(data)

data_dict = {'data':data}

sns.set()
fig,ax=plt.subplots(figsize=(6,6))
plt.scatter(data[:,0], data[:,1])
plt.xlim(-1.25, 20)
plt.ylim(-1.25, 10)
# plt.grid()
ax.set_yticks([0, 5, 10])
ax.set_xticks([0, 5, 10, 15, 20])
plt.savefig('./figs/expert_triangle.pdf', bbox_inches='tight')

# joblib.dump(data_dict, './demos/pm_triangle.pkl')

x = np.linspace(0, 15, 1000)

y1 = np.linspace(0, 15, 1000) + np.random.normal(0, 0.2, 1000)
y2 = np.zeros(1000) + np.random.normal(0, 0.2, 1000) + 15
y3 = np.linspace(0, 15, 1000) + np.random.normal(0, 0.2, 1000)
y4 = np.zeros(1000) + np.random.normal(0, 0.2, 1000)

x1 = np.random.uniform(0,1, 1000)
x2 = x
x3 = np.random.uniform(14,15, 1000)
x4 = x

data = [(x1[i], y1[i]) for i in range(int(len(x1)))] + [(x2[i], y2[i]) for i in range(int(len(x2)))] + [(x3[i], y3[i]) for i in range(len(x3[500:]))]  
data = np.array(data)

data_dict = {'data':data}

# joblib.dump(data_dict, './demos/pm_square.pkl')
