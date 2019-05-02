import numpy as np

# %matplotlib inline

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

np.set_printoptions(suppress=True,precision=3)

from matplotlib.patches import FancyArrowPatch

num_instances=6
X_men=np.array([[1.97,110,5],[1.80,70,4.8],[1.70,90,4.9]]).transpose()
X_women=np.array([[1.65,52,4.7],[1.75,65,4.8],[1.67,58,4.6]]).transpose()

X = np.hstack((X_men,X_women))
print (X)

print('\n')

mean=np.mean(X,axis=1) # Observar que tomamos la media de cada fila, por eso axis=1
std=np.std(X,axis=1)
var=np.var(X,axis=1)

print ("Media:",mean)
print ("Varianza:",var)
print ("Desviación estándar:", std)

print('\n')

X_r = X - X.mean(axis=1,keepdims=True)
print (X_r)

print('\n')

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')
# plt.rcParams['legend.fontsize'] = 10   
# ax.plot(X_r[0,0:int(num_instances/2)], X_r[1,0:int(num_instances/2)], X_r[2,0:int(num_instances/2)], 'o', markersize=8, color='red', alpha=0.5, label='class1')
# ax.plot(X_r[0,int(num_instances/2):num_instances], X_r[1,int(num_instances/2):num_instances], X_r[2,int(num_instances/2):num_instances], 'o', markersize=8, color='blue', alpha=0.5, label='class2')

# plt.show()

print('\n')

mean=np.mean(X_r,axis=1)
std=np.std(X_r,axis=1,ddof=1)
var=np.var(X_r,axis=1,ddof=1)

print ("Means:",mean)
print ("Standard deviations:",std)
print ("Variances:",var)
# mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print('\n')

cvm=np.cov(X_r)
print (cvm)

print('\n')

eig_val_cov, eig_vec_cov = np.linalg.eig(cvm)

for i in range(len(eig_val_cov)):
    eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T

    print('Valor propio {} de la matriz de covarianza {}'.format(i+1, eig_val_cov[i]))
    print('Vector propio:')
    print(eigvec_cov)

print('\n')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i)
    #    print(i[0])

print('\n')

# Just to principal components
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
print('Matrix W:\n', matrix_w)

print('\n')

# Transform instance to the new subspace

transformed = np.dot(X_r.T,matrix_w).transpose()
print (transformed)

print('\n')

# plt.plot(transformed[0,0:int(num_instances/2)], transformed[1,0:int(num_instances/2)], 'o', markersize=7, color='blue', alpha=0.5, label='men')
# plt.plot(transformed[0,int(num_instances/2):num_instances], transformed[1,int(num_instances/2):num_instances], '^', markersize=7, color='red', alpha=0.5, label='women')
# #plt.xlim([-4,4])
# #plt.ylim([-4,4])
# plt.xlabel('x_values')
# plt.ylabel('y_values')
# #plt.legend()
# plt.title('Instancias transformadas (con etiquetas)')

# plt.show()

print('\n')

# Just one dimension 
matrix_w2 = np.hstack((eig_pairs[0][1].reshape(3,1),))
print('Matrix W:\n', matrix_w2)

print('\n')

transformed2 = np.dot(X_r.T,matrix_w2).transpose()
print (transformed2)

print('\n')

plt.plot(transformed2[0,0:int(num_instances/2)],np.zeros(int(num_instances/2)), 'o', markersize=7, color='blue', alpha=0.5, label='hombres')
plt.plot(transformed2[0,int(num_instances/2):num_instances],np.zeros(int(num_instances/2)), '^', markersize=7, color='red', alpha=0.5, label='mujeres')
#plt.xlim([-4,4])
#plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Instancias transformadas (con etiquetas)')

plt.show()
