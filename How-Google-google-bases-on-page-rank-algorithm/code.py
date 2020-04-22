# --------------
# Code starts here

import numpy as np

# Code starts here

# Adjacency matrix
adj_mat = np.array([[0,0,0,0,0,0,1/3,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                   [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                  [0,0,1/2,1/3,0,0,1/3,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/3,0]])

# Compute eigenvalues and eigencevectrs
eigenvalues,eigencevectors  = np.linalg.eig(adj_mat)

eigen_first = eigencevectors[:,0]
eigen_1 = abs(eigen_first)/(np.linalg.norm(eigen_first,1))
max_idx = -1
maxi = eigen_1[0]
print(eigen_1)
for i in range(1,len(eigen_1)):
    if eigen_1[i]>maxi:
        maxi= eigen_1[i]
        max_idx = i
page = max_idx + 1
print(page)


# Eigen vector corresponding to 1


# most important page


# Code ends here


# --------------
# Code starts here

# Initialize stationary vector I
init_I = np.array([1,0,0,0,0,0,0,0]
)

# Perform iterations for power method
for i in range(10):
    init_I = np.dot(adj_mat,init_I)
    init_I = (init_I)/ (np.linalg.norm(init_I,1))

print(init_I) 
max_idx = -1
maxi = init_I[0]

for i in range(1,len(init_I)):
    if init_I[i]>maxi:
        maxi= init_I[i]
        max_idx = i
power_page = max_idx + 1
print(page)




# Code ends here


# --------------
# Code starts here

# New Adjancency matrix
# New Adjancency matrix
new_adj_mat = np.array([[0,0,0,0,0,0,0,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                  [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,0,1/2,1/3,0,0,1/2,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/2,0]])

# Initialize stationary vector I
new_init_I = np.array([1,0,0,0,0,0,0,0]
)

# Perform iterations for power method
for i in range(10):
    new_init_I = np.dot(new_adj_mat,new_init_I)
    new_init_I = (new_init_I)/ (np.linalg.norm(new_init_I,1))

print(new_init_I) 
max_idx = -1
maxi = new_init_I[0]

for i in range(1,len(new_init_I)):
    if new_init_I[i]>maxi:
        maxi= new_init_I[i]
        max_idx = i
new_power_page = max_idx + 1




# Code ends here


# --------------
# Alpha value
alpha = 0.85

# Code starts here

# Modified adjancency matrix
G = (alpha*new_adj_mat) + (1-alpha)*(1 / len(new_adj_mat))*np.ones(new_adj_mat.shape)

# Initialize stationary vector I
final_init_I = np.array([1,0,0,0,0,0,0,0])

# Perform iterations for power method
for i in range(1000):
    final_init_I = np.dot(G,final_init_I)
    final_init_I = final_init_I/(np.linalg.norm(final_init_I,1))

print(final_init_I)

# Code ends here


