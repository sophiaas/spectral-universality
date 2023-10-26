import matplotlib.pyplot as plt
import numpy as np


save_dir = '/home/giovanni/Desktop/spectral-universality/plots/dihedral_3.png'

# matr = np.random.randint(low=0, high=5, size=(5,5))

#cyclic 5
# matr = np.array([[0., 1., 2., 3., 4.],
#         [1., 2., 3., 4., 0.],
#         [2., 3., 4., 0., 1.],
#         [3., 4., 0., 1., 2.],
#         [4., 0., 1., 2., 3.]])

#cyclic 6:
# matr = np.array([[0., 1., 2., 3., 4., 5.],
#         [1., 2., 3., 4., 5., 0.],
#         [2., 3., 4., 5., 0., 1.],
#         [3., 4., 5., 0., 1., 2.],
#         [4., 5., 0., 1., 2., 3.],
#         [5., 0., 1., 2., 3., 4.]])

#dihedral 3
# matr = np.array([[0., 1., 2., 3., 4., 5.],
#         [1., 2., 0., 4., 5., 3.],
#         [2., 0., 1., 5., 3., 4.],
#         [3., 5., 4., 0., 2., 1.],
#         [4., 3., 5., 1., 0., 2.],
#         [5., 4., 3., 2., 1., 0.]])

#cyclic_2_cubed
# matr = np.array([[0., 1., 2., 3., 4., 5., 6., 7.],
#         [1., 0., 3., 2., 5., 4., 7., 6.],
#         [2., 3., 0., 1., 6., 7., 4., 5.],
#         [3., 2., 1., 0., 7., 6., 5., 4.],
#         [4., 5., 6., 7., 0., 1., 2., 3.],
#         [5., 4., 7., 6., 1., 0., 3., 2.],
#         [6., 7., 4., 5., 2., 3., 0., 1.],
#         [7., 6., 5., 4., 3., 2., 1., 0.]])


# plt.figure()
plt.matshow(matr, cmap='magma', alpha=.9)
plt.axis('off')


# plt.show()  
plt.savefig(save_dir, bbox_inches='tight')