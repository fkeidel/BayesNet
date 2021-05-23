import matplotlib.pyplot as plt
import numpy as np

file = "C:\BayesNet\examples\grid_sampling\plots\Exact.csv"
img_array = np.loadtxt(file, skiprows=1, delimiter=',')

print(img_array)

imgplot = plt.imshow(img_array, vmin=0, vmax=1)
plt.colorbar()
plt.xticks(np.arange(0,4,1))
plt.yticks(np.arange(0,4,1))
plt.show()