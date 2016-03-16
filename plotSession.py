import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import scipy.io as sio

data = sio.loadmat('ValidationData/Data_0000020.mat')

# Add BBox
[X, Y, W, H] = data['bbox'][0]
def drawBox():
    ax = plt.gca()
    ax.add_patch(pat.Rectangle((X,Y), W, H, alpha=1, facecolor='none', edgecolor='green'))

plt.figure(1)
plt.subplot(221)
plt.imshow(data['lbl'])
drawBox()
plt.title('Label')
plt.subplot(222)
plt.imshow(data['depth'])
drawBox()
plt.title('Depth')
plt.subplot(223)
plt.imshow(data['segmap'])
drawBox()
plt.title('Segmentation')
plt.subplot(224)
plt.imshow(np.sum(data['hmap'], axis=2))
drawBox()
plt.title('Finger Heatmap')

plt.show()
