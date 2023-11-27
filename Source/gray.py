import numpy as np
import matplotlib.pylap as plt
from scipy import misc

img_rgb = misc.face() # image load
print (img_rgb.shape)
width, height = img_rgb.shape[:2]
print (width, height)

plt.subplot(221)
plt.imshow(img_rgb, cmap=plt.cm.gray)
plt.axis('off')
plt.title('RGB image')
