import numpy
from PIL import Image
import matplotlib.pyplot as plt

pos_img = numpy.array(Image.open('data_set/uiuc/TestImages/test-3.pgm').convert('L'))
neg_img = numpy.array(Image.open('data_set/caltech/testing/testing-171.jpg').convert('L'))

fig1, ax1 = plt.subplots(nrows=1, ncols=2)
ax1[0].imshow(pos_img).set_cmap("gray")
ax1[0].get_xaxis().set_ticks([])
ax1[0].get_yaxis().set_ticks([])
ax1[0].set_xlabel('test-3.pgm')

ax1[1].imshow(neg_img).set_cmap("gray")
ax1[1].get_xaxis().set_ticks([])
ax1[1].get_yaxis().set_ticks([])
ax1[1].set_xlabel('test-171.jpg')

plt.savefig("sample_images.png", bbox_inches="tight")
plt.close(fig1)
