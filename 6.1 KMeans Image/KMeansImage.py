import pdb
from skimage import img_as_float
from skimage.io import imread, imsave
import pylab
from math import log10
from sklearn.cluster import KMeans
from createimg import create_img2, createx

image = imread('parrots.jpg')
img = img_as_float(image)

x = createx(img)

psnr_list_mean = []
psnr_list_median = []

for c in range(8, 20):
    clf = KMeans(init='k-means++', random_state=241, n_clusters=c)
    y = clf.fit_predict(x)
    img2 = create_img2(x, y, 'mean')
    img3 = create_img2(x, y, 'median')
    x2 = createx(img2)
    x3 = createx(img3)
    mse2 = 0
    mse3 = 0
    for n, i in enumerate(x):
        mse2 += abs(x[n] - x2[n]) ** 2
        mse3 += abs(x[n] - x3[n]) ** 2
    mse2 = sum(mse2) / (3 * len(x))
    mse3 = sum(mse3) / (3 * len(x))
    psnr2 = 20*log10(1.0) - 10*log10(mse2)
    psnr3 = 20*log10(1.0) - 10*log10(mse3)
    psnr_list_mean.append(psnr2)
    psnr_list_median.append(psnr3)
    print(c, psnr2, psnr3)

pdb.set_trace()

print(psnr_list_mean)
print(psnr_list_median)

pdb.set_trace()

imsave('parrots_mean' + str(c) + '.jpg', img2)
imsave('parrots_median' + str(c) + '.jpg', img3)

imsave('parrots_median.jpg', img2)

pdb.set_trace()
pylab.imshow(img2)
pylab.show()


pdb.set_trace()
