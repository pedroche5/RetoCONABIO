# Import Python 3's print function and division
from __future__ import print_function, division
from   sklearn  import neighbors


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


# Import GDAL, NumPy, and matplotlib
from osgeo import gdal, gdal_array
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Read the images with GDAL

img_ds_band2 = gdal.Open('../IMAGES/LANSAT8/LC08_L1TP_026047_20180110_20180119_01_T1_sr_band2.tif', gdal.GA_ReadOnly)
img_ds_band3 = gdal.Open('../IMAGES/LANSAT8/LC08_L1TP_026047_20180110_20180119_01_T1_sr_band3.tif', gdal.GA_ReadOnly)
img_ds_band4 = gdal.Open('../IMAGES/LANSAT8/LC08_L1TP_026047_20180110_20180119_01_T1_sr_band4.tif', gdal.GA_ReadOnly)
img_ds_band5 = gdal.Open('../IMAGES/LANSAT8/LC08_L1TP_026047_20180110_20180119_01_T1_sr_band5.tif', gdal.GA_ReadOnly)

roi_ds = gdal.Open('../IMAGES/Training/training.tif', gdal.GA_ReadOnly)

#Initialize the image shape
img = np.zeros((img_ds_band2.RasterYSize, img_ds_band2.RasterXSize, 4),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds_band2.GetRasterBand(1).DataType))

#Read the four bands of the tiff image
img[:, :, 0] = img_ds_band2.GetRasterBand(1).ReadAsArray()
img[:, :, 1] = img_ds_band3.GetRasterBand(1).ReadAsArray()
img[:, :, 2] = img_ds_band4.GetRasterBand(1).ReadAsArray()
img[:, :, 3] = img_ds_band5.GetRasterBand(1).ReadAsArray()

#Load the training data from the image

roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

# Display them
plt.subplot(121)
plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r) #Show the band 2 of the image
plt.title('SWIR1')

#Display the training data
plt.subplot(122)
plt.imshow(roi, cmap=plt.cm.Spectral)
plt.title('ROI Training Data')

plt.show()

# Find how many non-zero entries we have -- i.e. how many training data samples?
n_samples = (roi > 0).sum()
print('We have {n} samples'.format(n=n_samples))

# What are our classification labels?
labels = np.unique(roi[roi > 0])
print('The training data include {n} classes: {classes}'.format(n=labels.size, 
                                                                classes=labels))


#Print the matrix size of the image and the training data
print('Our img matrix is sized: {sz}'.format(sz=img.shape))
print('Our roi array is sized: {sz}'.format(sz=roi.shape))


#Initialize the feacture and the training data
X = img[:,:,:]
y = roi[roi > 0]

#Extra: cross validation with 5 splits

kf=KFold(n_splits=5, shuffle=True, random_state=2)

# Initialize our model with 500 trees
rf = RandomForestClassifier(n_estimators=10, oob_score=True)



#Split the feacture and training sample into 5 
for Train_index,Test_index in kf.split(X):
   X_Train, X_Test = X[Train_index],X[Test_index]
   y_Train, y_Test = y[Train_index],y[Test_index]
   print('Our X_Train is sized: {sz}'.format(sz=X_Train.shape))
   print('Our X_Test is sized: {sz}'.format(sz=X_Test.shape))
   
   print('Our y_Train is sized: {sz}'.format(sz=y_Train.shape))
   print('Our y_Test is sized: {sz}'.format(sz=y_Test.shape))
   
   nsamples, nx, ny = X_Train.shape
   d2_X_Train = X_Train.reshape((nsamples,nx*ny))
   print('Our d2_X_Train is sized: {sz}'.format(sz=d2_X_Train.shape))
   nsamples, nx, ny = X_Test.shape
   d2_X_Test  = X_Test.reshape((nsamples,nx*ny))
   rf.fit(d2_X_Train, y_Train) #Training the model
   

print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))





import pandas as pd


print('Our X_Test is sized: {sz}'.format(sz=X_Test.shape))
   




#Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
new_shape = (img.shape[2],  img.shape[1]* img.shape[0] )

img_as_array = img[:, :, :4].reshape(new_shape)
print('Reshaped from {o} to {n}'.format(o=img.shape,
                                        n=img_as_array.shape))

## Now predict for each pixel
class_prediction = rf.predict(img_as_array)
#class_prediction = rf.predict(img)

## Reshape our classification map
class_prediction = class_prediction.reshape(img[:, :, 0].shape)




#plt.subplot(122)
plt.imshow(class_prediction, cmap=cmap, interpolation='none')

plt.show()









