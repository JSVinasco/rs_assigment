# get familiar with rasterio

# import necesary librarys
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import otbApplication as otb

###################################
###     PART 1 PEPS SEN2 DATA   ###
###################################

# set location of the data and code
path_code = '/home/juan/Documentos/cesbio/assigments/Code/'
path_data = '/home/juan/Documentos/cesbio/assigments/Data/PEPS/S2A_MSIL1C_20200507T104031_N0209_R008_T31TDG_20200507T124549.SAFE/GRANULE/L1C_T31TDG_A025459_20200507T104558/IMG_DATA/'

os.chdir(path_code)

list_img = os.listdir(path_data)

# read a sample Data

dataset = rasterio.open(path_data+list_img[0])
dataset = dataset.read(1)

# display

plt.imshow(dataset, cmap='gray')
plt.show()

# read all bands of the sentinel data
# if necesary divide the data using the differents resolution
# of the MSI data

# see the posible bands
#for i in range(len(list_img)):
#    print(list_img[i][24:26])

# select the differents resolutions

# 10 meters resolution
m10 = ['02', '03', '04']
list_bands_10 = []
for i in range(len(list_img)):
    for j in range(len(m10)):
        if list_img[i][24:26]== m10[j]:
            list_bands_10.append(list_img[i])

# 20 meters resolution
m20 = ['05', '06', '07','8A', '11','12']
list_bands_20 = []
for i in range(len(list_img)):
    for j in range(len(m20)):
        if list_img[i][24:26]== m20[j]:
            list_bands_20.append(list_img[i])

# 60 meters resolution
m60 = ['01', '09', '10']
list_bands_60 = []
for i in range(len(list_img)):
    for j in range(len(m60)):
        if list_img[i][24:26]== m60[j]:
            list_bands_60.append(list_img[i])

# now read the data group by the spatial resolution

data_10m = []
np_array_10m = []
data_20m = []
np_array_20m = []
data_60m = []
np_array_60m = []

#########################################################
#                   10 m

for i in range(len(list_bands_10)):
    dataset = rasterio.open(path_data+list_bands_10[i])
    array = dataset.read(1)
    data_10m.append(dataset)
    np_array_10m.append(array)

s2a_10m = np.asarray(np_array_10m)
s2a_10m.shape
s2a_10m = np.reshape(s2a_10m, (s2a_10m.shape[2],s2a_10m.shape[1],s2a_10m.shape[0]))

s2a_rgb = np.stack((s2a_10m[:,:,0],s2a_10m[:,:,1],s2a_10m[:,:,2]),axis=2)


#plot_img = (s2a_10m[100:150,100:150,3])

plt.imshow(np.float32(s2a_rgb[9585:9635,3715:3765,2]))
plt.show()

plt.imshow(np.float32(s2a_10m[9585:9635,3715:3765,:]))
plt.show()

#######################################################
#                   20 m

for i in range(len(list_bands_20)):
    dataset = rasterio.open(path_data+list_bands_20[i])
    array = dataset.read(1)
    data_20m.append(dataset)
    np_array_20m.append(array)

s2a_20m = np.asarray(np_array_20m)
s2a_20m.shape
s2a_20m = np.reshape(s2a_20m, (s2a_20m.shape[2],s2a_20m.shape[1],s2a_20m.shape[0]))

plt.imshow(np.float32(s2a_20m))#, interpolation='none')
plt.show()

#######################################################
#                   60 m

for i in range(len(list_bands_60)):
    dataset = rasterio.open(path_data+list_bands_60[i])
    array = dataset.read(1)
    data_60m.append(dataset)
    np_array_60m.append(array)

s2a_60m = np.asarray(np_array_60m)
s2a_60m.shape
s2a_60m = np.reshape(s2a_60m, (s2a_60m.shape[2],s2a_60m.shape[1],s2a_60m.shape[0]))

plt.imshow(np.float32(s2a_60m))#, interpolation='none')
plt.show()











#
