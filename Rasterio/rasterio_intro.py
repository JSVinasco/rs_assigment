# get familiar with rasterio

# import necesary librarys
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
#import otbApplication as otb


# define functions
# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

# set location of the data and code
path_code = '/home/juan/Documentos/cesbio/assigments/Code/'
path_data = '/home/juan/Documentos/cesbio/assigments/Data/PEPS/S2A_MSIL1C_20200507T104031_N0209_R008_T31TDG_20200507T124549.SAFE/GRANULE/L1C_T31TDG_A025459_20200507T104558/IMG_DATA/'
path_data2 = '/home/juan/Documentos/cesbio/assigments/Data/Venus/VENUS_20190825-151920-000_L2A_COLOMBIA_D/VE_VM01_VSC_L2VALD_COLOMBIA_20190825.DBL.DIR/'

os.chdir(path_code)

#list_img = os.listdir(path_data)

###################################
###     PART 1 PEPS SEN2 DATA   ###
###################################



# Open the file:
raster = rasterio.open(path_data+'T31TDG_20200507T104031_stack_bgr_crop.tif')


# Convert to numpy arrays
red = raster.read(3)
green = raster.read(2)
blue = raster.read(1)

# Normalize band DN
red_norm = normalize(red)
green_norm = normalize(green)
blue_norm = normalize(blue)

# Stack bands
rgb = np.dstack((red_norm, green_norm, blue_norm))

# View the color composite
plt.imshow(rgb)
#plt.show()
plt.savefig('/home/juan/Documentos/cesbio/assigments/Documento/images/PEPS_show.png')

####################################
###     PART 2 VENUS SEN2 DATA   ###
####################################


# Open the file:
raster2 = rasterio.open(path_data2+'VE_VM01_VSC_PDTIMG_L2VALD_COLOMBIA_20190825_SRE_crop.DBL.TIF')


# Convert to numpy arrays
red2 = raster2.read(3)
green2 = raster2.read(2)
blue2 = raster2.read(1)

# Normalize band DN
red_norm2 = normalize(red2)
green_norm2 = normalize(green2)
blue_norm2 = normalize(blue2)

# Stack bands
rgb2 = np.dstack((red_norm2, green_norm2, blue_norm2))

# View the color composite
plt.imshow(rgb2)
#plt.show()
plt.savefig('/home/juan/Documentos/cesbio/assigments/Documento/images/VENUS_show.png')
