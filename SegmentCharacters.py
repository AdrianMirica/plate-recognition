import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import DetectPlate

# The invert was done so as to convert the black pixel to white pixel and vice versa
license_plate = np.invert(DetectPlate.plate_like_objects[0])

labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")

# facem presupunerea ca dimensiunile unui caracter din placuta de inmatriculare
# sunt: inaltime intre 35% si 80% din inaltimea placutei si latime intre 4% si 15%
character_dimensions = (0.35*license_plate.shape[0], 0.80*license_plate.shape[0], 0.040*license_plate.shape[1], 0.15*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

#print("min_height: " + str(min_height))
#print("max_height: " + str(max_height))
#print("min_width: " + str(min_width))
#print("max_width: " + str(max_width))
#print("----------------------------------------------")

characters = []
counter=0
column_list = []
for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    #print(str(x0) + " " + str(x1) + " " + str(y0) + " " + str(y1))
    region_height = y1 - y0
    region_width = x1 - x0
    #print("region width: " + str(region_width) + ", region_height: "+str(region_height))
    #print("----------------------------------------------")

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]
        #print("region_height > min_height: " + str(region_height > min_height))
        #print("region_height < max_height: " + str(region_height < max_height))
        #print("region_width > min_width: " + str(region_width > min_width))
        #print("region_width < max_width: " + str(region_width < max_width))
        #print("----------------------------------------------")

        # desenez un chenar rosu in jurul fiecarui caracter
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        # redimensionare imagine caracter
        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        # pastrare ordine caractere.
        column_list.append(x0)
# print(characters)
plt.show()