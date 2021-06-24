from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

filename = './videos/video2.mp4'
#filename = './video12.mp4'

import cv2
cap = cv2.VideoCapture(filename)


# cap = cv2.VideoCapture(0)
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    #videoclipul original este filmat la rezoluția 1920 x 1080, însă pentru o mai
    #buna vizualizare dorim să reducem dimensiunea ferestrei de redare la 1024 x 576, păstrând rația clipului original( 16:9)
    #resized_frame = cv2.resize(frame, (1024, 576)) 
    if ret == True:
        #cv2.imshow('window-name',resized_frame)
        cv2.imshow('window-name',frame)
        cv2.imwrite("./output/frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

# alegem o imagine cu autovehicului -> o transformam în imagine cu nivele de gri(grayscale) ->
# si apoi o transformam in imagine binara
import imutils
car_image = imread("./output/frame%d.jpg"%(count-1), as_gray=True)
car_image = imutils.rotate(car_image, 270)
#daca dorim să folosim o imagine ca intrare si nu un videoclip
# car_image = imread("car.png", as_gray=True)

# trebuie sa obtinem o matrice
print(car_image.shape)

#un pixel gri utilizand skimage are o valoare cuprinsa intre 0 si 1
#inmultind valoarea aceasta cu 255, vom obtine o valoare din intervalul 0 - 255
#interval cu care putem lucra mult mai usor

gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
# print(binary_car_image)
ax2.imshow(binary_car_image, cmap="gray")
# ax2.imshow(gray_car_image, cmap="gray")
plt.show()

# Aplicarea algoritmului CCA. Gasirea zonelor conectate din imaginea binarizata

from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# aceasta functie obtine toate regiunile si le grupează
label_image = measure.label(binary_car_image)

# print(label_image.shape[0]) #latimea imaginii


# setarea parametrilor unei placute de inmatriculare: latimea maxima, latimea minima si inaltimea maxima si  minima
plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray")
flag =0

# functia regionprops creaza o lista a proprietatilor regiunilor marcate
for region in regionprops(label_image):
    # print(region)
    if region.area < 50:
        # daca regiunea este foarte mica, este putin probabil sa fie o placuta de inmatriculare
        continue
        # altfel salvam coordonatele regiunii
    min_row, min_col, max_row, max_col = region.bbox
    # print(min_row)
    # print(min_col)
    # print(max_row)
    # print(max_col)

    region_height = max_row - min_row
    region_width = max_col - min_col
    # print(region_height)
    # print(region_width)

    # verificare ca regiunea identificata satisface presupunerea facuta
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        flag = 1
        plate_like_objects.append(binary_car_image[min_row:max_row,
                                  min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                         max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
        # trasam un chenar rosu peste regiunea respectiva
#verificam daca presupunerea initiala a fost validata, daca da desenam un grafic
#daca nu verificam presupunerea 2.        
if(flag == 1):
    # print(plate_like_objects[0])
    plt.show()




if(flag==0):
    min_height, max_height, min_width, max_width = plate_dimensions2
    plate_objects_cordinates = []
    plate_like_objects = []

    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_car_image, cmap="gray")

    # functia regionprops creaza o lista a proprietatilor regiunilor marcate
    for region in regionprops(label_image):
        if region.area < 50:
            # daca regiunea este foarte mica, este putin probabil sa fie o placuta de inmatricular
            continue
            # altfel salvam coordonatele regiunii
        min_row, min_col, max_row, max_col = region.bbox
        # print(min_row)
        # print(min_col)
        # print(max_row)
        # print(max_col)

        region_height = max_row - min_row
        region_width = max_col - min_col
        # print(region_height)
        # print(region_width)

        # verificare ca regiunea identificata satisface presupunerea facuta
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:

            plate_like_objects.append(binary_car_image[min_row:max_row,
                                      min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                             max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                           linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
            # trasam un chenar rosu peste regiunea respectiva
    # print(plate_like_objects[0])
    plt.show()