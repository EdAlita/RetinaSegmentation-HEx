import cv2
import matplotlib.pyplot as plt
import numpy as np

img= cv2.imread("coins2.jpg")
# img=cv2.imread("data/images/training/IDRiD_10.jpg")
cv2.imshow("Original Image",img)

#image convert to BGR to Gray
img_bainary=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#threshold in binary invers and otsu
ret, img_bainary=cv2.threshold(img_bainary,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Binarized Image",img_bainary)

# morphology close 
karnel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
img_bainary_close= cv2.morphologyEx(img_bainary,cv2.MORPH_CLOSE,karnel)
cv2.imshow("Closed binary image", img_bainary_close)

# distance transform 
distance_transform=cv2.distanceTransform(img_bainary_close,cv2.NORM_L1,3)

distance_transform_vis=cv2.normalize(distance_transform,0,255,cv2.NORM_MINMAX)
# distance_transform_vis1=distance_transform_vis.astype('uint8')
# cv2.imshow("Distance Transform Before", distance_transform_vis)
distance_transform_vis=cv2.convertScaleAbs(distance_transform_vis,cv2.CV_8U)

# cv2.imshow("Distance Transform", distance_transform_vis)

ret, dist_transform_bin=cv2.threshold(distance_transform_vis,0.1,255,cv2.THRESH_BINARY)
cv2.imshow("Distance Transform Binarized", dist_transform_bin)

img_overlaid_internal_markers = img.copy()
img_overlaid_internal_markers[dist_transform_bin != 0] = (0, 0, 255)
cv2.imshow("Internal markers overload", img_overlaid_internal_markers)

markers = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)

internal_markers_objs, _ = cv2.findContours(distance_transform_vis, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for k in range(len(internal_markers_objs)):
    cv2.drawContours(markers, internal_markers_objs, k, k + 1, cv2.FILLED)

# # # Assuming img_bin is a numpy.ndarray object and markers is a numpy.ndarray object
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
img_bin_dilated = cv2.dilate(img_bainary, kernel)
cv2.imshow("Dilated binarized image", img_bin_dilated)

# external_markers_objs, _ = cv2.findContours(img_bin_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(markers, external_markers_objs, 0, len(internal_markers_objs) + 1, 2)

# markers_vis = np.zeros_like(markers, dtype=np.uint8)
# cv2.normalize(markers, markers_vis, 0, 255, cv2.NORM_MINMAX)
# cv2.imshow("Markers image", markers_vis)

# # cv2.watershed(img, markers)

# # # # Clone the "markers" image to create a new "dams" image
# # dams = markers.copy()

# # # # Scale the "dams" image to the [0, 510] range and add 255
# # dams = dams * 255 + 255

# # # # Apply saturation and convert the "dams" image to 8-bit unsigned integer
# # dams = cv2.convertScaleAbs(dams)

# # # # Invert the "dams" image to create a binary mask where 0 is for dams
# # dams = 255 - dams

# # # # Apply morphological dilation to the "dams" image using a 3x3 elliptical kernel
# # dams = cv2.dilate(dams, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

# # # # Create a new image "dam_watershed_output" by cloning the input "img" image
# # dam_watershed_output = img.copy()

# # # # Set all pixels in "dam_watershed_output" that correspond to dams in "dams" to red color
# # dam_watershed_output[dams != 0] = (0, 0, 255)

# # # # Display the final output image
# # cv2.imshow("Final result (dams)", dam_watershed_output)

# # # Add 1 to all marker values to shift the range to [1, internal_markers_objs.size() + 1]
# markers += 1

# # # Create a dictionary to store a unique RGB color for each marker value
# object2colors = []
# for i in range(1, len(internal_markers_objs) + 1):
#     object2colors[i] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

# # Create a new image "colored_watershed_output" by cloning the input "img" image
# colored_watershed_output = img.copy()

# # # Assign a color to each marker value in "markers"
# for y in range(colored_watershed_output.shape[0]):
#     ythSegRow = colored_watershed_output[y]
#     ythWatRow = markers[y]

#     for x in range(colored_watershed_output.shape[1]):
#         if ythWatRow[x] in object2colors:
#             color = object2colors[ythWatRow[x]]
#         else:
#             color = (0, 0, 0)  # Default color for missing keys
#         ythSegRow[3 * x + 0] = color[2]
#         ythSegRow[3 * x + 1] = color[1]
#         ythSegRow[3 * x + 2] = color[0]
# # Display the final output image
# cv2.imshow("Final result (regions)", colored_watershed_output)

# Wait for a key press and then close the windows
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
