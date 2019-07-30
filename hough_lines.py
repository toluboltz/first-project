import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2

# read in and grayscale the image
image = img.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# define kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# define parameters for Canny edge detection and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color = 255

# This time we are defining a four sided polygon to mask
imshape = image.shape
print(imshape)
#vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
#vertices = np.array([[(50,imshape[0]),(420, 300), (510, 300), (910,imshape[0])]], dtype=np.int32)
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
print(vertices)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# define the Hough transform parameters
rho = 2
theta = np.pi/180
threshold = 10
min_line_length = 40
max_line_gap = 20

# create a blank image (same size as the input image) to draw the output on
line_image = np.copy(image)*0

# run Hough transform on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

# iterate over the output "lines" and draw the lines on the blank image
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

# create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))

# draw the lines on the edge of the image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(combo)
plt.show()

# save image
#plt.imsave('<filename>', <image>)