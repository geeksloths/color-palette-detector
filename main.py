from __future__ import print_function
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import requests
from io import BytesIO
import json
import cv2 

NUM_CLUSTERS = 5
result = {}
url = input("image url: ")
response = requests.get(url)
print('reading image')
im = Image.open(BytesIO(response.content))
width = im.size[0]
height = im.size[1]
im.save('image.jpg')
im = im.resize((300, 300))      # optional, to reduce time
ar = np.asarray(im)
shape = ar.shape
ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
all_colors = codes
vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
counts, bins = np.histogram(vecs, len(codes))    # count occurrences
index_max = np.argmax(counts)                    # find most frequent
peak = codes[index_max]
colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
frequent_color = ",".join([str(i) for i in peak.tolist()])
all_colors = [ ",".join([str(i) for i in color]) for color in all_colors ]
result['frequent'] = frequent_color
result['colors'] = all_colors

with open('colors.json', 'w') as f:
    f.write(json.dumps(result))

current = 0
image = cv2.imread('image.jpg', cv2.COLOR_BGR2RGB)
image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

color_width = width * 0.25 * 0.2
color_height = color_width

print(f"color_width: {color_width}")
print(f"color_height: {color_height}")
print(f"image_width: {width}")
print(f"image_height: {height}")


# add most frequent color
color = tuple([float(i) for i in result['frequent'].split(',')])
start_y = (height * 0.5) - (color_height * 0.5)
start_y = int(start_y)
start_x =  width - color_width
start_x =  int(start_x)
end_y = (height * 0.5) + (color_height * 0.5)
end_y = int(end_y)
end_x = width
end_x = int(end_x)
start_point = (start_x, start_y)
end_point = (end_x, end_y)
image = cv2.rectangle(image, start_point, end_point, color, -1)

for color in result['colors']:
    window_name = 'Image'
    start_y = height - color_height
    start_y = int(start_y)
    start_x =  width - ((current+1) * color_width)
    start_x =  int(start_x)
    end_y = int(height)
    end_x = width - (current * color_width)
    end_x = int(end_x)
    print(f"start_x: {start_x}, start_y: {start_y}, end_x: {end_x}, end_y:{end_y}")
    start_point = (start_x, start_y)
    end_point = (end_x, end_y)
    color = tuple([float(i) for i in color.split(',')])
    print(color)
    thickness = -1
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    current +=1
# cv2.imshow(window_name, image)
# cv2.imwrite("colored.jpg",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
im = Image.fromarray(image)
im.save("colored.jpg")
# cv2.waitKey(0)