import numpy as np
import cv2 
import json
import time

result = {}

with open('colors.json', 'r') as f:
    result = json.loads(f.read())

def create_img(title, color):
    fr_color = color
    fr_color = tuple([float(i) for i in fr_color.split(',')])
    img = np.zeros((400, 400, 3), dtype = "uint8")
    img.fill(255)
    window_name = 'Frequent Color'
    center_coordinates = (120, 100)
    radius = 30
    color = fr_color
    thickness = -1
    image = cv2.circle(img, center_coordinates, radius, color, thickness)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    
create_img('Frequent Color',result['frequent'])

for color in result['colors']:
    create_img('Color', color)
