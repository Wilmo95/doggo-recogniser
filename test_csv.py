from PIL import Image
import numpy as np
img = np.array(Image.open("archive/images/Images/affenpinscher/n02110627_233.jpg"))
# print(img)

import csv

def csvWriter(fil_name, nparray):
  example = nparray.tolist()
  with open(fil_name+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(example)

csvWriter("myfilename", img)