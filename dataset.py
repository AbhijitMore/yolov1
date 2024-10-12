import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image


class FruitImagesDataset(torch.utils.data.Dataset):
    def __init__(self, df=None, files_dir=None, S=7, B=2, C=3, transform =None):
        self.annotations = df
        self.files_dir = files_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.files_dir, self.annotations.iloc[index, 1])
        boxes = []
        tree = ET.parse(label_path)
        root = tree.getroot()

        class_dictionary = {'apple': 0, 'banana': 1, 'orange':2}

        if(int(root.find('size').find('height').text)==0):
            filename = root.find('filename').text
            img = Image.open(self.files_dir + '/' + filename)
            img_width, img_height = img.size

            for member in root.findall('object'):
                klass = member.find('name').text
                klass = class_dictionary[klass]

                #bounding box
                xmin = int(member.find('bndbox').find('xmin').text)
                xmax = int(member.find('bndbox').find('xmax').text)
                ymin = int(member.find('bndbox').find('ymin').text)
                ymax = int(member.find('bndbox').find('ymax').text)

                centerx = ((xmax + xmin)/2) /img_width
                centery = ((ymax+ ymin)/2) / img_height
                boxwidth = (xmax - xmin) / img_width
                boxheight = (ymax - ymin) / img_height

                boxes.append([klass, centerx, centery, boxwidth, boxheight])

        elif(int(root.find('size').find('height').text)!=0):

            for member in root.findall('object'):
                klass = member.find('name').text
                klass = class_dictionary[klass]

                #bounding box
                xmin = int(member.find('bndbox').find('xmin').text)
                xmax = int(member.find('bndbox').find('xmax').text)
                img_width = int(root.find('size').find('width').text)
                ymin = int(member.find('bndbox').find('ymin').text)
                ymax = int(member.find('bndbox').find('ymax').text)
                img_height = int(root.find('size').find('height').text)

                centerx = ((xmax + xmin)/2) /img_width
                centery = ((ymax+ ymin)/2) / img_height
                boxwidth = (xmax - xmin) / img_width
                boxheight = (ymax - ymin) / img_height

                boxes.append([klass, centerx, centery, boxwidth, boxheight])

        boxes = torch.tensor(boxes)
        img_path = os.path.join(self.files_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform:
            image, boxes = self.transform(image, boxes)

        #convert to cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            #i,j represents the cell_row and cell_column
            i,j = int(self.S*y), int(self.S*x)
            x_cell, y_cell = self.S*x-j, self.S*y-i

            """
            Calculating the width and height of cell of bounding box, relative to the cell is done by following, with width as the example:

            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)

            Then to find the width relative to the cell is simply:
            width_pixels/ cell_pixels, simplification leads to the formulas below.
            """

            width_cell, height_cell = (width * self.S, height * self.S)

            #If no object already found for specific cell i, j Note: this means we restrict to ONE object per cell !

            if label_matrix[i, j, self.C] == 0:
                #set that there exists an object
                label_matrix[i, j , self.C] = 1
                # box coordinates
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_matrix[i, j, 4:8] = box_coordinates
                label_matrix[i,j, class_label] = 1

        return image, label_matrix