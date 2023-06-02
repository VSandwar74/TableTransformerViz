# my conda tables-detr instance has problems with opencv so use requirements.txt if needed
import os
import sys
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Constants and configuration options
if 'ipykernel' in sys.modules:
    # Running in Jupyter Notebook
    BASE_DIR = os.getcwd()
else:
    # Running in a standalone Python script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pass the data dir to BASE_DIR
BASE_DIR = os.path.join(BASE_DIR ,'data',"test_examples_dataset_generation")

# Constants and configuration options
CONFIG = {
    "BASE_DIR": BASE_DIR,
    "CSV_FILE": os.path.join(BASE_DIR, "snippet_document_mapping.csv"),
    "IMAGES_AND_ANNOTATIONS": os.path.join(BASE_DIR, "images-and-annotations"),
    "VISUALIZE_ANNOTATIONS": os.path.join(BASE_DIR, "visualize-annotations"),
    "ANNOTATION_LIMIT": 1000,
    "SUB_DIRS": [
        "table-column",
        "table",
        "table-row",
        "table-spanning-cell",
        "table-projected-row-header",
        "table-column-header"
    ]
}


# Color map for annotations
COLOR_MAP = {
    "table column": (0, 0, 255),  # Red
    "table row": (0, 255, 0),  # Green
    "table spanning cell": (255, 0, 0),  # Blue
    "table": (255, 255, 0),  # Yellow
    "table projected row header": (255, 0, 255),  # Magenta
    "table column header": (255, 128, 0),  # Orange
}

# Annotations for draw boxes function
ANNOTATION_TYPES = [
    "table column",
    "table row",
    "table spanning cell",
    "table",
    "table projected row header",
    "table column header",
]   

def parse_annotation(annotation_file):
    """
    Parse an XML annotation file into a list of bounding boxes.

    :param annotation_file: Path to the XML annotation file.
    :returns: A list of bounding boxes. Each box is a tuple containing
        the box name and its coordinates: (name, xmin, ymin, xmax, ymax).
    """
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    bounding_boxes = []
    for obj in root.iter("object"):
        name = obj.find("name").text
        box = obj.find("bndbox")
        xmin = int(float(box.find("xmin").text))
        ymin = int(float(box.find("ymin").text))
        xmax = int(float(box.find("xmax").text))
        ymax = int(float(box.find("ymax").text))
        bounding_boxes.append((name, xmin, ymin, xmax, ymax))

    return bounding_boxes

def draw_boxes(image_path, annotation_path, specific_annotation, color_map):
    """
    Draw bounding boxes on an image based on an annotation file.

    :param image_path: Path to the image file.
    :param annotation_path: Path to the XML annotation file.
    :param specific_annotation: If not empty, only draw boxes with this annotation.
    :param color_map: Dictionary mapping annotations to RGB color tuples.
    :returns: The image with bounding boxes drawn on it.
    """
    image = cv2.imread(image_path)
    bounding_boxes = parse_annotation(annotation_path)
    
    for box in bounding_boxes:
        name, xmin, ymin, xmax, ymax = box
        if specific_annotation == "" or name == specific_annotation:
            color = color_map.get(name, (0, 255, 255))  # Cyan is default
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)

    return image

def main():
    """
    Main function to execute the script
    """
    try:
        # Make directories
        sub_dirs = [annotation.replace(' ', '-') for annotation in ANNOTATION_TYPES]
        os.makedirs("visualize_annotations", exist_ok=True)
        for sub_dir in sub_dirs:
            path = os.path.join("visualize_annotations", sub_dir)
            os.makedirs(path, exist_ok=True)
        
        # Iterate through files
        for file_name in set(os.listdir('data')):
            prefix = os.path.splitext(file_name)[0]
            if prefix == '.DS_Store':
                continue
            else:
                for annotation in ANNOTATION_TYPES:
                    image = draw_boxes(f"data/{prefix}.jpg", f"data/{prefix}.xml", annotation, COLOR_MAP)
                    cv2.imwrite(f"visualize_annotations/{annotation.replace(' ', '-')}/{prefix}_{annotation}.png", image)
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise e

if __name__ == '__main__':
    main()
