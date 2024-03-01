import base64
from autodistill.detection import CaptionOntology
from autodistill_detic import DETIC
from autodistill_grounded_sam import GroundedSAM
from autodistill_yolov8 import YOLOv8
from IPython.display import Image
import cv2
import numpy as np
import supervision as sv
import os
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch
torch.cuda.empty_cache()

base_model = DETIC(ontology=CaptionOntology({"Apple": "Apple"}))
target_model = YOLOv8("yolov8n.pt")

@app.route('/vid_to_img', methods=['GET'])
def vid_to_img():
    vidcap = cv2.VideoCapture('apple.mp4')
    count = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(round(fps / 1))
    os.makedirs("images", exist_ok=True)
    while True:
        ret, frame = vidcap.read()    
        # Break the loop if no more frames are available
        if not ret:
            break

        # Save the frame as a JPEG file with a unique name
        image_path = os.path.join("images", f"frame{count}.jpg")
        cv2.imwrite(image_path, frame)
        # Skip frames to achieve the desired fps
        for i in range(skip_frames - 1):
            vidcap.read()
        count += 1
        return "converting"


@app.route('/label', methods=['GET'])
def label():
    base_model.label(input_folder="data", output_folder="./dataset", extension=".jpeg")
    return "Labeling"

@app.route('/save_labelplot', methods=['GET'])
def save_labelplot():
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path="dataset/train/images",
        annotations_directory_path="dataset/train/labels",
        data_yaml_path="dataset/data.yaml")

    image_names = list(dataset.images.keys())
    box_annotator = sv.BoxAnnotator()
    plot_images = []

    for image_name in image_names:
        image = dataset.images[image_name]
        annotations = dataset.annotations[image_name]
        labels = [dataset.classes[class_id] for class_id in annotations.class_id]
        annotates_image = box_annotator.annotate(scene=image,detections=annotations,labels=labels)
        plot_images.append(annotates_image) 

    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(16,16))
    for idx, ax in enumerate(axes.flat):
        if idx < len(plot_images):        
            ax.imshow(cv2.cvtColor(plot_images[idx], cv2.COLOR_BGR2RGB))
    plt.savefig("dataset/images/label.jpeg")

    frame = cv2.imread("dataset/images/label.jpeg")
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return render_template('display_label.html', image_base64=image_base64)


@app.route('/train', methods=['GET'])
def train():
    target_model = YOLOv8("yolov8n.pt")
    target_model.train("dataset/data.yaml", epochs=10)
    return "training"


@app.route('/save_prediction', methods=['GET'])
def save_prediction():
    image_folder = "data/"
    # List all files in the folder
    image_files = [f for f in os.listdir(image_folder)] 

    # Iterate over each image file
    for image_file in image_files:
        # Full path to the current image
        test_image = os.path.join(image_folder, image_file)
        # Read the image
        image = cv2.imread(test_image)
        # Perform object detection
        detections = base_model.predict(test_image)

        # Create an annotator
        box_annotator = sv.BoxAnnotator()
        classes = ["Apple"]
        labels = [f"{classes[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        # Display or save the annotated image
        save_path = os.path.join("annotated_images", f"annotated_{image_file}")
        cv2.imwrite(save_path, annotated_frame)

        image_paths = [os.path.join("annotated_images", filename) for filename in os.listdir("annotated_images")]
        image_base64_list = []
        # Loop through each image file
        for i in image_paths:
            frame = cv2.imread(i)
            _, buffer = cv2.imencode('.jpeg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_base64_list.append(image_base64)
        return render_template('training_result.html', image_base64_list=image_base64_list)
    
@app.route('/')
def index():  
    return render_template('index.html', autodistill="autodistill")

if __name__ == '__main__':
    app.run(debug=True)



'''# run inference on a single image
def plotimage():
    annotator = sv.BoxAnnotator()
    results = base_model.predict("data/img_p1_8.jpeg")
    image=cv2.imread("data/img_p1_8.jpeg")
    classes=base_model.ontology.classes()
    labels = [f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in results]
    annotated_frame = annotator.annotate(scene=image, labels=labels, detections=results)
    sv.plot_image(annotated_frame, size=(8, 8))'''

       










