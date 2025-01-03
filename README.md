# YOLOv5 Vehicle Detection Project

This repository contains a project for training and deploying a YOLOv5 model to detect vehicles such as Ambulances, Buses, Cars, Motorcycles, and Trucks. The dataset used in this project follows a structured format with training, validation, and test sets, and the implementation is designed to run on Google Colab for ease of use.

---

## Project Overview

The goal of this project is to utilize the YOLOv5 (You Only Look Once) object detection framework to identify and classify different types of vehicles in images. The project includes the following steps:

1. Setting up the YOLOv5 framework.
2. Preparing the dataset and ensuring the correct directory structure.
3. Training the YOLOv5 model using a labeled dataset.
4. Running inference on test images.
5. Saving and deploying the trained model.

---

## Dataset Structure

The dataset should be organized as follows:

```
VehiclesDetectionDataset/
   train/
      images/
      labels/
   valid/
      images/
      labels/
      dataset.yaml
   test/
      images/
      labels/
```

### Dataset YAML File
The `dataset.yaml` file contains the dataset configuration, including paths to the training, validation, and test sets, the number of classes, and their names. Below is an example:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 5
names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
```

---

## Steps to Run the Project

### 1. Install YOLOv5
Install the YOLOv5 library in your Google Colab environment:
```bash
!pip install ultralytics
```

### 2. Train the Model
Train the YOLOv5 model using the following code:
```python
from ultralytics import YOLO

model = YOLO('YOLOv5n.pt')

results = model.train(
    data='/content/VehiclesDetectionDataset/valid/dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=416,
    save=True,
    callbacks=False
)
```

### 3. Run Inference
Perform predictions on the test set:
```python
test_results = model.predict(
    source="/content/VehiclesDetectionDataset/test/images",
    save=True
)
```

### 4. Save the Trained Model
Save the best-trained model to Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

shutil.copy("runs/detect/train/weights/best.pt", "/content/drive/My Drive/best.pt")
```

---

## Results
The trained YOLOv5 model can detect and classify the following vehicle types:
- Ambulance
- Bus
- Car
- Motorcycle
- Truck

Results from the inference step will be saved in the `runs/detect/predict/` directory. Example images with predictions are displayed directly in the Colab notebook.

---

## Requirements

- Python 3.8+
- Google Colab
- Libraries: `ultralytics`, `shutil`, `os`
- A labeled dataset following the specified structure

---

## How to Use

1. Clone this repository to your local machine or Google Colab environment.
2. Upload your dataset with the specified directory structure.
3. Run the provided Colab script step-by-step.
4. Use the trained model for inference or further fine-tuning.

---

## Acknowledgments

This project uses the YOLOv5 framework from [Ultralytics](https://github.com/ultralytics/ultralytics). Special thanks to Roboflow for providing the dataset URL and tools for annotation.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
