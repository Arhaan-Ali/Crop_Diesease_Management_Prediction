# Crop Disease Management Prediction

## Overview

This repository contains the implementation of a crop disease prediction and management system built in Python. The project uses machine learning techniques to detect and classify crop diseases from input images and provides corresponding health status assessments. The objective is to assist agricultural stakeholders by enabling automated, data-driven identification of plant diseases, thereby improving crop health monitoring and facilitating timely intervention.

## What It Does?
* Uses a trained model to analyze leaf or crop images.

* Predicts whether a crop is **healthy or diseased.**

* Outputs a disease category and **confidence score.**

* Includes training, evaluation, preprocessing, and **prediction tools.**

## Key Features
* Deep Learning model built with PyTorch

* Image preprocessing using torchvision

* Model training and weight saving

* FastAPI backend for inference

* Interactive Swagger API docs

* Clean virtual environment setup


## Project Structure
# Project Structure

```
Crop_Diesease_Management_Prediction/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── training/
│   └── CDMP.py
|   └── divider.py 
│
├── finalised_dataset/
│   ├── Train/
│   └── Validation/
│
├── rice_disease_model.pt
├── requirements.txt
└── README.md
```

## Model Training
* Loads dataset using torchvision.datasets.ImageFolder

* Applies transformations (resize, normalization)

* Trains CNN model

* Saves model weights using:
```
torch.save(model.state_dict(), "rice_disease_model.pt")
```

* To train the model:
```
python training/CDMP.py
```

## Setup Instructions
1. Clone Repository
```
git clone https://github.com/Arhaan-Ali/Crop_Diesease_Management_Prediction.git
cd Crop_Diesease_Management_Prediction
```

2. Create Virtual Environment
```
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install Dependencies 
```
pip install -r requirements.txt
```

4. Run FastAPI Server
```
uvicorn app.main:app --reload
```
Server will start at:
```
http://127.0.0.1:8000
```

## Tech Stack
* Python 3.11
* PyTorch
* Torchvision
* FastAPI
* Uvicorn
* Pillow
