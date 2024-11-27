# SignSpeak: Sign Language Translation to Multiple Languages

**SignSpeak** is a project designed to translate sign language (in the form of images) into English, and then into multiple languages. The project leverages computer vision and deep learning techniques to identify gestures and translate them into readable text.

---

## Project Overview

This repository contains the initial steps for a sign language recognition system. We have processed and prepared a dataset of sign language images, preprocessed them, and are now ready to build a model to predict sign language gestures from images.

---

## Key Features
- **Sign Language Recognition**: Translate images of hand gestures (American Sign Language) into text.
- **Preprocessing**: Dataset images are resized, normalized, and organized for training.
- **Split into Train, Test, and Validation Sets**: Dataset is divided into different sets for model training and evaluation.
  
---

## Steps Completed So Far

### 1. **Dataset Selection and Exploration**
   - We selected the **ASL Alphabet Dataset** consisting of images representing letters of the American Sign Language (ASL) alphabet.
   - The dataset was organized into directories based on each letter ('a' to 'z').

### 2. **Data Preprocessing**
   - **Resizing**: All images were resized to a consistent shape of **224x224** for easier processing.
   - **Normalization**: Images were normalized using the standard ImageNet normalization values to prepare them for neural network processing.
   - **Data Augmentation**: Optional augmentation steps (e.g., random rotations) can be applied, if necessary.

### 3. **Data Inspection**
   - We examined the dataset for image resolutions, corrupted images, and balanced class distribution.
   - Images were resized to a common resolution of **224x224** for training consistency.

### 4. **Dataset Organization**
   - Data was split into **Training**, **Validation**, and **Test** sets, ensuring balanced class distribution.
   - The splits were ready for use in model training and evaluation.

### 5. **Custom Dataset Creation**
   - A custom PyTorch Dataset class was created to load and process images with associated labels.
   - **DataLoader** was set up for efficient batching and shuffling during training.

### 6. **Visualization**
   - A few sample images from the dataset were visualized to understand the dataset's contents and confirm preprocessing steps.
   - We plotted the **class distribution** to check if the data was balanced across different letters.

---

## Data Structure

The dataset is structured as follows:

data/ ├── a/ │ ├── a_01.jpg │ ├── a_02.jpg │ └── ... ├── b/ ├── c/ ├── ... └── z/


Each folder corresponds to a letter of the alphabet, with images representing the ASL sign for that letter.

---

## Requirements



Install the required packages using:

```bash
pip install -r requirements.txt

Getting Started
1. Clone the Repository

git clone https://github.com/yourusername/signspeak.git
cd signspeak

2. Prepare the Dataset

Download the ASL Alphabet Dataset and place it under the data/ directory. The directory structure should look like the example above.
3. Preprocess the Data

Run the preprocessing code to resize, normalize, and split the data into train, validation, and test sets.
4. Build and Train the Model

We will define and train a convolutional neural network (CNN) or use a pre-trained model for sign language recognition. The model will be trained on the training data and validated on the validation set.
5. Evaluate the Model

Evaluate the model on the test data to assess its performance and make improvements where needed.
Future Work

    Model Improvement: Implement more sophisticated models such as pre-trained networks (e.g., ResNet, VGG) for better accuracy.
    Real-Time Inference: Extend the system for real-time sign language recognition using webcam input.
    Multi-Language Translation: After translating to English, the system can be extended to translate into other languages.

License

This project is licensed under the MIT License - see the LICENSE file for details.