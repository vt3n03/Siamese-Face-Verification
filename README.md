# Siamese Face Verification

This project implements a Siamese neural network for face verification. The system compares an input image against a set of reference images and determines whether they belong to the same individual. It includes model training, data preparation pipelines, and a Kivy-based application for real time verification using a webcam.

# Overview
Face verification is the task of confirming whether two face images represent the same person. Unlike classification, which predicts a fixed identity label, verification focuses on similarity. A Siamese network is ideal for this task because it learns to extract feature embeddings from images and measure their distance in embedding space.

This project uses TensorFlow and Keras to train the Siamese model and then deploys it within an interactive application that performs live face verification.

# Key Features
1. Siamese network with shared convolutional architecture

2. Contrastive-style loss optimization

3. Positive and negative pair generation for training

4. Real time webcam capture and inference through Kivy

5. Adjustable detection and verification thresholds

6. Organized dataset structure for anchors, positive samples, and negative samples


# Project Structure
```graphql
Siamese-Face-Verification/
│
├── data/
│   ├── anchor/              # Reference images of the target identity
│   ├── positive/            # Additional images of the same identity
│   ├── negative/            # Images belonging to different identities
│
├── data/
│   ├── LFW                  # Labelled Faces in the Wild (LFW) Dataset
├── application_data/
│   ├── verification_images/ # Images for from positive
│   └── input_image/         # Temporary folder for captured webcam images
│
├── model/                   # Saved model weights and architecture
├── training/                # Training scripts and utilities
│
├── main.py                  # Kivy application for real time verification
└── README.md
```

# How the Model Works
The Siamese architecture consists of two identical subnetworks that encode images into embeddings. During training:

1. Positive pairs (same identity) are encouraged to have small embedding distance.

2. Negative pairs (different identities) are pushed apart.

This process teaches the network to capture identity specific facial features rather than lighting, angle, or background noise.

A distance threshold is later applied during verification. If the embedding distance between the input image and the stored reference images falls below this threshold, the system classifies them as a match.

# Dataset Requirements
Place your images inside the ``` data ``` folder:
- anchor
One or more reference images of the person you want the system to recognize.

- positive
Additional images of the same person under different angles or lighting.

- negative
Images of other people used to teach the model how to distinguish identities.

# Training
Training scripts generate positive and negative pairs, preprocess images, and build the Siamese model. You can configure:

- number of epochs

- batch size

- embedding dimension

- distance threshold

After training, the final model is saved in the model/ directory and loaded by the verification application.


# Real Time Verification App
The Kivy interface:

- Opens the webcam stream

- Captures a still image

- Runs the face detector

- Computes the embedding for the captured frame

- Compares it to stored anchor embeddings

- Displays detection and verification scores

- Outputs a final Boolean decision

Thresholds can be tuned to reduce false positives or false negatives.
