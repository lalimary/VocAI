# VocAI Project

## Overview

VocAI is a project that combines sign language recognition and emotion detection to enhance communication. The system captures sign language gestures using computer vision and recognizes associated emotions in real-time.

## Technologies Used

- **Python**: The project is primarily developed in Python, leveraging its versatility and extensive libraries.

- **Streamlit**: A web app framework for creating interactive user interfaces. Streamlit is used to build the project's user interface.

- **OpenCV**: An open-source computer vision and machine learning software library. OpenCV is utilized for image and video processing.

- **Mediapipe**: A library for building perception pipelines. In this project, Mediapipe is employed for holistic human pose estimation.

- **TensorFlow/Keras**: Used for developing and training machine learning models. TensorFlow is the backend, and Keras is used for building and training neural networks.

- **Replicate API**: An API used for natural language generation. It's employed for converting recognized sign language gestures into spoken words.

## Features

- **Real-time Sign Language Recognition**: The system recognizes sign language gestures using computer vision.

- **Emotion Detection**: Emotion is detected in real-time, enhancing the user experience.

- **Audio Generation**: The recognized sign language gestures are converted into spoken words using the Replicate API.

## Run the application
- streamlit run app.py

