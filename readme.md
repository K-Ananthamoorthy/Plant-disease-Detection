# Plant Disease Detection

This project is a web application for detecting plant diseases using a Convolutional Neural Network (CNN) model. The front end is built using Flask, and image processing is handled by OpenCV.

## Table of Contents

- [Plant Disease Detection](#plant-disease-detection)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Training](#model-training)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Project Structure

plant_disease_detection/
│
├── plant-env/ # Virtual environment directory
├── app.py # Main Flask application
├── model.py # Script to define and load your model
├── preprocess.py # Script for image preprocessing
├── templates/
│ └── index.html # HTML template for the front end
└── uploads/ # Directory to save uploaded images



## Installation

Follow these steps to set up the project on your local machine.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/plant_disease_detection.git
    cd plant_disease_detection
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv plant-env
    source plant-env/bin/activate  # On Windows use `plant-env\Scripts\activate`
    ```

3. **Install the necessary dependencies:**

    ```bash
    pip install flask opencv-python tensorflow keras numpy
    ```

4. **Set up the uploads directory:**

    ```bash
    mkdir uploads
    ```

## Usage

1. **Run the Flask application:**

    ```bash
    python app.py
    ```

2. **Open your web browser and go to:**

    ```
    http://127.0.0.1:5000/
    ```

3. **Upload an image to detect plant disease.**

## Model Training

The model used in this project is a Convolutional Neural Network (CNN) implemented using TensorFlow and Keras. Below is a brief overview of how to train the model.

1. **Prepare your dataset:**
   - Collect images of healthy and diseased plants.
   - Split the dataset into training and testing sets.
   - Save the trained model weights to use in the application.

2. **Define and train the model:**

    ```python
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    def create_model():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax')  # Adjust for the number of classes
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Assuming X_train and y_train are your training data and labels
    model = create_model()
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    model.save_weights('path_to_save_model_weights.h5')
    ```

3. **Load the trained model in the Flask application:**

    ```python
    def load_model():
        model = create_model()
        model.load_weights('path_to_your_model_weights.h5')
        return model
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for 
