# Handwritten-Digit-Recognition-with-CNN
April 8, 2024

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The model was trained to classify grayscale images of handwritten numbers (0-9), achieving an accuracy of nearly 98%.

## Project Overview

The main goals of this project are to:
- **Develop a CNN model** optimized for image data, leveraging the spatial relationships between pixels.
- **Train and evaluate the model** to classify handwritten digits with high accuracy.
- **Reflect on the ethical implications** of machine learning in image recognition applications.

## Dataset

- **MNIST Dataset**: A standard dataset of 28x28 grayscale images of handwritten digits (0-9). The dataset is commonly used in image processing and deep learning for benchmarking.

## Methodology

### 1. Data Preprocessing

- **Image Reshaping**: Added a channel dimension for compatibility with CNN layers.
- **Normalization**: Pixel values were scaled to a range of 0 to 1 by dividing by 255.
- **Tensor Conversion**: The data was converted to PyTorch tensors, with `TensorDataset` and `DataLoader` used to batch and shuffle the data for efficient processing.

### 2. Model Architecture

- **CNN Layers**: 
  - Two convolutional layers with ReLU activation and max-pooling to detect features.
  - Three fully connected layers (fc1, fc2, fc3), with the final layer outputting 10 units for the 10 classes (digits 0-9).
- **Loss and Optimization**:
  - **Loss Function**: CrossEntropyLoss for classification.
  - **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.001 and momentum of 0.9 to accelerate convergence and avoid local minima.

### 3. Training Process

- **Epochs**: The model was trained for 10 epochs, with mini-batch training.
- **Training Loop**: Each epoch involved a forward pass, loss calculation, backpropagation, and weight update.
- **Progress Monitoring**: Loss was printed every 200 mini-batches to track learning progress.

### 4. Evaluation

- The model achieved an accuracy of nearly **98%** on the test set.
- **Testing Function**: A custom function was created to evaluate model accuracy and loss on the test data. `torch.no_grad()` was used to improve computational efficiency during testing.

## Results

- **Model Accuracy**: The model demonstrated an accuracy of approximately 98%, validating the CNN’s effectiveness for image classification.
- **Ethical Considerations**: While CNNs are powerful for tasks like handwriting recognition, similar technology in areas like facial recognition raises ethical concerns, such as privacy and potential biases.

## Installation and Usage

### Prerequisites

- Python 3.x
- Required libraries: `pandas`, `numpy`, `torch`, `matplotlib`

### Running the Project

1. **Download the Notebook**:
   - Download `AML_project_2.ipynb` from this repository.

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy torch matplotlib
3. **Open and Execute the Notebook**:
   - Run the notebook in Jupyter Notebook, JupyterLab, or Google Colab to view the full process, including data visualization, model training, and evaluation.
