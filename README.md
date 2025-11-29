# CIFAR-10 Image Classification

A deep learning project that implements an Artificial Neural Network (ANN) for classifying images from the CIFAR-10 dataset using PyTorch.

## 📋 Project Overview

This project demonstrates image classification on the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 different classes. The implementation uses a deep ANN with ReLU activations and CrossEntropyLoss, trained with the Adam optimizer.

## 🎯 Features

- **Deep ANN Architecture**: 5-layer fully connected neural network
- **ReLU Activation**: Non-linear activation functions for better learning
- **Dropout Regularization**: Prevents overfitting with 30% dropout
- **Comprehensive Evaluation**: Training metrics, confusion matrix, and sample predictions
- **Visualization**: Training progress and results visualization

## 📊 Dataset

**CIFAR-10 Dataset** contains 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

**Dataset Statistics:**
- Training samples: 50,000
- Test samples: 10,000
- Image size: 32×32×3 (RGB)
- Number of classes: 10

## 🛠 Technologies Used

- **Python** - Programming language
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **scikit-learn** - Evaluation metrics
- **seaborn** - Confusion matrix visualization

## 🏗 Model Architecture

```python
CIFAR10Classifier(
  (fc1): Linear(3072 → 512)      # Input layer
  (fc2): Linear(512 → 256)       # Hidden layer 1
  (fc3): Linear(256 → 128)       # Hidden layer 2
  (fc4): Linear(128 → 64)        # Hidden layer 3
  (fc5): Linear(64 → 10)         # Output layer
  (dropout): Dropout(p=0.3)      # Regularization
)
```

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd cifar10-classification
```

2. **Install dependencies**
```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn
```

3. **Run the training script**
```bash
python cifar10_classification.py
```

## 🚀 Usage

The script automatically:
- Downloads the CIFAR-10 dataset
- Preprocesses and normalizes the images
- Builds and trains the neural network
- Evaluates model performance
- Generates visualizations and saves the model

### Key Parameters

- **Epochs**: 50
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Activation**: ReLU

## 📈 Results

The model provides:
- Training loss and accuracy curves
- Test set evaluation metrics
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization
- Sample predictions with true vs predicted labels

### Expected Performance
While accuracy may vary, the model typically achieves reasonable performance on this challenging dataset. For improved results, consider using Convolutional Neural Networks (CNNs).

## 📁 Project Structure

```
cifar10-classification/
│
├── cifar10_classification.py  # Main training script
├── cifar10_ann_model.pth      # Saved model (after training)
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## 📊 Outputs Generated

1. **Training Plots**:
   - Loss curve over epochs
   - Accuracy progression

2. **Evaluation Metrics**:
   - Classification report
   - Confusion matrix heatmap

3. **Visualizations**:
   - Sample test images with predictions
   - Model performance analysis

## 🔧 Customization

You can modify the following parameters in the script:

- **Network architecture** (hidden layers, neurons)
- **Training parameters** (epochs, batch size, learning rate)
- **Regularization** (dropout rate)
- **Optimizer settings**

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

**Note**: This implementation uses a fully connected ANN. For better performance on image data, consider extending the project with Convolutional Neural Networks (CNNs) which are more suitable for image classification tasks.
