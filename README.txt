================================================================================
                    CATS VS. DOGS IMAGE CLASSIFIER PROJECT
================================================================================

PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project implements a Convolutional Neural Network (CNN) using TensorFlow 
and Keras to classify images of cats and dogs. The classifier is built following 
Course 2 concepts with proper data handling, augmentation, and training practices.

DATE: February 3, 2026
FRAMEWORK: TensorFlow/Keras
PYTHON VERSION: 3.x


PROJECT STRUCTURE
--------------------------------------------------------------------------------
vs-code (jupyter)/
│
├── Image-Classifier.ipynb          # Main Jupyter notebook with complete code
├── cats_and_dogs_filtered.zip      # Downloaded dataset (compressed)
├── test_image.jpg                  # Sample image for testing predictions
├── vectorize.py                    # Additional preprocessing script
├── README.txt                      # This file
│
├── cats_and_dogs_filtered/         # Extracted dataset directory
│   ├── train/                      # Training data (2000 images)
│   │   ├── cats/                   # 1000 cat training images
│   │   └── dogs/                   # 1000 dog training images
│   │
│   └── validation/                 # Validation data (1000 images)
│       ├── cats/                   # 500 cat validation images
│       └── dogs/                   # 500 dog validation images
│
└── tf-env/                         # Python virtual environment
    ├── Scripts/                    # Activation scripts (Windows)
    └── Lib/site-packages/          # Installed Python packages


DATASET INFORMATION
--------------------------------------------------------------------------------
Source: Google ML Education Datasets
URL: https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip
Total Images: 3000
  - Training: 2000 images (1000 cats, 1000 dogs)
  - Validation: 1000 images (500 cats, 500 dogs)
Image Format: JPEG
Original Size: Variable (resized to 150x150 during preprocessing)


REQUIREMENTS
--------------------------------------------------------------------------------
Python Packages (installed in tf-env/):
  - tensorflow >= 2.x
  - keras
  - numpy
  - matplotlib
  - jupyter
  - pillow

Hardware:
  - CPU: Intel/AMD processor (GPU optional for faster training)
  - RAM: Minimum 4GB (8GB recommended)
  - Storage: ~1GB for dataset and models


INSTALLATION & SETUP
--------------------------------------------------------------------------------
1. Activate Virtual Environment:
   Windows:
     tf-env\Scripts\activate
   
   Linux/Mac:
     source tf-env/bin/activate

2. Verify Installation:
   python -c "import tensorflow as tf; print(tf.__version__)"

3. Launch Jupyter Notebook:
   jupyter notebook Image-Classifier.ipynb

4. Run all cells sequentially


NOTEBOOK STRUCTURE
--------------------------------------------------------------------------------
Cell 1: Import Libraries
  - TensorFlow, Keras, OS, zipfile, urllib, matplotlib

Cell 2: Download & Extract Dataset
  - Downloads cats_and_dogs_filtered.zip using urllib with User-Agent
  - Extracts to current directory using zipfile
  - Avoids 403 Forbidden errors

Cell 3: Set Up Data Directories
  - Defines paths for train/validation directories
  - Verifies directory existence
  - Counts images in each category

Cell 4: Create Data Generators
  - Training Generator:
    * Pixel normalization (rescale=1./255)
    * Data augmentation (rotation_range=40, horizontal_flip=True)
  - Validation Generator:
    * Pixel normalization only (no augmentation)
  - Both use:
    * target_size=(150, 150)
    * batch_size=32
    * class_mode='binary'

Cell 5: Build CNN Model
  - Architecture:
    * Conv2D(32, 3x3) + ReLU + MaxPooling2D(2x2)
    * Conv2D(64, 3x3) + ReLU + MaxPooling2D(2x2)
    * Conv2D(128, 3x3) + ReLU + MaxPooling2D(2x2)
    * Flatten
    * Dense(512) + ReLU
    * Dense(1) + Sigmoid (output layer)
  - Total Parameters: ~7.5 million

Cell 6: Compile Model
  - Optimizer: Adam
  - Loss Function: binary_crossentropy
  - Metrics: accuracy

Cell 7: Train Model
  - Epochs: 10-15
  - Steps per epoch: ~63 (2000 images / 32 batch size)
  - Validation steps: ~32 (1000 images / 32 batch size)
  - Training time: ~5-10 minutes on CPU per epoch

Cell 8: Plot Training History
  - Generates two plots:
    * Training vs. Validation Accuracy
    * Training vs. Validation Loss
  - Visualizes model performance and potential overfitting


MODEL ARCHITECTURE DETAILS
--------------------------------------------------------------------------------
Layer (type)                 Output Shape              Param #
================================================================================
conv2d_1 (Conv2D)           (None, 148, 148, 32)      896
max_pooling2d_1             (None, 74, 74, 32)        0
conv2d_2 (Conv2D)           (None, 72, 72, 64)        18,496
max_pooling2d_2             (None, 36, 36, 64)        0
conv2d_3 (Conv2D)           (None, 34, 34, 128)       73,856
max_pooling2d_3             (None, 17, 17, 128)       0
flatten (Flatten)           (None, 36992)             0
dense_1 (Dense)             (None, 512)               18,940,416
dense_2 (Dense)             (None, 1)                 513
================================================================================
Total params: 19,034,177
Trainable params: 19,034,177
Non-trainable params: 0


KEY CONCEPTS IMPLEMENTED
--------------------------------------------------------------------------------
1. Data Augmentation:
   - Prevents overfitting by artificially increasing dataset diversity
   - Applies random transformations only to training data
   - Validation data remains unchanged for consistent evaluation

2. Binary Classification:
   - Sigmoid activation outputs probability [0, 1]
   - Threshold 0.5: <0.5 = Cat, >=0.5 = Dog
   - Binary crossentropy loss optimized for two-class problems

3. CNN Architecture:
   - Convolutional layers extract hierarchical features
   - MaxPooling reduces spatial dimensions and computational cost
   - Dense layers perform final classification

4. Train/Validation Split:
   - Training set: Model learns patterns
   - Validation set: Monitors generalization to unseen data
   - Prevents overfitting detection

5. Normalization:
   - Rescales pixel values from [0, 255] to [0, 1]
   - Improves training stability and convergence speed


EXPECTED RESULTS
--------------------------------------------------------------------------------
After 10-15 epochs:
  - Training Accuracy: 85-95%
  - Validation Accuracy: 70-85%
  - Training Loss: 0.2-0.4
  - Validation Loss: 0.4-0.6

Note: Gap between training and validation metrics indicates overfitting
To reduce overfitting:
  - Add Dropout layers (0.3-0.5)
  - Increase data augmentation
  - Use L2 regularization
  - Reduce model complexity


TROUBLESHOOTING
--------------------------------------------------------------------------------
Issue: 403 Forbidden Error during download
Solution: Use urllib.request with User-Agent header (already implemented)

Issue: "One or more data directories are missing"
Solution: Ensure notebook runs from project root, not tf-env/ directory
         Verify base_dir = 'cats_and_dogs_filtered'

Issue: Out of Memory (OOM) errors
Solution: Reduce batch_size from 32 to 16 or 8

Issue: Slow training on CPU
Solution: Use Google Colab with GPU or install CUDA-enabled TensorFlow

Issue: Model not improving
Solution: Increase epochs, adjust learning rate, or modify architecture


USAGE EXAMPLES
--------------------------------------------------------------------------------
1. Make Predictions on New Images:
   
   from tensorflow.keras.preprocessing import image
   import numpy as np
   
   img = image.load_img('test_image.jpg', target_size=(150, 150))
   img_array = image.img_to_array(img)
   img_array = np.expand_dims(img_array, axis=0)
   img_array /= 255.0
   
   prediction = model.predict(img_array)
   if prediction[0] < 0.5:
       print("It's a CAT!")
   else:
       print("It's a DOG!")

2. Save Trained Model:
   
   model.save('cats_dogs_classifier.h5')

3. Load Saved Model:
   
   from tensorflow.keras.models import load_model
   model = load_model('cats_dogs_classifier.h5')


FUTURE IMPROVEMENTS
--------------------------------------------------------------------------------
1. Transfer Learning:
   - Use pre-trained models (VGG16, ResNet, MobileNet)
   - Fine-tune on cats/dogs dataset
   - Achieve 95%+ accuracy

2. Advanced Augmentation:
   - Width/height shifts
   - Zoom range
   - Brightness adjustments
   - Elastic transformations

3. Regularization Techniques:
   - Dropout layers
   - Batch normalization
   - L2 weight decay

4. Hyperparameter Tuning:
   - Learning rate scheduling
   - Different optimizers (SGD, RMSprop)
   - Batch size experiments

5. Model Deployment:
   - Convert to TensorFlow Lite for mobile
   - Deploy as REST API using Flask/FastAPI
   - Create web interface for image uploads


REFERENCES & RESOURCES
--------------------------------------------------------------------------------
- TensorFlow Documentation: https://www.tensorflow.org/api_docs
- Keras Documentation: https://keras.io/
- Course 2 Materials
- Dataset Source: Google ML Education Datasets
- Research Paper: ImageNet Classification with Deep CNNs (Krizhevsky et al.)


VERSION HISTORY
--------------------------------------------------------------------------------
v1.0 (February 3, 2026)
  - Initial implementation
  - Basic CNN architecture
  - Data augmentation
  - Training/validation split
  - Visualization plots

================================================================================
                              END OF README
================================================================================