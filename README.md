# 🔍 TraceFinder - Forensic Scanner Identification System

> **Forensic scanner source identification using deep learning and PRNU (Photo Response Non-Uniformity) pattern analysis.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

**TraceFinder** is a forensic tool that identifies the source scanner device used to scan a document or image. By analyzing unique patterns and artifacts left during the scanning process, the system can distinguish between different scanner brands and models with high accuracy.

### Key Achievements
- **Validation Accuracy:** 92.26%
- **Test Accuracy:** 91.11%
- **Training Accuracy:** 97.93%
- **Scanner Classes:** 11 (Canon, Epson, HP)
- **Technology:** ResNet18 Ensemble with Test-Time Augmentation

### Use Cases

**Digital Forensics**
- Identify source scanner used to forge legal documents
- Detect fake certificates and unauthorized reproductions

**Document Authentication**
- Verify authenticity of scanned certificates and agreements
- Differentiate between scans from authorized and unauthorized devices

**Legal Evidence Verification**
- Ensure scanned documents came from authorized devices
- Validate document origin in court proceedings

**Copyright Protection**
- Track document origin for intellectual property cases
- Identify unauthorized document reproduction

---

## ✨ Features

### Machine Learning Pipeline
- PRNU (Photo Response Non-Uniformity) feature extraction
- Bilateral filtering for edge-preserving noise isolation
- Ensemble of 3 independently trained ResNet18 models
- Test-Time Augmentation with 10x augmentations
- Label smoothing for improved generalization
- Advanced regularization techniques

### Web Application
- Interactive Streamlit-based user interface
- Drag-and-drop image upload (TIF, PNG, JPG, BMP)
- Real-time scanner prediction with confidence scores
- Top-5 alternative predictions with probability distribution
- Interactive charts and visualization
- Prediction history and statistics dashboard
- Export results as JSON or CSV
- Automatic prediction logging

### Model Explainability
- Grad-CAM visualizations showing model focus areas
- Feature importance analysis
- Confidence distribution analysis
- Misclassification pattern identification
- Per-class performance metrics

---

## 📊 Dataset

### Source
**SUPATLANTIQUE Scanner Dataset**

### Statistics
- **Total Images:** 4,568 scanned documents
- **Scanner Models:** 11 different scanners
- **Document Types:** Officials, Wikipedia pages
- **Resolutions:** 150 DPI and 300 DPI
- **Image Format:** TIFF, grayscale

### Scanner Classes
**Canon Scanners:** Canon120-1, Canon120-2, Canon220, Canon9000-1, Canon9000-2

**Epson Scanners:** EpsonV370-1, EpsonV370-2, EpsonV39-1, EpsonV39-2, EpsonV550

**HP Scanners:** HP

### Data Split
- **Training Set:** 70% (3,197 images)
- **Validation Set:** 15% (685 images)
- **Test Set:** 15% (686 images)

---

## 📈 Results

### Overall Performance

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Training | 97.93% | 0.9795 | 0.9793 | 0.9793 |
| Validation | 92.26% | 0.9231 | 0.9226 | 0.9225 |
| Test | 91.11% | 0.9118 | 0.9111 | 0.9110 |

### Per-Class Performance (Selected Examples)

| Scanner Model | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Canon120-1 | 0.9545 | 0.9130 | 0.9333 |
| EpsonV370-1 | 0.9333 | 0.8750 | 0.9032 |
| HP | 0.9500 | 0.9268 | 0.9383 |

### Key Findings
- High accuracy across all scanner brands
- Excellent discrimination between different manufacturers
- Some confusion between similar models (e.g., Canon120-1 vs Canon120-2)
- Model confidence correlates strongly with prediction accuracy
- Test-Time Augmentation provides 1-2% accuracy boost

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB disk space for dataset and models
- Internet connection for package installation

### Step-by-Step Installation

**1. Clone the Repository**

Download or clone the project repository to your local machine.

**2. Create Virtual Environment**

Create and activate a Python virtual environment to isolate dependencies.

**3. Install Dependencies**

Install all required Python packages using the provided requirements file.

**4. Download Dataset**

Obtain the SUPATLANTIQUE dataset and place it in the data folder.

**5. Download Pre-trained Models**

Download the trained model files and place them in the milestone_3 folder.

---

## 💻 Usage

### Option 1: Web Application

**Launch the Streamlit App**

Run the app using Streamlit command. The application will open in your web browser at localhost:8501.

**Using the Interface**

1. Upload a scanned image using the file uploader
2. Click the "Identify Scanner" button
3. View prediction results with confidence scores
4. Explore top-5 alternative predictions
5. Download results as JSON or CSV
6. Check prediction statistics and history

### Option 2: Python API

**Programmatic Usage**

Load the trained model and use it to predict scanner sources directly from Python code. Suitable for batch processing and integration into other applications.

### Option 3: Command Line

**Terminal-based Predictions**

Run predictions from the command line with options for input file, output format, and TTA settings.

---

## 📁 Project Structure

The project is organized into the following main directories:

**Root Directory**
- Main application file (app.py)
- Configuration files (requirements.txt, README.md, LICENSE)
- Documentation files

**Notebooks Folder**
- Milestone 1: Data collection and preprocessing
- Milestone 2: Feature engineering and baseline models
- Milestone 3: Deep learning model development
- Week 6: Model evaluation and explainability

**Source Code Folder**
- Model architecture definitions
- Dataset loading and preprocessing utilities
- Training and evaluation scripts
- Helper functions and utilities

**Models Folder**
- Trained model checkpoints
- Best performing models
- Ensemble model weights
- Training history and logs

**Data Folder**
- Raw scanned images (not included in repository)
- Processed and normalized images
- Train/validation/test splits

**Results Folder**
- Confusion matrices
- Performance metrics
- Grad-CAM visualizations
- Classification reports
- Prediction logs

---

## 🔬 Methodology

### Milestone 1: Data Collection & Preprocessing (Weeks 1-2)

**Data Collection**
- Collected 4,568 scanned images from 11 scanner models
- Organized images by scanner model, document type, and DPI
- Created comprehensive labeling system

**Preprocessing**
- Resized all images to consistent dimensions (256×256 pixels)
- Converted to grayscale format
- Normalized pixel intensities to 0-1 range
- Applied denoising to remove content while preserving scanner artifacts

**Data Validation**
- Analyzed image properties (resolution, format, color channels)
- Verified label accuracy
- Created stratified train/validation/test splits

### Milestone 2: Feature Engineering & Baseline Models (Weeks 3-4)

**Feature Extraction**
- **PRNU (Photo Response Non-Uniformity):** Extracted unique sensor noise patterns
- **Local Binary Patterns (LBP):** Captured texture descriptors
- **FFT Analysis:** Analyzed frequency domain characteristics
- **Noise Residuals:** Computed difference between original and denoised images

**Baseline Models**
- Logistic Regression: 22.5% accuracy
- Support Vector Machine (SVM): 22.5% accuracy  
- Random Forest: 47.5% accuracy

**Key Insights**
- Hand-crafted features showed limited discriminative power
- Complex patterns required deep learning approach
- Random Forest outperformed linear models significantly

### Milestone 3: Deep Learning Development (Weeks 5-6)

**Week 5: Model Training**

**Architecture Design**
- Base model: ResNet18 with pre-trained weights
- Modified input layer for grayscale images
- Custom classification head with dropout layers
- Total parameters: approximately 11 million

**Training Strategy**
- Image size increased to 224×224 for better feature capture
- Bilateral filtering for superior PRNU extraction
- Strong data augmentation (rotation, flips, noise, brightness)
- Label smoothing (0.15) to prevent overconfidence
- Mixup augmentation for improved generalization
- Cosine annealing learning rate schedule
- Gradient clipping for training stability
- Early stopping with patience mechanism

**Ensemble Approach**
- Trained 3 models with different random initializations
- Each model learns slightly different patterns
- Predictions averaged for final output
- Significantly improved robustness

**Results After Week 5**
- Validation accuracy: 89.78% (single model)
- Improved to 92.26% with ensemble and TTA

**Week 6: Evaluation & Explainability**

**Comprehensive Evaluation**
- Calculated accuracy, precision, recall, F1-score for all datasets
- Generated confusion matrices showing per-class performance
- Analyzed prediction confidence distributions
- Identified misclassification patterns

**Explainability Analysis**
- **Grad-CAM Visualizations:** Heat maps showing where model focuses
- **Feature Importance:** Identified key discriminative regions
- **Confidence Analysis:** Correct predictions have higher confidence
- **Pattern Recognition:** Model correctly identifies PRNU patterns

**Key Discoveries**
- Model focuses on texture patterns and noise characteristics
- High confidence (>80%) on well-separated scanner types
- Confusion mainly between similar models from same manufacturer
- Edge artifacts and compression patterns are key features

### Week 7: Deployment

**Web Application Development**
- Built interactive Streamlit interface
- Implemented real-time prediction pipeline
- Created visualization components for results
- Added prediction logging and export functionality

**Production Features**
- Error handling and input validation
- Automatic model loading and caching
- Responsive design for different screen sizes
- Statistics dashboard with metrics tracking

---

## 🛠️ Technologies Used

### Core Technologies

**Deep Learning Framework**
- PyTorch 2.2.0 for model development and training
- TorchVision for pre-trained models and transformations
- GPU acceleration support (optional)

**Computer Vision**
- OpenCV 4.9.0 for image processing operations
- Pillow for image loading and manipulation
- Custom PRNU extraction algorithms

**Web Framework**
- Streamlit 1.31.0 for interactive web application
- Plotly for interactive visualizations
- HTML/CSS for custom styling

**Data Science**
- NumPy for numerical computations
- Pandas for data manipulation and logging
- Scikit-learn for metrics and baseline models
- Matplotlib and Seaborn for static visualizations

### Model Architecture

**Base Model:** ResNet18 (Residual Neural Network)
- Pre-trained on ImageNet
- Modified for grayscale input
- Custom classification layers

**Input Processing**
- PRNU residual extraction using bilateral filtering
- Image resizing to 160×160 or 224×224 pixels
- Normalization to zero mean and unit variance

**Optimization Techniques**
- Label smoothing for better calibration
- Mixup augmentation for generalization
- Test-Time Augmentation for robust inference
- Ensemble voting for reduced variance
- Gradient clipping for stable training
- Cosine annealing learning rate schedule

---

## 🎓 Learning Outcomes

This project demonstrates comprehensive skills in:

**Machine Learning & Deep Learning**
- Deep neural network architecture design
- Transfer learning and fine-tuning strategies
- Ensemble methods and model averaging
- Hyperparameter optimization
- Regularization techniques

**Computer Vision**
- Image preprocessing and augmentation
- Feature extraction (PRNU, LBP, FFT)
- Noise pattern analysis
- Scanner fingerprint identification

**Model Development**
- Training pipeline implementation
- Loss function design
- Learning rate scheduling
- Early stopping and checkpointing
- Performance monitoring

**Model Evaluation**
- Comprehensive metrics calculation
- Confusion matrix analysis
- Cross-validation strategies
- Confidence calibration
- Error analysis

**Explainability & Interpretability**
- Grad-CAM visualization implementation
- Feature importance analysis
- Model behavior understanding
- Decision process transparency

**Software Engineering**
- Web application development
- API design and implementation
- Code organization and documentation
- Version control practices
- Production deployment

**Domain Knowledge**
- Digital forensics principles
- Scanner technology understanding
- Document authentication methods
- Legal evidence requirements

---

## 📚 References

### Academic Papers

**PRNU-based Source Attribution**

Research on photo response non-uniformity as a unique sensor fingerprint for device identification.

**Scanner Identification Methods**

Studies on identifying scanner sources through pattern noise analysis and machine learning techniques.

**Deep Learning for Forensics**

Applications of convolutional neural networks in forensic image analysis and device identification.

**Transfer Learning**

Techniques for leveraging pre-trained models in specialized domains with limited data.

### Datasets

**SUPATLANTIQUE Scanner Dataset**

Comprehensive collection of scanned documents from multiple scanner models used for forensic research.

### Tools & Frameworks

**PyTorch**

Open-source machine learning framework for building and training neural networks.

**OpenCV**

Computer vision library for image processing and analysis operations.

**Streamlit**

Framework for creating interactive data applications and machine learning demos.

**Scikit-learn**

Machine learning library for classical algorithms and evaluation metrics.

---

## 🤝 Contributing

Contributions to improve TraceFinder are welcome. Areas for potential contribution include:

- Support for additional scanner models and brands
- Mobile application development for on-device inference
- Real-time video stream analysis capabilities
- Enhanced visualization and reporting tools
- Multi-language support for international users
- REST API development for system integration
- Docker containerization for easier deployment
- Performance optimization for faster inference
- Documentation improvements and tutorials
- Test coverage expansion

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for full details.

The MIT License allows free use, modification, and distribution of this software for both commercial and non-commercial purposes, with attribution to the original authors.

---

## 🙏 Acknowledgments

**Dataset Providers**

Thanks to the creators and maintainers of the SUPATLANTIQUE scanner dataset for making this research possible.

**Open Source Community**

Gratitude to the PyTorch, OpenCV, and Streamlit communities for excellent documentation and support.

**Academic Institutions**

Recognition of the computational resources and academic support provided during this project.

**Research Community**

Appreciation for prior research in digital forensics and scanner identification that informed this work.

---

## 📞 Contact

**Project Repository**

Access the complete source code, documentation, and issues tracker on GitHub.

**Technical Support**

For bug reports, feature requests, or technical questions, please create an issue on the GitHub repository.

**Collaboration Inquiries**

For research collaboration, commercial licensing, or partnership opportunities, please reach out via email.

**Professional Network**

Connect on LinkedIn for professional networking and project updates.

---

## 📊 Project Statistics

**Development Timeline:** 8 weeks (Milestones 1-4)

**Code Metrics**
- Total lines of code: Approximately 2,500+
- Python modules: 10+
- Jupyter notebooks: 4
- Training time: 2-3 hours on CPU
- Inference time: 3-5 seconds per image with TTA

**Dataset Metrics**
- Total images processed: 4,568
- Image preprocessing operations: 10,000+
- Model training iterations: 15,000+
- Prediction logs generated: Growing database

**Performance Metrics**
- Validation accuracy improvement: 89.78% → 92.26%
- Total accuracy gain: +48% from baseline Random Forest
- Ensemble improvement: +2.48% over single model
- TTA improvement: +1-2% over base prediction

---

## 🌟 Future Enhancements

**Planned Features**
- Support for additional scanner manufacturers
- Multi-page document batch processing
- Confidence threshold customization
- Export to multiple report formats
- Integration with document management systems
- Mobile app for iOS and Android
- Cloud deployment with API access
- Real-time monitoring dashboard

**Research Directions**
- Printer source identification
- Camera model identification
- Combined scanner-printer attribution
- Deepfake document detection
- Cross-domain transfer learning

---

## ✅ Project Completion Checklist

**Milestone 1: Data Collection & Preprocessing**
- Dataset acquired and organized
- Preprocessing pipeline implemented
- Data splits created and validated

**Milestone 2: Feature Engineering & Baseline Models**
- PRNU extraction implemented
- LBP and FFT features computed
- Baseline models trained and evaluated

**Milestone 3: Deep Learning Model Development**
- ResNet18 architecture implemented
- Advanced training techniques applied
- Ensemble model created
- Validation accuracy >92% achieved

**Milestone 4: Deployment & Documentation**
- Streamlit web application developed
- Prediction logging system implemented
- Comprehensive evaluation completed
- Full documentation provided
- Project ready for production use

---

<div align="center">

**TraceFinder - Making Digital Forensics Accessible**

**Version 1.0 | 2025**

</div>
