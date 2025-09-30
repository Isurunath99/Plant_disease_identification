# 🌿 Plant Disease Detection System

A comprehensive deep learning solution for detecting diseases in plant leaves using computer vision and neural networks. This project implements state-of-the-art EfficientNet-B0 and ResNet50 models to classify plant diseases from leaf images with 93.87% accuracy.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.12+-orange.svg)
![Gradio](https://img.shields.io/badge/Gradio-v4.15+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project uses deep learning to identify plant diseases from leaf images, helping farmers and agricultural professionals make informed decisions about crop health. The system supports three plant types: peppers, potatoes, and tomatoes, detecting 15 different disease conditions.

### 🔍 Supported Plant Diseases

- **Pepper Bell**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy  
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-Spotted), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

## 🚀 Features

- **Multiple Model Architectures**: ResNet50 and EfficientNet-B0
- **High Accuracy**: Achieves 93.87% test accuracy
- **Web Interface**: Interactive Gradio app deployed on Hugging Face
- **Professional Frontend**: Beautiful HTML/CSS/JS interface deployed on Netlify
- **Real-time Predictions**: Fast inference with confidence scores and top-5 predictions
- **Treatment Recommendations**: AI-powered treatment suggestions for each disease
- **Data Augmentation**: Advanced image preprocessing with rotation, flipping, and color jittering
- **Transfer Learning**: Utilizes pre-trained ImageNet weights

## 🌐 Live Demos

🎨 **Professional Frontend**: [https://plantdiseaseidentification.netlify.app/](https://plantdiseaseidentification.netlify.app/)

🤖 **Gradio Backend API**: [https://huggingface.co/spaces/Isurunath/Plant_disease_identification](https://huggingface.co/spaces/Isurunath/Plant_disease_identification)

## 🗃️ Model Architectures

### 1. ResNet50

- Deep residual learning with skip connections
- Pre-trained on ImageNet with fine-tuning
- 77,926 trainable parameters
- Validation accuracy: 92.92%
- Test accuracy: ~92.08%

### 2. EfficientNet-B0 ⭐ (Best Model)

- Compound scaling methodology
- Mobile inverted bottleneck convolutions
- Pre-trained on ImageNet
- 48,678 trainable parameters (37% fewer than ResNet50)
- **Validation accuracy: 95.28%**
- **Test accuracy: 93.87%**
- **Precision: 0.9393 | Recall: 0.9387 | F1-Score: 0.9380**

## 📊 Dataset

- **Source**: PlantVillage Dataset (Kaggle)
- **Images**: 20,638 plant leaf images
- **Classes**: 15 different plant disease categories
- **Split**: 70% Training (14,446), 15% Validation (3,095), 15% Testing (3,097)
- **Resolution**: 224×224 pixels (standardized)

## 🛠️ Installation

### Prerequisites

```bash
python >= 3.8
pip >= 21.0
```

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/Isurunath99/Plant_disease_identification.git
cd Plant_disease_identification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=9.0.0
jupyter>=1.0.0
gradio>=4.15.0
```

## 🎮 Usage

### 1. Training Models

Run the Jupyter notebook to train both models:

```bash
jupyter notebook plantdiseasedetection.ipynb
```

Execute all cells in sequence (Cells 1-20) to:

- Download and prepare the PlantVillage dataset
- Apply preprocessing and data augmentation
- Train ResNet50 and EfficientNet-B0 models
- Evaluate performance with comprehensive metrics
- Generate training curves and confusion matrices
- Save trained models with checkpoints

### 2. Web Application (Hugging Face)

Access the deployed Gradio interface:

🌐 **Live Demo**: [https://huggingface.co/spaces/Isurunath/Plant_disease_identification](https://huggingface.co/spaces/Isurunath/Plant_disease_identification)

The app provides:

- Drag-and-drop image upload
- Real-time disease prediction
- Confidence scores and top-5 predictions
- Disease descriptions and treatment recommendations
- Interactive user interface with modern design

### 3. Professional Frontend

Access the custom HTML/CSS/JS frontend:

🎨 **Frontend Demo**: [https://plantdiseaseidentification.netlify.app/](https://plantdiseaseidentification.netlify.app/)

Features:

- Modern gradient design with smooth animations
- Comprehensive disease information database
- Model performance metrics display
- Mobile-responsive interface
- Drag-and-drop file upload
- Connects to Hugging Face backend API

### 4. Local Testing (Gradio)

For local development and testing:

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

### 5. Command Line Prediction

```python
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model(num_classes):
    model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, num_classes)
    )
    return model

checkpoint = torch.load('trained_models/efficientnet_plant_classifier.pth', map_location=device)
model = create_model(checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Make prediction
image = Image.open('path/to/leaf.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print(f"Disease: {checkpoint['classes'][predicted.item()]}")
print(f"Confidence: {confidence.item()*100:.2f}%")
```

## 📁 Project Structure

```
Plant_disease_identification/
├── plantdiseasedetection.ipynb  # Main training notebook (20 cells)
├── app.py                        # Gradio web application
├── index.html                    # Custom frontend HTML
├── style.css                     # Frontend styling
├── script.js                     # Frontend JavaScript logic
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── Report.pdf                    # Academic report (8 pages)
├── .gitignore                   # Git ignore rules
├── PlantVillage/                # Dataset (auto-downloaded)
│   ├── Pepper__bell___Bacterial_spot/
│   ├── Pepper__bell___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   ├── Potato___healthy/
│   ├── Tomato_Bacterial_spot/
│   ├── Tomato_Early_blight/
│   ├── Tomato_Late_blight/
│   ├── Tomato_leaf_Mold/
│   ├── Tomato_Septoria_leaf_spot/
│   ├── Tomato_Spider_mites_Two_spotted_sp/
│   ├── Tomato__Target_Spot/
│   ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
│   ├── Tomato__Tomato_mosaic_virus/
│   └── Tomato_healthy/
└── trained_models/              # Saved models (auto-generated)
    ├── resnet50_plant_classifier.pth
    └── efficientnet_plant_classifier.pth
```

## 🎯 Model Performance

| Model | Val Accuracy | Test Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|-------------|---------------|-----------|--------|----------|------------|
| ResNet50 | 92.92% | ~92.08% | 0.9210 | 0.9208 | 0.9207 | 77,926 |
| EfficientNet-B0 | 95.28% | 93.87% | 0.9393 | 0.9387 | 0.9380 | 48,678 |

### Key Observations

- EfficientNet-B0 outperformed ResNet50 by 2.36% on validation accuracy
- Achieved 93.87% test accuracy with balanced precision and recall
- 37% fewer trainable parameters than ResNet50, demonstrating superior efficiency
- Small train-validation gap (3.59%) indicates excellent generalization
- High metrics across the board confirm reliability for agricultural deployment

## 📈 Training Details

### Preprocessing & Augmentation

- **Image Resizing**: 224×224 pixels
- **Normalization**: ImageNet statistics
- **Augmentation (Training)**:
  - Random horizontal flip (50%)
  - Random vertical flip (25%)
  - Random rotation (±25°)
  - Color jittering (brightness, contrast, saturation)

### Training Configuration

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (lr=0.001, weight_decay=0.0001)
- **Batch Size**: 32
- **Max Epochs**: 15
- **Early Stopping**: Patience=4 (monitoring validation loss)
- **Regularization**: Dropout (0.4)

### Results Visualization

The notebook generates comprehensive visualizations:

- Training/validation loss curves
- Training/validation accuracy curves
- Confusion matrix (15×15 for all disease classes)
- Per-class accuracy rankings
- Model comparison charts
- Performance metrics comparison

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or use CPU:

```python
device = torch.device('cpu')
```

#### 2. Dataset Download Issues

```
ConnectionError: Failed to download dataset
```

**Solution**:

- Check internet connection
- Manually download from Kaggle: [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- Extract to `PlantVillage/` directory

#### 3. Module Not Found Error

```
ModuleNotFoundError: No module named 'torch'
```

**Solution**: Install dependencies:

```bash
pip install -r requirements.txt
```

#### 4. Gradio API Connection Issues

**Solution**:

- Ensure Hugging Face Space is awake (may take 30 seconds to load)
- Check the replica URL is current
- Clear browser cache

### Performance Optimization

- **GPU Training**: Use CUDA-enabled PyTorch for 10x faster training
- **Data Loading**: Increase `num_workers` in DataLoader for faster data loading
- **Model Quantization**: Convert to TorchScript for production deployment
- **Batch Inference**: Process multiple images simultaneously

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is submitted as part of academic coursework and is for educational purposes only.

## 👨‍💻 Author

**Isurunath**

- GitHub: [@Isurunath99](https://github.com/Isurunath99)
- Hugging Face: [@Isurunath](https://huggingface.co/Isurunath)
- Project Repository: [Plant_disease_identification](https://github.com/Isurunath99/Plant_disease_identification)
- Live Demo: [PlantCare AI](https://plantdiseaseidentification.netlify.app/)

## 🙏 Acknowledgments

- PlantVillage Dataset creators for providing comprehensive plant disease images
- PyTorch Team for the excellent deep learning framework
- Hugging Face for hosting the Gradio application
- Netlify for frontend deployment
- Course Instructor and TAs for guidance and support
- EfficientNet and ResNet authors for groundbreaking architectures

---

⭐ **Star this repository if you find it helpful!**

## 🔮 Future Enhancements

- [ ] Expand to more plant species (corn, grape, apple, etc.)
- [ ] Mobile app development (iOS/Android using TorchScript)
- [ ] Disease severity assessment (mild, moderate, severe)
- [ ] Integration with agricultural IoT devices
- [ ] Real-time disease progression tracking
- [ ] Multi-language support for global adoption
- [ ] Drone-based image capture support
- [ ] Treatment recommendation system with cost estimates
- [ ] Weather correlation analysis
- [ ] Integration with precision agriculture platforms
- [ ] Ensemble models for improved accuracy
- [ ] Attention mechanism visualization (Grad-CAM)
- [ ] Field data collection for domain adaptation

## 📚 References

1. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., Devito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Ai, Q., Steiner, B., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf

2. Plant Village. (n.d.). Kaggle. https://www.kaggle.com/datasets/arjuntejaswi/plant-village

3. PyTorch. (2023). PyTorch. https://pytorch.org/

4. Randellini, E. (2023, January 5). Image classification: ResNet vs EfficientNet vs EfficientNet_v2 vs Compact Convolutional…. Medium. https://medium.com/@enrico.randellini/image-classification-resnet-vs-efficientnet-vs-efficientnet-v2-vs-compact-convolutional-c205838bbf49

5. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei, L. (2015). ImageNet large scale visual recognition challenge. International Journal of Computer Vision, 115(3), 211-252. https://doi.org/10.1007/s11263-015-0816-y
