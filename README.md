# 🚗 Road Accident Severity Prediction System

An AI-powered web application that predicts the severity of road accidents using Machine Learning. Built with **Random Forest Classifier** and a modern **React + Flask** stack.

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18.2.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 About

The **Road Accident Severity Prediction System** is a machine learning project designed to predict the severity of road accidents based on various factors such as driver information, vehicle details, road conditions, and environmental factors.

**Prediction Categories:**
- 🟢 **Slight Injury** - Minor injuries
- 🟠 **Serious Injury** - Significant injuries requiring medical attention
- 🔴 **Fatal Injury** - Life-threatening or fatal accidents

The system provides:
- Real-time severity predictions
- Probability distribution across all severity levels
- Safety recommendations based on predicted risk
- Detailed scenario analysis

---

## ✨ Features

### Core Functionality
- ✅ **Real-time Predictions** - Instant accident severity assessment
- 📊 **Probability Analysis** - Confidence scores for each severity level
- 🛡️ **Safety Recommendations** - Tailored safety advice based on risk level
- 📈 **Model Metrics** - Transparent model performance information
- 🎨 **Modern Dark UI** - Professional, user-friendly interface

### Technical Features
- 🤖 **Random Forest ML Model** - 74.68% accuracy
- ⚖️ **SMOTE Balancing** - Handles imbalanced dataset
- 🔄 **RESTful API** - Flask backend with CORS support
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile
- 🌙 **Dark Theme** - Easy on the eyes with modern aesthetics

---

## 🛠️ Technology Stack

### Frontend
- **React** 18.2.0 - UI framework
- **Axios** - HTTP client for API requests
- **CSS3** - Modern styling with dark theme
- **JavaScript ES6+** - Modern JavaScript features

### Backend
- **Flask** 3.0.0 - Web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Python** 3.8+ - Programming language

### Machine Learning
- **scikit-learn** 1.3.2 - ML algorithms
- **imbalanced-learn** - SMOTE for class balancing
- **pandas** - Data manipulation
- **numpy** - Numerical computing

---

## 📁 Project Structure

```
RoadAccident/
│
├── backend/
│   ├── app.py                          # Flask API server
│   ├── train_model_random_forest.py    # ML training script
│   ├── cleaned_data.csv                # Training dataset
│   ├── random_forest_model.pkl         # Trained model
│   ├── label_encoders.pkl              # Feature encoders
│   ├── feature_info.pkl                # Feature metadata
│   └── requirements.txt                # Python dependencies
│
└── frontend/
    ├── public/
    ├── src/
    │   ├── App.js                      # Main React component
    │   ├── App.css                     # Styling
    │   ├── components/
    │   │   ├── PredictionForm.js       # Input form
    │   │   ├── Results.js              # Results display
    │   │   └── ModelInfo.js            # Model information modal
    │   └── index.js                    # React entry point
    ├── package.json
    └── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 14.0 or higher
- npm 6.0 or higher

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/road-accident-prediction.git
cd road-accident-prediction
```

### Step 2: Backend Setup

```bash
# Navigate to backend folder
cd backend

# Install Python dependencies
python -m pip install -r requirements.txt

# Train the model (takes 5-10 minutes)
python train_model_random_forest.py

# Start Flask server
python app.py
```

Backend will run on `http://localhost:5000`

### Step 3: Frontend Setup

```bash
# Open new terminal
cd frontend

# Install Node dependencies
npm install

# Start React development server
npm start
```

Frontend will run on `http://localhost:3000`

---

## 💻 Usage

### Making a Prediction

1. **Open the application** at `http://localhost:3000`
2. **Fill in the accident details:**
   - Driver Information (age, experience, education)
   - Vehicle Details (type, ownership)
   - Time & Location (day, time period, area)
   - Road Conditions (surface, alignment, junction type)
   - Environmental Factors (weather, lighting)
   - Accident Details (collision type, cause, casualties)

3. **Click "Predict Severity"** to get results
4. **Review the prediction:**
   - Predicted severity level
   - Confidence score
   - Probability distribution
   - Safety recommendations

### Viewing Model Information

Click the **"ℹ️ Model Info"** button in the header to see:
- Model architecture details
- Performance metrics (accuracy, F1-score, precision, recall)
- Training information
- Prediction classes

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 74.68% |
| **F1-Score** | 72.43% |
| **Precision** | 73.15% |
| **Recall** | 71.89% |

### Model Specifications
- **Algorithm:** Random Forest Classifier
- **Technique:** Ensemble Learning (Bagging)
- **Trees:** 150 estimators
- **Max Depth:** 12
- **Balancing:** SMOTE (Synthetic Minority Over-sampling)
- **Features:** 23 input features
- **Training Samples:** 7,873 records
- **Classes:** 3 (Fatal, Serious, Slight)

---

## 📸 Screenshots

### Main Prediction Interface
![Prediction Form](screenshots/prediction-form.png)

### Results Display
![Results Panel](screenshots/results-panel.png)

### Model Information
![Model Info Modal](screenshots/model-info.png)

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
```http
GET /
```
**Response:**
```json
{
  "status": "online",
  "message": "Road Accident Severity Prediction API",
  "model": "Random Forest Classifier"
}
```

#### 2. Get Features
```http
GET /features
```
**Response:**
```json
{
  "features": ["Age_band_of_driver", "Sex_of_driver", ...],
  "categorical_options": {
    "Age_band_of_driver": ["18-30", "31-50", "Over 51"],
    ...
  }
}
```

#### 3. Get Model Info
```http
GET /model-info
```
**Response:**
```json
{
  "model_type": "Random Forest Classifier",
  "metrics": {
    "accuracy": "74.68%",
    "f1_score": "72.43%"
  },
  "total_features": 23
}
```

#### 4. Make Prediction
```http
POST /predict
Content-Type: application/json

{
  "Age_band_of_driver": "18-30",
  "Sex_of_driver": "Male",
  ...
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "severity": "Serious Injury",
    "risk_level": "High",
    "confidence": 85.2
  },
  "probabilities": {
    "Fatal injury": 10.5,
    "Serious Injury": 85.2,
    "Slight Injury": 4.3
  }
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React code
- Write meaningful commit messages
- Add tests for new features
- Update documentation as needed

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Dataset source: [Road Accident Dataset]
- Random Forest algorithm: scikit-learn
- UI inspiration: Modern dark theme design patterns
- Special thanks to Vidyalankar Institute of Technology

---

## 📞 Contact

For questions, suggestions, or issues:

- **Email:** your.email@example.com
- **GitHub Issues:** [Create an issue](https://github.com/yourusername/road-accident-prediction/issues)
- **LinkedIn:** [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

## 🔄 Future Enhancements

- [ ] Analytics dashboard with visualizations
- [ ] Historical prediction logging
- [ ] Batch prediction support
- [ ] Export predictions to PDF/CSV
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Real-time traffic data integration
- [ ] Ensemble model comparison

---

## 📚 References

1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for safer roads

</div>
