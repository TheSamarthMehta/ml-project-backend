# Backend - Cardiovascular Risk Assessment API

This is the Flask backend API for the cardiovascular risk assessment application.

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models (first time only):**

   ```bash
   python train_models.py
   ```

   This will:

   - Train all 6 machine learning models
   - Save them as `.pkl` files in the `models/` directory
   - Display performance metrics for each model

3. **Run the server:**
   ```bash
   python app.py
   ```
   The API will:
   - Automatically load pre-trained models from `models/` folder
   - If models are not found, it will train them automatically
   - Start on http://localhost:5000

## Models

The following models are trained and saved:

- **logistic-regression.pkl** - Primary model for classification
- **knn.pkl** - K-Nearest Neighbors classifier
- **svm.pkl** - Support Vector Machine classifier
- **ann.pkl** - Artificial Neural Network classifier
- **linear-regression.pkl** - Linear regression model
- **polynomial-regression.pkl** - Polynomial regression model

All models are stored in the `models/` directory.

## API Endpoints

### Health Check

```
GET /health
```

Returns server status and loaded model information.

### Prediction

```
POST /predict
Content-Type: application/json

{
  "age": 45,
  "gender": "male",
  "blood_pressure": 130,
  "cholesterol": 220,
  "heart_rate": 75,
  "diabetes": false,
  "smoking": false,
  "obesity": false
}
```

Returns risk assessment with probability, risk level, and recommendations.

## Model Training

The models are trained on synthetic cardiovascular data with the following features:

- Age
- Gender
- Blood Pressure
- Cholesterol
- Heart Rate
- Diabetes (binary)
- Smoking (binary)
- Obesity (binary)

To retrain models with different parameters or data, run:

```bash
python train_models.py
```

## File Structure

```
backend/
├── app.py                 # Flask API server
├── model_utils.py         # Model training and utilities
├── train_models.py        # Script to train and save models
├── requirements.txt       # Python dependencies
├── models/                # Trained model files
│   ├── logistic-regression.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── ann.pkl
│   ├── linear-regression.pkl
│   └── polynomial-regression.pkl
└── README.md             # This file
```
