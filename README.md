# 🌾 Crop Recommendation System

An intelligent crop recommendation web application that predicts the most suitable crop to grow based on soil nutrient levels and climatic conditions. The machine learning model, data preprocessing, and training pipeline were built entirely from scratch.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikit-learn&logoColor=white)

---

## 📖 Overview

Choosing the right crop is critical for maximizing yield and sustainability. This system takes **7 key agricultural parameters** as input and uses a trained machine learning model to recommend the best crop for the given conditions.

### Input Features

| Parameter       | Description                              | Range       |
| --------------- | ---------------------------------------- | ----------- |
| **Nitrogen (N)**    | Ratio of nitrogen content in the soil    | 0 – 140     |
| **Phosphorus (P)**  | Ratio of phosphorus content in the soil  | 5 – 145     |
| **Potassium (K)**   | Ratio of potassium content in the soil   | 5 – 205     |
| **Temperature**     | Temperature in degrees Celsius           | 8 – 45 °C   |
| **Humidity**        | Relative humidity in percentage          | 12 – 100 %  |
| **pH Value**        | pH value of the soil                     | 3 – 10      |
| **Rainfall**        | Rainfall in millimeters                  | 20 – 300 mm |

### Supported Crops

The model can recommend **22 different crops**:

> Rice · Maize · Chickpea · Kidney Beans · Pigeon Peas · Moth Beans · Mung Bean · Black Gram · Lentil · Pomegranate · Banana · Mango · Grapes · Watermelon · Muskmelon · Apple · Orange · Papaya · Coconut · Cotton · Jute · Coffee

---

## 🧠 Machine Learning

The entire ML pipeline — **data preprocessing, feature engineering, model selection, and training** — was designed and implemented from scratch.

- **Model**: Trained using scikit-learn and serialized as `crop_prediction.pkl` via `joblib`
- **Input**: 7 numerical features (N, P, K, temperature, humidity, pH, rainfall)
- **Output**: Predicted crop name (one of 22 classes)

---

## 🛠️ Tech Stack

| Layer         | Technology                  |
| ------------- | --------------------------- |
| Frontend / UI | Streamlit                   |
| ML Framework  | scikit-learn                |
| Data Handling | pandas                      |
| Serialization | joblib                      |
| Package Mgmt  | uv                          |
| Language      | Python 3.12                 |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AdityaJare/crop-recommendation-app.git
   cd crop-recommendation-app
   ```

2. **Install dependencies**

   Using `uv` (recommended):

   ```bash
   uv sync
   ```

   Or using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   uv run streamlit run main.py
   ```

   Or without `uv`:

   ```bash
   streamlit run main.py
   ```

4. **Open in browser** — Navigate to `http://localhost:8501`

---

## 📂 Project Structure

```
crop-recommendation-app/
├── main.py                 # Streamlit application entry point
├── crop_prediction.pkl     # Trained ML model (serialized)
├── pyproject.toml          # Project metadata & dependencies
├── requirements.txt        # Pip-compatible dependency list
├── uv.lock                 # Locked dependency versions (uv)
├── .python-version         # Python version specification
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## 🖥️ Usage

1. Adjust the **sliders** for each soil and climate parameter to match your field conditions.
2. Click the **"Predict Crop"** button.
3. The app will display:
   - ✅ The **recommended crop name**
   - 🖼️ A **reference image** of the predicted crop

---

## 📦 Dependencies

| Package        | Version   |
| -------------- | --------- |
| streamlit      | ≥ 1.64.0  |
| scikit-learn   | 1.9.0     |
| pandas         | ≥ 3.0.6   |
| joblib         | ≥ 1.6.0   |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📝 License

This project is open source and available for personal and educational use.

---

## 👤 Author

**AdityaJare**

Built with ❤️ as a hands-on machine learning project — from data preprocessing and model training to deployment.
