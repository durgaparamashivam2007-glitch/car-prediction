# 🚗 Car Price Predictor - Machine Learning Web Application

An end-to-end, professional, and deployment-ready Machine Learning web application built using **Streamlit**, **Scikit-learn**, and **Plotly** to estimate resale car selling prices in **Indian Rupees (₹)** based exclusively on the **`car details v3.csv`** dataset.

---

## 📌 Project Overview & Problem Statement

Used car prices depend on multiple complex factors such as vehicle age, kilometers driven, fuel type, transmission, seller type, engine capacity, power output, and brand equity. This application leverages machine learning to estimate the resale selling price of a used car based on historical transaction data from **`car details v3.csv`**.

---

## 🎯 Target Column & Input Features

### Target Column
- **`selling_price`**: Continuous numerical target variable representing the resale selling price in Indian Rupees (₹).

### Input Features Used (11 Features)
1. **`brand`**: Car manufacturer brand extracted from the car full `name` column (32 unique brands e.g. Maruti, Hyundai, Honda, Toyota, Mahindra, BMW, Mercedes-Benz, etc.).
2. **`year`**: Manufacturing year of the car.
3. **`km_driven`**: Total distance driven in kilometers.
4. **`fuel`**: Fuel type (`Petrol`, `Diesel`, `CNG`, `LPG`).
5. **`seller_type`**: Type of seller (`Individual`, `Dealer`, `Trustmark Dealer`).
6. **`transmission`**: Transmission type (`Manual`, `Automatic`).
7. **`owner`**: Ownership history (`First Owner`, `Second Owner`, `Third Owner`, `Fourth & Above Owner`, `Test Drive Car`).
8. **`mileage_kmpl`**: Extracted numeric fuel efficiency from `mileage` text (e.g. `23.4 kmpl` -> `23.4`).
9. **`engine_cc`**: Extracted numeric engine displacement in CC from `engine` text (e.g. `1248 CC` -> `1248`).
10. **`max_power_bhp`**: Extracted numeric horsepower from `max_power` text (e.g. `74.02 bhp` -> `74.02`).
11. **`seats`**: Seating capacity parsed to integer range (2 to 14 seats).

---

## 🧹 Machine Learning & Data Pipeline

1. **Dataset Cleaning**:
   - 8,128 raw records cleaned to **6,926 unique records** by removing 1,202 exact duplicate entries.
2. **Feature Extraction**:
   - Extracted float numerical values from text columns (`mileage`, `engine`, `max_power`) using Regex.
   - Extracted manufacturer `brand` from the car name.
3. **Scikit-Learn Pipeline**:
   - `SimpleImputer(strategy='median')` + `StandardScaler()` for numerical features.
   - `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` for categorical features.
   - Combined into a `ColumnTransformer` and serialized into `models/car_price_pipeline.pkl`.

---

## 📊 Model Benchmark & Results

Six regression algorithms were evaluated using an **80/20 Train-Test split**:

| Model | MAE (₹) | RMSE (₹) | R² Score |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** 🏆 | **₹ 73,454.13** | **₹ 1,28,198.91** | **0.9251** |
| **Gradient Boosting Regressor** | ₹ 82,764.75 | ₹ 1,33,058.00 | 0.9193 |
| **Decision Tree Regressor** | ₹ 88,084.78 | ₹ 1,57,698.99 | 0.8866 |
| **Extra Trees Regressor** | ₹ 78,298.81 | ₹ 1,58,724.80 | 0.8851 |
| **Ridge Regression** | ₹ 1,33,399.83 | ₹ 2,59,161.84 | 0.6938 |
| **Linear Regression** | ₹ 1,33,972.91 | ₹ 2,61,704.44 | 0.6877 |

### 🏆 Best Model Selected
- **Random Forest Regressor** (`n_estimators=100`, `random_state=42`) achieving an **R² score of 0.9251** (92.51% variance explained).

---

## 📁 Project Structure

```
car-prediction/
│
├── Car details v3.csv         # Target dataset used for model training & EDA
├── app.py                     # Main Streamlit web application entry point
├── train_model.py             # Data preprocessing & model training pipeline script
├── requirements.txt           # Python dependencies for deployment
├── README.md                  # Comprehensive documentation
│
└── models/
    ├── car_price_pipeline.pkl # Serialized Scikit-Learn pipeline
    └── model_metadata.pkl     # Pre-computed metrics, ranges & metadata
```

---

## 💻 How to Run Locally

1. **Clone the repository and enter directory**:
   ```bash
   cd "car prediction"
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train model (Optional - pre-built pipeline included)**:
   ```bash
   python train_model.py
   ```

4. **Launch Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

1. Push your repository to GitHub.
2. Sign into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select your repository, set the main entry file to `app.py`.
4. Click **Deploy**!
