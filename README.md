
# 🩺 Brest Cancer Detection Web App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview
Breast cancer is one of the most common cancers worldwide, and **early detection** is crucial for effective treatment.  
This project is a **machine learning powered web application** that predicts whether a tumor is **Benign** or **Malignant** based on user-provided medical features.  

The app provides a **simple and interactive interface** for healthcare professionals, students, and researchers to test predictions quickly.

---

## 🚀 Features
- 🔍 Predicts **Breast Cancer (Benign vs. Malignant)**  
- 🌐 Web-based UI built with **Flask**  
- 📊 Machine Learning model trained on **Scikit-learn**  
- 💾 Saves and loads trained models for efficient inference  
- 🖼️ Demo screenshots (see below)

---

## 🛠️ Tech Stack
- **Frontend:** HTML, CSS, Bootstrap  
- **Backend:** Flask (Python)  
- **Machine Learning:** Scikit-learn  
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) dataset (from `sklearn.datasets`)  

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amon-Mugo/Breast-Cancer-detection-web-app.git
   cd Breast-Cancer-detection-web-app
Create a virtual environment

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the app

bash
Copy code
python app.py
Visit the app at 👉 http://127.0.0.1:5000

📸 Demo (Screenshots)
⚠️ Add screenshots to make your project stand out!
Save them inside an assets/ folder, then link here.

Homepage

Prediction Result

📊 Model
Algorithm: Logistic Regression (Scikit-learn)

Accuracy: ~95% (on Breast Cancer Wisconsin dataset)

Trained and saved as model.pkl for real-time predictions.

🤝 Contributing
Contributions are welcome!
If you’d like to improve this project:

Fork the repo

Create a new branch (feature/your-feature)

Commit changes

Open a Pull Request

📜 License
This project is licensed under the MIT License.
See the LICENSE file for more details.

👤 Author
Amon Mugo
GitHub | LinkedIn
