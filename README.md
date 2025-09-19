# 🎬 Movie Rating Prediction System

A Machine Learning project to **predict IMDb movie ratings** based on movie metadata like Genre, Director, Actors, Duration, and Votes.  
This project was developed as part of the **CODSOFT Data Science Internship (Task 2)**.

---

## 📌 Project Overview
IMDb ratings are influenced by factors such as cast, crew, genre, and popularity.  
In this project, we:
- Preprocessed and cleaned a dataset of **15,000+ Indian movies**.
- Engineered features:
  - `Decade` (derived from Year)
  - `Is_Long_Movie` (flag for movies longer than 120 minutes)
  - Log-transformed `Votes`
  - Grouped rare Directors/Actors into **"Other"**
- Trained and compared models:
  - ✅ Random Forest
  - ✅ Gradient Boosting
  - ✅ XGBoost
- Saved the **best model (`movie_model.pkl`)** for deployment.

---

## ⚙️ Tech Stack
- **Python 3.12**
- **Libraries**:
  - pandas, numpy
  - scikit-learn
  - xgboost
  - streamlit
  - joblib

---

## 📊 Model Performance

| Model             | RMSE    | R² Score       |
|-------------------|---------|----------------|
| Random Forest     | 1.25    | 0.16           |
| Gradient Boosting | 1.23    | 0.19           |
| **XGBoost**       | **1.18**| **0.26** ✅   |

➡️ The **XGBoost model** gave the best results and was saved as `movie_model.pkl`.

---

## 🚀 How to Run

### 1. Clone the Repository
```
git clone https://github.com/Sauravt25-a11y/Movie_Prediction_System.git
cd Movie_Prediction_System
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
```
python train_model.py
```

### 4. Run the Streamlit App
```
streamlit run app.py
```


📂 Project Structure

Movie_Prediction_System/
│── IMDb_Movies_India.csv      # Dataset
│── train_model.py             # Training script
│── movie_model.pkl            # Saved trained model
│── app.py                     # Streamlit app
│── requirements.txt           # Dependencies
│── README.md                  # Documentation


📄 Requirments 

pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.1.1
streamlit==1.38.0
joblib==1.4.2
matplotlib==3.9.2
seaborn==0.13.2


---

## 🙌 Acknowledgments
This project was developed as part of the **CODSOFT Data Science Internship**.
Special thanks to the open-source community and **IMDb dataset providers** for making this project possible.

---
## 📢 Connect with Me
If you find this project useful or interesting, feel free to ⭐ star the repo and connect with me:

- 🔗 [GitHub](https://github.com/Sauravt25-a11y)  
- 💼 [LinkedIn](https://www.linkedin.com/in/saurav-thakur-810755320/)  

---
### 🚀 Final Note
Predicting IMDb ratings is inherently challenging because ratings depend on subjective factors like audience taste and cultural trends.  
Still, this project demonstrates how **machine learning + feature engineering + boosting models** can provide meaningful insights and predictive power.

**Thank you for exploring this project! 🎬📊**

