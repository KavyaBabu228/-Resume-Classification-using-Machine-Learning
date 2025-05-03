# 📄 Resume Classification using Machine Learning

This project aims to develop a robust machine learning solution to automatically classify resumes into predefined job categories. The system is designed to streamline the recruitment process by reducing manual screening time, increasing accuracy in candidate matching, and improving the overall efficiency of hiring workflows.

---

## 📌 Objective

The objective is to build an end-to-end automated pipeline that:

- Cleans and processes raw resume data  
- Extracts meaningful features using NLP techniques  
- Trains and evaluates multiple machine learning models  
- Optimizes performance through hyperparameter tuning  
- Deploys the best model as a web application for real-time classification  

---

## 🔍 Exploratory Data Analysis (EDA)

A comprehensive EDA was conducted to understand the structure and content of the resume dataset:

- Analyzed class imbalances and text length variations  
- Explored frequency of keywords, job titles, and skills  
- Applied text normalization, stopword removal, and lemmatization  
- Created visualizations to support feature engineering decisions  

---

## 🧠 Model Development

Multiple classification algorithms were tested and compared:

- Models: Logistic Regression, Naive Bayes, SVM, Random Forest  
- Feature extraction: TF-IDF Vectorization  
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
- Cross-validation was used for model selection and performance comparison  

---

## 🔧 Hyperparameter Tuning

Model performance was optimized using:

- `GridSearchCV` for exhaustive search  
- `RandomizedSearchCV` for probabilistic search  
- Manual tuning of thresholds and TF-IDF parameters  

This significantly improved model performance and reduced overfitting.

---

## 🚀 Deployment

The best-performing model was deployed as a web app using **[Streamlit or Flask]**.

### Features:

- Clean UI to upload resumes (text or PDF)  
- Backend processing: text cleaning, vectorization, classification  
- Real-time prediction of job category  
- Ready for deployment on Heroku, Render, or Streamlit Cloud  

---

## ✅ Project Outcomes

- Built a complete ML pipeline from EDA to deployment  
- Achieved high classification accuracy  
- Deployed a fully functional web app for HR and recruitment use cases  

---

## 🧰 Technologies & Tools

- **Programming**: Python  
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`  
- **Deployment**: Streamlit / Flask  
- **Environment**: Jupyter Notebook, Git  

---

## 📈 Future Enhancements

- Integrate transformer-based models (BERT, RoBERTa)  
- Enable multi-label classification for multi-domain resumes  
- Add PDF parsing and structured extraction support  
- Containerize using Docker and set up CI/CD pipelines for production readiness  

---

## 🙋‍♀️ Author

**Kavya Babu**  
[LinkedIn](https://www.linkedin.com/in/kavya-babu-15a36a2b5/)  
