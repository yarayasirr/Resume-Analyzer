# 📄 Resume Analyzer

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-success)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **Streamlit web app** that analyzes resumes (PDF, DOCX, TXT) and predicts the **job category** using a machine learning model. The app also extracts key skills, shows prediction confidence, and allows users to download the results.

---

## 🚀 Features

- ✅ Upload resumes in **PDF, DOCX, or TXT** format  
- ✅ Extract and display **resume text**  
- ✅ Predict **job category** using a pre-trained ML model  
- ✅ Show **prediction confidence**  
- ✅ Extract and highlight **key skills**  
- ✅ Download results as a **CSV file**  
- ✅ Clean and **interactive UI** with Streamlit

---

## 🛠 How It Works

1. **File Upload**  
   Upload your resume in PDF, DOCX, or TXT format.

2. **Text Extraction & Cleaning**  
   - PDF → `PyPDF2`  
   - DOCX → `python-docx`  
   - TXT → plain text with UTF-8 / Latin-1 fallback  
   - Cleaning using regex: remove URLs, special characters, extra spaces, and non-ASCII symbols.

3. **Natural Language Processing (NLP)**  
   - Text is vectorized using **TF-IDF** (Term Frequency-Inverse Document Frequency).  
   - Optional: deep learning-based embeddings could be integrated for semantic understanding.

4. **Machine Learning Model**  
   - **Classifier:** `LinearSVC` wrapped in `OneVsRestClassifier` for multi-class classification.  
   - Model predicts job category based on vectorized resume text.  
   - Uses **label encoder** to map predicted numeric labels to category names.  

5. **Prediction & Confidence**  
   - If available, the model outputs **probability/confidence scores** for each category.  
   - Top category is displayed with confidence.

6. **Skill Extraction**  
   - Predefined key skills (e.g., Python, SQL, Machine Learning, Excel) are extracted and displayed.  
   - Can be extended with NLP-based Named Entity Recognition (NER).

7. **Downloadable Results**  
   - Results can be downloaded as CSV including predicted category, confidence, and detected skills.

---

## 📂 Supported File Types

- PDF  
- DOCX  
- TXT  

---

## ⚡ Installation

1. Ensure your pre-trained ML files are present:
- clf.pkl → `trained classifier`  
- tfidf.pkl → `TF-IDF vectorizer`
- encoder.pkl → `Label encoder`

2. Clone the repository:

```bash
git clone https://github.com/yarayasirr/resume-analyzer.git
cd resume-analyzer


