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

1. Upload a resume (PDF, DOCX, TXT)  
2. Extract text from the file  
3. Clean the text (remove URLs, special characters, etc.)  
4. Vectorize the text using TF-IDF  
5. Predict the job category with the trained model  
6. Display results, confidence, and key skills  
7. Download a CSV report of the analysis  

---

## 📂 Supported File Types

- PDF  
- DOCX  
- TXT  

---

## ⚡ Installation

1. Clone the repository:

```bash
git clone https://github.com/yarayasirr/resume-analyzer.git
cd resume-analyzer
