import streamlit as st
import pickle
import docx
import PyPDF2
import re
import pandas as pd

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open(r'C:/Users/yaray/clf.pkl', 'rb'))
tfidf = pickle.load(open(r'C:/Users/yaray/tfidf.pkl', 'rb'))
le = pickle.load(open(r'C:/Users/yaray/encoder.pkl','rb'))

# ----------------- Functions -----------------

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    
    # Get prediction probability/confidence
    if hasattr(svc_model, "predict_proba"):
        pred_probs = svc_model.predict_proba(vectorized_text)
        confidence = max(pred_probs[0])
    else:
        confidence = None

    return predicted_category_name[0], confidence

def extract_skills(text, skills_list=None):
    """Optional simple keyword extraction for demo purposes"""
    if skills_list is None:
        skills_list = ['Python', 'Java', 'SQL', 'Excel', 'Machine Learning', 'Communication']
    found_skills = [skill for skill in skills_list if skill.lower() in text.lower()]
    return found_skills

# ----------------- Streamlit App -----------------

def main():
    st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")
    
    # Sidebar
    st.sidebar.header("About this App")
    st.sidebar.info("""
        This app analyzes resumes and predicts the **job category**.
        Supported file types: PDF, DOCX, TXT.
        Features:
        - Display extracted text
        - Show predicted category with confidence
        - Highlight skills found in the resume
    """)

    st.markdown("<h1 style='color: darkblue;'>📄 Resume Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("Upload a resume and get insights with predictions and key skills detected.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Resume text extracted successfully!")

            # Option to show extracted text
            if st.checkbox("Show extracted text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Extract skills
            skills = extract_skills(resume_text)
            if skills:
                st.markdown(f"**Detected Skills:** {', '.join(skills)}")
            else:
                st.info("No predefined skills detected.")

            # Prediction
            category, confidence = pred(resume_text)
            st.subheader("Predicted Job Category")
            st.markdown(f"**Category:** {category}")
            if confidence is not None:
                st.markdown(f"**Confidence:** {confidence*100:.2f}%")

            # Optional: allow download of results
            results_df = pd.DataFrame({
                "Predicted Category": [category],
                "Confidence": [confidence*100 if confidence else None],
                "Skills Detected": [', '.join(skills) if skills else None]
            })
            st.download_button(
                "📥 Download Results",
                results_df.to_csv(index=False),
                file_name="resume_analysis.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
