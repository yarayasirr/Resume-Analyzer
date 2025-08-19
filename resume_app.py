import streamlit as st
import pickle
import docx
import PyPDF2
import re
import pandas as pd

# ------------------ Load ML Model ------------------
svc_model = pickle.load(open(r'C:/Users/yaray/clf.pkl', 'rb'))
tfidf = pickle.load(open(r'C:/Users/yaray/tfidf.pkl', 'rb'))
le = pickle.load(open(r'C:/Users/yaray/encoder.pkl','rb'))

# ------------------ Predefined Sections & Skills ------------------
REQUIRED_SECTIONS = ["Education", "Experience", "Skills", "Projects", "Certifications"]
DEFAULT_SKILLS = ['Python', 'Java', 'SQL', 'Excel', 'Machine Learning', 'Communication']

# ------------------ Helper Functions ------------------
def cleanResume(txt):
    txt = re.sub('http\S+\s', ' ', txt)
    txt = re.sub('RT|cc', ' ', txt)
    txt = re.sub('#\S+\s', ' ', txt)
    txt = re.sub('@\S+', '  ', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    confidence = max(svc_model.predict_proba(vectorized_text)[0]) if hasattr(svc_model, "predict_proba") else None
    return predicted_category_name[0], confidence

def extract_skills(text, skills_list=DEFAULT_SKILLS):
    return [skill for skill in skills_list if skill.lower() in text.lower()]

def detect_sections(text):
    present = [s for s in REQUIRED_SECTIONS if s.lower() in text.lower()]
    missing = [s for s in REQUIRED_SECTIONS if s.lower() not in text.lower()]
    return present, missing

def calculate_score(present_sections, found_skills, total_skills):
    section_score = len(present_sections) / len(REQUIRED_SECTIONS)
    skill_score = len(found_skills) / total_skills if total_skills else 1
    return int((section_score + skill_score)/2 * 100)

def badge(text, color):
    return f"<span style='background-color:{color};color:white;padding:5px 12px;border-radius:12px;margin:3px'>{text}</span>"

# ------------------ Streamlit App ------------------
def main():
    st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

    # Sidebar
    st.sidebar.header("About this App")
    st.sidebar.info("""
        Upload your resume (PDF, DOCX, TXT) to get:
        - Predicted job category with confidence
        - Resume strength score
        - Detected/missing sections
        - Highlighted skills
        - Downloadable analysis report
    """)

    st.markdown("<h1 style='color: darkblue;'>📄 Resume Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("Upload your resume to get detailed insights and recommendations.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Resume text extracted successfully!")

            # Show extracted text
            if st.checkbox("Show extracted text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Prediction
            category, confidence = pred(resume_text)
            st.subheader("Predicted Job Category")
            st.markdown(f"**Category:** {category}")
            if confidence is not None:
                st.markdown(f"**Confidence:** {confidence*100:.2f}%")

            # Sections
            present_sections, missing_sections = detect_sections(resume_text)
            st.subheader("📂 Section Analysis")
            for sec in REQUIRED_SECTIONS:
                color = "#0d47a1" if sec in present_sections else "#90caf9"
                status = "Present ✅" if sec in present_sections else "Missing ❌"
                st.markdown(badge(f"{sec}: {status}", color), unsafe_allow_html=True)

            # Skills
            found_skills = extract_skills(resume_text)
            st.subheader("🛠 Skills Detected")
            if found_skills:
                for skill in found_skills:
                    st.markdown(badge(skill, "#0d47a1"), unsafe_allow_html=True)
            else:
                st.info("No predefined skills detected.")

            # Resume Score
            score = calculate_score(present_sections, found_skills, len(DEFAULT_SKILLS))
            st.subheader("💪 Resume Strength Score")
            st.markdown(f"<h2 style='color:#0d47a1'>{score}/100</h2>", unsafe_allow_html=True)

            # Suggestions
            if missing_sections or not found_skills:
                with st.expander("💡 Suggestions"):
                    for sec in missing_sections:
                        st.info(f"Add a **{sec}** section for a stronger resume.")
                    for skill in [s for s in DEFAULT_SKILLS if s not in found_skills]:
                        st.warning(f"Include **{skill}** skill to match common requirements.")

            # Downloadable report
            report_df = pd.DataFrame({
                "Predicted Category": [category],
                "Confidence": [confidence*100 if confidence else None],
                "Resume Score": [score],
                "Sections Present": [", ".join(present_sections)],
                "Sections Missing": [", ".join(missing_sections)],
                "Skills Detected": [", ".join(found_skills)]
            })
            st.download_button(
                "📥 Download Report",
                report_df.to_csv(index=False),
                file_name="resume_analysis.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
