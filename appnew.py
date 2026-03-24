
# IMPORTING ALL THE ESSENTIAL LIBRARIES.

import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO

# PDF generator function
def generate_pdf_report(student_name, student_id, feedback_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]
    bold_style = ParagraphStyle(name="Bold", parent=normal_style, fontName="Helvetica-Bold")

    story.append(Paragraph("Student Performance Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Student Name:</b> {student_name}", bold_style))
    story.append(Paragraph(f"<b>Student ID:</b> {student_id}", bold_style))
    story.append(Spacer(1, 12))

    # code to handle the markdown files in the pdf's
    feedback_text = feedback_text.replace("***", "<b>").replace("***", "</b>")  # Converts ***text*** to bold in PDF

    for line in feedback_text.split("\n"):
        if line.strip() == "":
            story.append(Spacer(1, 12))
        elif line.strip().startswith("#"):
            level = line.count("#")
            text = line.replace("#", "").strip()
            if level == 1:
                story.append(Paragraph(text, styles["Heading1"]))
            elif level == 2:
                story.append(Paragraph(text, styles["Heading2"]))
            else:
                story.append(Paragraph(text, styles["Heading3"]))
        else:
            story.append(Paragraph(line.strip(), normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Loading Environment Variables such as  API's && Py varients
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY is missing in .env")
    st.stop()

@st.cache_data
def load_questions():
    with open("question.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_student_data():
    return pd.read_csv("clustered_students_no_norm.csv")

st.set_page_config(page_title="AI Test & Feedback Assistant", page_icon="🧠", layout="wide")
st.title("🧠 AI Test & Feedback Assistant")
st.markdown("Take the test below and receive personalized feedback based on your performance.")

if 'current_step' not in st.session_state:
    st.session_state.current_step = 'student_id'
if 'student_answers' not in st.session_state:
    st.session_state.student_answers = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""

questions = load_questions()
student_df = load_student_data()

# Step 1: Student ID and Name
if st.session_state.current_step == 'student_id':
    with st.container():
        st.subheader("Step 1: Identify Yourself")
        student_name = st.text_input("Enter your Name:")
        student_id = st.text_input("Enter your Student ID:")
        if st.button("Start Test") and student_id and student_name:
            st.session_state.student_id = student_id
            st.session_state.student_name = student_name
            st.session_state.current_step = 'test'
            st.rerun()

# Step 2: TEST QUESTION     (CN: Computer Networks, ML: Machine Learning , Q[CN] = 21, Q[ML] = 20)
elif st.session_state.current_step == 'test':
    st.subheader(f"Step 2: Complete the Test - {st.session_state.student_name} ({st.session_state.student_id})")
    col1, col2 = st.columns(2)
    network_qs = [q for q in questions if q['topic'] == 'Computer Networks']
    ml_qs = [q for q in questions if q['topic'] == 'Machine Learning']

    with col1:
        st.markdown("### Computer Networks")
        for i, q in enumerate(network_qs):
            with st.expander(f"Q{i+1}: {q['question']}"):
                selected = st.radio("Choose one:", q['options'], key=f"network_q_{i}")
                st.session_state.student_answers[q['question']] = selected

    with col2:
        st.markdown("### Machine Learning")
        for i, q in enumerate(ml_qs):
            with st.expander(f"Q{i+1}: {q['question']}"):
                selected = st.radio("Choose one:", q['options'], key=f"ml_q_{i}")
                st.session_state.student_answers[q['question']] = selected

    if st.button("Submit Test"):
        ml_score, network_score = 0, 0
        incorrect_ml, incorrect_network = [], []
        for q in questions:
            ans = st.session_state.student_answers.get(q["question"], "")
            if ans == q["answer"]:
                if q["topic"] == "Machine Learning": ml_score += 1
                else: network_score += 1
            else:
                if q["topic"] == "Machine Learning": incorrect_ml.append(q["question"])
                else: incorrect_network.append(q["question"])

        st.session_state.results = {
            "ml_score": ml_score,
            "network_score": network_score,
            "incorrect_ml": incorrect_ml,
            "incorrect_network": incorrect_network,
            "total_ml": len(ml_qs),
            "total_network": len(network_qs)
        }
        st.session_state.current_step = 'results'
        st.rerun()

# Step 3: Results and Feedback
elif st.session_state.current_step == 'results':
    res = st.session_state.results
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Machine Learning Score", f"{res['ml_score']}/{res['total_ml']}")
        st.progress(res['ml_score'] / res['total_ml'])

    with col2:
        st.metric("Computer Networks Score", f"{res['network_score']}/{res['total_network']}")
        st.progress(res['network_score'] / res['total_network'])

    student_id = st.session_state.student_id
    student_name = st.session_state.student_name
    weak_subjects_text = "None"
    cluster_info = "Not available"
    if student_id in student_df['Student_ID'].values:
        row = student_df[student_df['Student_ID'] == student_id].iloc[0]
        weak_subjects_text = row['Weak Subject'] if not pd.isna(row['Weak Subject']) else "None"
        cluster_info = f"Cluster {row['Cluster']}"
    # Removed warning for Student ID not found in dataset

    student_summary = f"""
    # Student Performance Analysis

    ## Student Information
    - Student Name: {student_name}
    - Student ID: {student_id}
    - Historical weak subjects: {weak_subjects_text}
    - Student cluster: {cluster_info}

    ## Current Test Performance
    - Machine Learning score: {res['ml_score']}/{res['total_ml']} ({(res['ml_score']/res['total_ml'])*100:.1f}%)
    - Computer Networks score: {res['network_score']}/{res['total_network']} ({(res['network_score']/res['total_network'])*100:.1f}%)

    ## Incorrect Machine Learning Questions:
    {', '.join(res['incorrect_ml']) if res['incorrect_ml'] else 'None'}

    ## Incorrect Computer Networks Questions:
    {', '.join(res['incorrect_network']) if res['incorrect_network'] else 'None'}
    """

    with st.spinner("Generating personalized feedback..."):
        docs = [Document(page_content=student_summary)]
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
        retriever = Chroma.from_documents(chunks, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")).as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        prompt = ChatPromptTemplate.from_messages([("system", """You are a knowledgeable and supportive AI tutor...
            Provide feedback with headings, topics to focus, resources and exercises.
        """), ("human", "{context}")])
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        result = chain.invoke({"input": student_summary})
        st.session_state.feedback = result['answer']

    st.subheader("💡 Your Personalized Learning Plan")
    st.markdown(st.session_state.feedback)

    # PDF generation and download
    pdf_buffer = generate_pdf_report(student_name, student_id, st.session_state.feedback)
    st.download_button(
        label="📄 Download Feedback Report (PDF)",
        data=pdf_buffer,
        file_name=f"{student_name.replace(' ', '_')}_Performance_Report.pdf",
        mime="application/pdf"
    )

    if st.button("Retake Test"):
        st.session_state.current_step = 'student_id'
        st.session_state.student_answers = {}
        st.session_state.results = {}
        st.session_state.feedback = ""
        st.rerun()
