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

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY is missing in .env")
    st.stop()

# Load MCQs from the JSON file
@st.cache_data
def load_questions():
    with open("question.json", "r") as f:
        return json.load(f)

# Load student data
@st.cache_data
def load_student_data():
    csv_path = "clustered_students_no_norm.csv"
    return pd.read_csv(csv_path)

# Setup the app
st.set_page_config(page_title="AI Test & Feedback Assistant", page_icon="🧠", layout="wide")

# App UI
st.title("🧠 AI Test & Feedback Assistant")
st.markdown("Take the test below and receive personalized feedback based on your performance.")

# Initialize state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'student_id'
if 'student_answers' not in st.session_state:
    st.session_state.student_answers = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""

# Load data
questions = load_questions()
student_df = load_student_data()

# Step 1: Student ID
if st.session_state.current_step == 'student_id':
    with st.container():
        st.subheader("Step 1: Identify Yourself")
        student_id = st.text_input("Enter your Student ID:")
        if st.button("Start Test") and student_id:
            st.session_state.student_id = student_id
            st.session_state.current_step = 'test'
            st.rerun()  # Updated from experimental_rerun()

# Step 2: Show Questions
elif st.session_state.current_step == 'test':
    st.subheader(f"Step 2: Complete the Test - Student ID: {st.session_state.student_id}")
    
    # Create two columns - one for Computer Networks, one for Machine Learning
    col1, col2 = st.columns(2)
    
    # Filter questions by topic
    network_questions = [q for q in questions if q['topic'] == 'Computer Networks']
    ml_questions = [q for q in questions if q['topic'] == 'Machine Learning']
    
    # Display Computer Networks questions
    with col1:
        st.markdown("### Computer Networks")
        for i, q in enumerate(network_questions):
            with st.expander(f"Q{i+1}: {q['question']}"):
                options = q['options']
                selected = st.radio("Choose one:", options, key=f"network_q_{i}")
                st.session_state.student_answers[q['question']] = selected
    
    # Display Machine Learning questions
    with col2:
        st.markdown("### Machine Learning")
        for i, q in enumerate(ml_questions):
            with st.expander(f"Q{i+1}: {q['question']}"):
                options = q['options']
                selected = st.radio("Choose one:", options, key=f"ml_q_{i}")
                st.session_state.student_answers[q['question']] = selected
    
    # Submit button
    if st.button("Submit Test"):
        # Calculate scores
        ml_score = 0
        network_score = 0
        incorrect_ml = []
        incorrect_network = []
        
        for q in questions:
            student_answer = st.session_state.student_answers.get(q["question"], "")
            if student_answer == q["answer"]:
                if q["topic"] == "Machine Learning":
                    ml_score += 1
                else:
                    network_score += 1
            else:
                if q["topic"] == "Machine Learning":
                    incorrect_ml.append(q["question"])
                else:
                    incorrect_network.append(q["question"])
        
        # Store results
        st.session_state.results = {
            "ml_score": ml_score,
            "network_score": network_score,
            "incorrect_ml": incorrect_ml,
            "incorrect_network": incorrect_network,
            "total_ml": len(ml_questions),
            "total_network": len(network_questions)
        }
        
        # Move to results page
        st.session_state.current_step = 'results'
        st.rerun()  # Updated from experimental_rerun()

# Step 3: Results and Feedback
elif st.session_state.current_step == 'results':
    res = st.session_state.results
    
    # Display scores
    st.subheader("Step 3: Test Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Machine Learning Score", f"{res['ml_score']}/{res['total_ml']}")
        st.progress(res['ml_score']/res['total_ml'])
        
    with col2:
        st.metric("Computer Networks Score", f"{res['network_score']}/{res['total_network']}")
        st.progress(res['network_score']/res['total_network'])
    
    # Get weak topics from CSV
    student_id = st.session_state.student_id
    weak_subjects_text = "None"
    cluster_info = "Not available"
    
    if student_id in student_df['Student_ID'].values:
        student_row = student_df[student_df['Student_ID'] == student_id].iloc[0]
        weak_subjects_text = student_row['Weak Subject'] if not pd.isna(student_row['Weak Subject']) else "None"
        cluster_info = f"Cluster {student_row['Cluster']}"
    else:
        st.warning("⚠️ Student ID not found in clustered dataset. Only test performance will be considered.")
    
    # Format student response summary for the LLM
    student_summary = f"""
    # Student Performance Analysis
    
    ## Student Information
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
    
    with st.container():
        st.subheader("💡 AI Analysis in Progress...")
        st.markdown("Getting personalized feedback based on your performance...")
        
        # Prepare LangChain pipeline
        docs = [Document(page_content=student_summary)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)

        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(split_docs, embedding=embedding)
        retriever = vectorstore.as_retriever()

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable and supportive AI tutor specializing in Computer Networks and Machine Learning. 
            
            Based on the student's historical weak subjects and current test performance, provide:
            
            1. A detailed analysis of their weak areas
            2. Specific topics they should focus on improving
            3. Personalized study strategies
            4. Recommended learning resources for each weak topic (books, online courses, websites, etc.)
            5. Practice exercises they can use to improve
            
            Format your response with clear headings and bullet points for easy reading.
            """),
            ("human", "{context}")
        ])

        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        # Run RAG Chain
        with st.spinner("Generating personalized feedback..."):
            result = rag_chain.invoke({"input": student_summary})
            st.session_state.feedback = result['answer']

        # Display AI feedback
        st.subheader("💡 Your Personalized Learning Plan")
        st.markdown(st.session_state.feedback)
        
        # Add option to retake test
        if st.button("Retake Test"):
            st.session_state.current_step = 'student_id'
            st.session_state.student_answers = {}
            st.session_state.results = {}
            st.session_state.feedback = ""
            st.rerun()  # Updated from experimental_rerun()