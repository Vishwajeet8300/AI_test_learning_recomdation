# 🧠 AI-Based Test Generator & Learning Recommendation System
A smart education tool designed to automatically generate customized tests and provide personalized learning recommendations based on user performance. This system uses NLP, RAG (Retrieval-Augmented Generation), and LLMs to understand content, assess knowledge, and guide students toward mastery.

## 🚀 Features
📄 Content Ingestion: Accepts PDFs, PPTs, Word documents, and text files as input.

# 🧠 AI-Powered Test Generation:

Generates MCQs using extractive and abstractive methods.

Supports difficulty-based question classification.

### 📊 Performance Analysis:

Evaluates test responses.

Detects weak concepts/topics.

### 🎯 Learning Recommendation System:

Suggests personalized topics to improve.

Recommends video tutorials, articles, and concepts.

### 💬 LLM-Powered Feedback:

Uses a RAG system to provide insights and explanations.

Optionally integrated with Gemini Pro, GPT-4, or LLaMA.

# 🛠 Tech Stack

Area	Tech
Backend	Python, FastAPI
Frontend	Streamlit / React (based on use-case)
LLMs	Gemini Pro / OpenAI GPT / LLaMA
Vector DB	Chroma / FAISS
Embeddings	Gemini Embeddings / Sentence Transformers
Data Parsing	PyMuPDF, python-docx, python-pptx
MCQ Generation	T5 / GPT-3.5 / Gemini Pro
Analysis	Custom NLP pipelines, RAG

# 📂 Project Structure
css
Copy
Edit
📦AI-Test-Generator
├── 📁data
│   └── input_files/
├── 📁backend
│   └── main.py (FastAPI endpoints)
├── 📁llm
│   ├── summarizer.py
│   ├── mcq_generator.py
│   └── feedback_analyzer.py
├── 📁rag
│   ├── vector_store.py
│   └── retriever.py
├── 📁frontend
│   └── app.py (Streamlit / React components)
├── 📁utils
│   └── parser.py
├── requirements.txt
└── README.md

# 🧪 Example Workflow
Upload Content: Users upload lecture slides, notes, or textbooks.

Generate Test: AI creates multiple-choice questions from the content.

Take Test: Student answers questions via a web interface.

Analyze Performance: The model identifies weak areas.

Get Recommendations: System suggests study material and resources.

Repeat: Adaptive testing improves learning continuously.

# 📌 Installation
bash
Copy
Edit
git clone https://github.com/your-username/AI-Test-Generator.git
cd AI-Test-Generator
pip install -r requirements.txt


# 🧠 Future Enhancements
Adaptive learning path generation using reinforcement learning.

Gamification of tests.

Integration with LMS (e.g., Moodle, Google Classroom).

Support for subjective questions and evaluation.

# 🤝 Contribution
Contributions, issues, and suggestions are welcome!

Fork this repo

Create a feature branch (git checkout -b feature-name)

Commit your changes (git commit -am 'Add new feature')

Push to the branch (git push origin feature-name)

Open a Pull Request

