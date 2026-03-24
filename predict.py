import os
import pandas as pd
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
    print("❌ GOOGLE_API_KEY is missing in .env")
    exit(1)

# 1. Load the CSV student dataset
csv_path = "clustered_students_no_norm.csv"
df = pd.read_csv(csv_path)

# 2. Convert each row to a LangChain Document
documents = []
for index, row in df.iterrows():
    content = f"Student_ID: {row['Student_ID']}, Machine Learning: {row['Machine Learning']}, Computer Networks: {row['Computer Networks']}, Weak Subject: {row['Weak Subject']}"
    documents.append(Document(page_content=content))

# 3. Split the document chunks (optional here due to short texts)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 4. Embed using Gemini
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 5. Create Chroma vector store
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="./vector_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 6. Gemini LLM Setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=1000)

# 7. Prompt Template
system_prompt = (
    "You are a helpful assistant for analyzing student performance based on their Machine Learning and Computer Networks marks. "
    "Based on the student's data, analyze and suggest which subject(s) are weak. If both subjects are weak, mention both. "
    "\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# 8. Create RAG Chain
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# 9. Interactive Querying
def ask_student_query():
    print("Ask about a student's performance (e.g., 'Compare S010 and S030' or 'Who is weak in Machine Learning?')")
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit']:
            break
        result = rag_chain.invoke({"input": query})
        print(f"\n🧠 AI:\n{result['answer']}\n")

# Run it
if __name__ == "__main__":
    ask_student_query()
