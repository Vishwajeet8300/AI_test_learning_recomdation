from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os

# Set Gemini Pro API key
os.environ["GOOGLE_API_KEY"] = "Your gemini api key"

def analyze_score_with_gemini(total_score, subject_scores, subject_total):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    analysis_input = f"""
    A student took an MCQ test. Here is the performance data:

    Total Score: {total_score}
    Subject-wise scores:
    """

    for subject in subject_total:
        analysis_input += f"\n- {subject}: {subject_scores.get(subject, 0)}/{subject_total[subject]}"

    analysis_input += """

    Based on this data:
    1. Identify if the student is weak in any subject.
    2. Suggest 3 key steps to improve weak subjects.
    3. Recommend learning strategies if needed.
    """

    response = model.invoke([HumanMessage(content=analysis_input)])
    return response.content
