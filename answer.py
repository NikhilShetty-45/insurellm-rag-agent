from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
OPENAI_API_CODE = os.getenv("OPENAI_API_KEY")
DB_NAME = str(Path(__file__).parent / "vector_db")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable and friendly assistant representing Insurellm. 
Your primary role is to assist users with questions about Insurellm's products, services, and policies.
Guidelines for your responses:
1) If a user asks a question that is directly related to Insurellm, use the provided context to give a helpful and accurate answer. 
    If the context contains the necessary information, share it. If the question is about Insurellm but the specific answer is not in the provided context,
    respond with: "I don't have the necessary details about that in my current resources. I recommend contacting Insurellm support for the most accurate and up-to-date information."
2) If a user asks about a topic that is completely unrelated to Insurellm or the insurance industry, 
    respond politely with: "I'm here to help with questions about Insurellm. I don't have the necessary details about that topic."
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model=MODEL)

openai = OpenAI(api_key=OPENAI_API_CODE)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    if isinstance(question, list): 
        if len(question) > 0 and isinstance(question[0], dict): 
            question = question[0].get("text", "") 
        else: 
            question = str(question)
            
    return retriever.invoke(question, k= RETRIEVAL_K)


def answer_question(question: str, history: list[dict] = []) ->  tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """  
    docs = fetch_context(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.extend([HumanMessage(content=question)])
    response = llm.invoke(messages)
    return response.content, docs
