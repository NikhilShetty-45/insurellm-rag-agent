from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
OPENAI_API_CODE = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT = """
You are a help Ai assistant with a personality of AI tutor.
You answer all the question Knowledgeably i.e clear and logically. 
"""

openai = OpenAI(api_key=OPENAI_API_CODE)


def answer_question(question: str) -> str:
    """
    Answer the given question using AI return the answer
    """  
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question},
        ]
    )
    return response.choices[0].message.content
