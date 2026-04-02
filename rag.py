from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ingest import get_vectorstore
import time

SYSTEM_PROMPT = """You are a helpful AI assistant 
with access to a knowledge base.

Your behavior:
- Answer ONLY based on the retrieved context from the knowledge base.
- If no relevant info is found, say: "I could not find relevant information in the uploaded documents."
- Never make up or guess answers.
- Present numbers, dates, and amounts exactly as they appear.
- Keep answers clear, concise, and professional.
- Use lists or tables for invoice/financial data where helpful.
"""


def _build_prompt():
    system = SystemMessagePromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\nContext:\n{context}"
    )
    human = HumanMessagePromptTemplate.from_template("{question}")
    return ChatPromptTemplate.from_messages([system, human])


def build_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        streaming=True,
        temperature=0.1,
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Build the prompt
    prompt = _build_prompt()

    # Create the chain using LCEL (LangChain Expression Language)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain