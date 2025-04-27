import os
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import gradio as gr

from utils import VoyageEmbeddingFunction, classify_question_and_language, format_sources, chatbot_response

load_dotenv()

PERSIST_DIRECTORY = "chroma_db"
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

vectorstore = Chroma(
    embedding_function=VoyageEmbeddingFunction(model="voyage-3-large"),
    persist_directory=PERSIST_DIRECTORY
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

titles_path = os.path.join(PERSIST_DIRECTORY, "titles.txt")
titles = []
if os.path.exists(titles_path):
    with open(titles_path, "r", encoding="utf-8") as f:
        titles = [line.strip() for line in f.readlines()]

iface = gr.ChatInterface(
    fn=lambda msg, hist: chatbot_response(msg, hist, qa_chain, llm),
    title="Intelligens dokumentumalapú kérdezőrendszer - Diplomamunka - SzM",
    description=(
        "<h3>Kérdezzen a dokumentumokról (magyarul vagy angolul)</h3>"
        "<p>Hivatkozásokat tartalmaz azokra a válaszokra, amelyek dokumentum információkon alapulnak.</p>"
        "<details><summary><small>**Dokumentumok:**</small></summary>"
        f"<small>{', '.join(titles)}</small>"
        "</details>"
    ),
    cache_examples=True,
)

if __name__ == "__main__":
    iface.launch(share=True)
