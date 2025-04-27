import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

PERSIST_DIRECTORY = "chroma_db"
PDF_DIRECTORY = "pdfs"  

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def clear_vectorstore(directory):
    """Törli a vektortároló mappát, ha létezik."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"'{directory}' mappa törölve.")

def get_document_title(pdf_path, llm):
    """Kinyeri a címeket az első két oldal tartalmából, egyenlően kezelve őket."""
    pages = PyPDFLoader(pdf_path).load_and_split()

    combined_content = ""
    if len(pages) >= 2:
        combined_content = pages[0].page_content + "\n\n" + pages[1].page_content
    elif pages:
        combined_content = pages[0].page_content
    else:
        return os.path.basename(pdf_path).replace(".pdf", "")

    prompt = (
        "Extract the document title from the following content (Hungarian or English). "
        "The title is usually on the first or second page. If no clear title, return 'Untitled Document'. "
        "Format the title for human readability: remove extra spaces, fix capitalization, avoid all caps, and ensure it looks like a natural, well-written title.\n\n"
    f"{combined_content[:2000]}"
    )
    title = llm.invoke(prompt).content.strip()
    return title if title and title != "Untitled Document" else os.path.basename(pdf_path).replace(".pdf", "")


def process_and_save_documents(pdf_directory, llm):
    """PDF-ek betöltése, feldolgozása, Chroma vektortároló létrehozása, címek mentése."""
    splits, titles = [], []

    pdf_paths = [
        os.path.join(pdf_directory, filename)
        for filename in os.listdir(pdf_directory)
        if filename.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        print("Nincsenek PDF fájlok a megadott könyvtárban!")
        return

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        title = get_document_title(path, llm)
        titles.append(title)

        for doc in RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages):
            doc.metadata['title'] = title
            splits.append(doc)

    vectorstore = Chroma.from_documents(
        splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory=PERSIST_DIRECTORY
    )
    vectorstore.persist()

    with open(os.path.join(PERSIST_DIRECTORY, "titles.txt"), "w", encoding="utf-8") as f:
        for title in titles:
            f.write(title + "\n")

    print(f"{len(pdf_paths)} dokumentum feldolgozva és elmentve a vektortárolóba.")
    print("Dokumentumcímek:", ", ".join(titles))

if __name__ == "__main__":
    print("Chroma vektortároló törlése és újraépítése...")
    clear_vectorstore(PERSIST_DIRECTORY)
    process_and_save_documents(PDF_DIRECTORY, llm)
    print("Feldolgozás befejezve.")
