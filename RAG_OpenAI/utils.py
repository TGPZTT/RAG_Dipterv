def classify_question_and_language(message, llm):
    prompt = (
        f"Classify the following question in two ways:\n"
        f"1. Identify the language of the message (e.g., 'English', 'Hungarian', 'Czech').\n"
        f"2. Determine if the question is 'document-related' or 'conversational'. "
        f"Only classify as 'conversational' if it is a generic, everyday question that has no need for specific document-based information, "
        f"like greetings or weather questions. Otherwise, classify it as 'document-related'.\n\n"
        f"Question: '{message}'\n\n"
        "Respond with only two words separated by a comma: the language name and the classification ('document-related' or 'conversational')."
    )
    response = llm.predict(prompt).strip()
    language, classification = [part.strip() for part in response.split(",")]
    is_document_related = classification == "document-related"
    return language, is_document_related


def format_sources(sources):
    grouped_sources = {}
    for doc in sources:
        title = doc.metadata.get('title', 'Unknown Document')
        page = doc.metadata.get('page', 0) + 1
        if title not in grouped_sources:
            grouped_sources[title] = set()
        grouped_sources[title].add(page)

    formatted_sources = []
    for title, pages in grouped_sources.items():
        top_pages = sorted(pages)[:2]
        pages_str = ', '.join(map(str, top_pages))
        formatted_sources.append(f"{title}, {pages_str}. oldal")

    return "Források:\n" + "\n".join(formatted_sources)


def process_history(history, max_pairs=3):
    """Levágja a history-t, csak az utolsó N kérdés-válasz pár marad."""
    return history[-max_pairs:] if history else []


def convert_history_for_chain(history):
    """Átalakítja a Gradio history-t (list of lists) a LangChain által várt list of tuples formátumra."""
    return [tuple(pair) for pair in history]


def chatbot_response(message, history, qa_chain, llm):
    # Nyelv és kérdéstípus osztályozás
    response_language, document_related = classify_question_and_language(message, llm)

    # History feldolgozása (limitálás és formátum konverzió)
    processed_history = process_history(history)
    processed_history_tuples = convert_history_for_chain(processed_history)

    # Prompt összeállítása
    response_prompt = (
        f"Please respond to the following question in {response_language}. "
        f"Question: '{message}'\n\n"
    )

    if document_related:
        # Válasz generálása dokumentum-alapú kérdés esetén
        result = qa_chain({"question": response_prompt, "chat_history": processed_history_tuples})
        answer = result["answer"]

        # Releváns dokumentumok keresése
        retriever = qa_chain.retriever  
        docs_with_scores = retriever.vectorstore.similarity_search_with_score(response_prompt, k=5)
        threshold = 0.87  
        sources = [doc for doc, score in docs_with_scores if score < threshold]

        # Debug kiírás
        print("\n Releváns dokumentumok és a pontok:")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            print(f"{i}. {doc.metadata.get('title', 'No Title')} (score: {score:.3f})")

        # Források formázása
        if sources:
            source_info = format_sources(sources)
            full_response = f"{answer}\n\n[Dokumentum alapú információk]\n{source_info}"
        else:
            full_response = (
                f"{answer}\n\n"
                "[A kérdés túl általános, vagy egyik dokumentum relevanciája sem számottevő. "
                "Próbáljon meg konkrétabban fogalmazni, vagy pontosítsa a kérdést!]"
            )
    else:
        # Conversational kérdés esetén válaszgenerálás
        answer = llm.predict(response_prompt)
        full_response = answer

    # CSAK az "answer" fog bekerülni a history-ba, nem a források!
    return full_response
