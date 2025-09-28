from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    VideoUnavailable,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


from sklearn.metrics.pairwise import cosine_similarity


def main():
    # Indexing

    video_id = "wjZofJX0v4M"
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)

        transcript = " ".join([i.text for i in transcript_list.snippets])

    except (TranscriptsDisabled, VideoUnavailable):
        print("No caption available")

    # Chunking

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embeddings Generation

    embededdings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embededdings)

    # Retriever

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    retrieved_docs = retriever.invoke("What initials GPT stands for?")

    # Augmentation

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = PromptTemplate(
        template="""
                          You are a helpful assistant.
                          Answer ONLY from the provided transcript context.
                          If the context is insufficient, just say you don't know.
                          
                          {context}
                          Question: {question}
                          """,
        input_variables=["context", "question"],
    )
    question = "Did he discussed about Nuclear weapon , if yes give detailed explanation"

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({"context": context_text, "question": question})

    res = llm.invoke(final_prompt)

    print(res.content)

    # Convert every step in langchain's chain


if __name__ == "__main__":
    main()
