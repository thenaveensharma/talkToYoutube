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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)

load_dotenv()


def text_join(transcript_list):
    return " ".join([i.text for i in transcript_list.snippets])


def format_doc(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def main():
    # Indexing
    parser = StrOutputParser()
    video_id = "wjZofJX0v4M"

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        transcript = text_join(transcript_list)

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

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_doc),
            "question": RunnablePassthrough(),
        }
    )

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

    main_chain = parallel_chain | prompt | llm | parser
    question = (
        "Did he discussed about LLM , give 5 points but line should not exceed 15 words"
    )

    # res = main_chain.invoke(question)
    print(main_chain.get_graph().draw_ascii())

    


if __name__ == "__main__":
    main()
