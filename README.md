# TalkToYoutube

A Python application that enables you to have conversations with YouTube videos by extracting transcripts and using AI to answer questions about the video content.

## Features

- **Transcript Extraction**: Automatically fetches YouTube video transcripts using the YouTube Transcript API
- **Intelligent Chunking**: Splits long transcripts into manageable chunks for better processing
- **Vector Search**: Uses FAISS vector store with OpenAI embeddings for semantic similarity search
- **AI-Powered Q&A**: Leverages OpenAI's GPT models to answer questions based on video transcript content
- **RAG Implementation**: Implements Retrieval-Augmented Generation for accurate, context-aware responses

## How It Works

1. **Extract**: Fetches the transcript from a specified YouTube video
2. **Chunk**: Splits the transcript into overlapping chunks for better context preservation
3. **Embed**: Converts text chunks into vector embeddings using OpenAI's embedding model
4. **Store**: Saves embeddings in a FAISS vector database for fast similarity search
5. **Retrieve**: Finds the most relevant chunks based on your question
6. **Generate**: Uses GPT to generate answers based only on the retrieved transcript context

## Requirements

- Python 3.10+
- OpenAI API key (set in `.env` file)

## Dependencies

- `faiss-cpu` - Vector similarity search
- `langchain-community` - LangChain community integrations
- `langchain-openai` - OpenAI integration for LangChain
- `numpy` - Numerical computing
- `python-dotenv` - Environment variable management
- `scikit-learn` - Machine learning utilities
- `tiktoken` - OpenAI tokenization
- `youtube-transcript-api` - YouTube transcript extraction

## Setup

1. Clone the repository
2. Install dependencies: `pip install -e .`
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application: `python main.py`

## Current Implementation

The current version demonstrates the complete RAG pipeline with:

- Hardcoded YouTube video ID (`wjZofJX0v4M`)
- Sample questions about the video content
- Text chunking with 1000 character chunks and 200 character overlap
- Similarity search returning top 4 most relevant chunks
- GPT-4o-mini for answer generation

## Future Enhancements

- Interactive CLI for custom video URLs and questions
- Support for multiple video formats
- Web interface for easier usage
- Conversation history and context management
