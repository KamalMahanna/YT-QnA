# YouTube QnA

This project is a Streamlit application that allows you to ask questions about YouTube videos. It downloads the transcript of a YouTube video, stores it in a vector database, and then uses a large language model to answer questions about the video.

## How it Works

The application works as follows:

1.  The user enters a YouTube video URL, a Groq API key, and a Gemini API key.
2.  The application extracts the video ID from the URL.
3.  The application downloads the transcript of the video using either the YouTube API or Whisper (via the Groq API).
4.  The application chunks the transcript into smaller pieces and stores them in a vector database using ChromaDB.
5.  The user asks a question about the video.
6.  The application retrieves the relevant transcript chunks from the vector database.
7.  The application uses a large language model (Gemini) to answer the question based on the retrieved transcript chunks.
8.  The application generates audio from the summary using Gemini LLM.
9.  The application displays the answer to the user.

## Directory Tree

```
.
├── DataBases
│   ├── __init__.py
│   └── VectorStore.py
├── Dockerfile
├── LICENSE
├── main.py
├── notebooks
│   └── test.ipynb
├── requirements.txt
├── utils
│   ├── AudioDownloader.py
│   ├── __init__.py
│   ├── HelperFunctions.py
│   ├── LLM.py
│   ├── Summarizer.py
│   └── Transcript.py
└── .gitignore
```

## Tools and Technologies Used

*   [Streamlit](https://streamlit.io/): A Python library for creating interactive web applications.
*   [ChromaDB](https://www.trychroma.com/): A vector database for storing and retrieving embeddings.
*   [Gemini API](https://ai.google.dev/): A large language model from Google AI.
*   [Groq API](https://console.groq.com/): An API for accessing Whisper for audio transcription.
*   [pytubefix](https://pytube.io/): A Python library for downloading YouTube videos.
*   [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api): A Python library for retrieving YouTube transcripts.
*   [python-dotenv](https://github.com/theskumar/python-dotenv): A Python library for loading environment variables from a .env file.
*   [nltk](https://www.nltk.org/): A Python library for natural language processing.
*   [pydub](https://github.com/jiaaro/pydub): A Python library for manipulating audio files.

## How to Run Locally

1.  Clone the repository:

    ```bash
    git clone https://github.com/KamalMahanna/YT-QnA.git
    cd YT-QnA
    ```

2.  Install Docker

    Click [here](https://docs.docker.com/get-started/get-docker/) for the Docker installation page
2.  Build the Docker image:

    ```bash
    docker build -t yt-qna .
    ```

3.  Run the Docker container:

    ```bash
    docker run -p 8501:8501 -v database_volume:/DataBases/my_chroma_db yt-qna:latest
    ```

4.  Open the application in your browser:

    [http://0.0.0.0:8501/](http://0.0.0.0:8501/)


## API Keys

*   **Groq API Key:** [https://console.groq.com/keys](https://console.groq.com/keys)
*   **Gemini API Key:** [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## Hosted Version

[https://yt-qna.streamlit.app/](https://yt-qna.streamlit.app/)
