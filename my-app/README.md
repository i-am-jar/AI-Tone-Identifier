# PDF Query Answering with Writing Style Identification

This project demonstrates how to use the Langchain library to load a PDF document, split it into chunks, create embeddings for the chunks, and set up a retrieval chain to answer queries based on the content of the PDF. Additionally, it identifies the writing style of the text and answers queries in the same style.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- langchain
- faiss-cpu
- openai

You can install them using pip:
pip install langchain faiss-cpu openai

## Setup

1. Replace `"ENTER API KEY HERE"` with your actual OpenAI API key.
2. Replace `"LOAD-PDF-HERE"` with the path to the PDF file you want to load.

## Usage

1. Run the script.
2. Enter your query when prompted.
3. The script will identify the writing style of the text in the PDF and answer your query in the same style.

## How it works

1. The script loads the PDF document using `PyPDFLoader` from the Langchain library.
2. It splits the text into chunks using `CharacterTextSplitter` with a specified chunk size and overlap.
3. Embeddings are created for the text chunks using `OpenAIEmbeddings`.
4. A retrieval chain is set up using `RetrievalQA` with the specified language model (OpenAI) and the document search index.
5. A writing style identification prompt is defined to analyze the text based on various linguistic features.
6. The retrieval chain is queried with the style prompt to identify the writing style of the text.
7. A query answering prompt template is defined, incorporating the identified writing style.
8. The query answering chain is set up using the prompt template and the language model.
9. The user is prompted to enter a query.
10. The query chain is run with the user query, and the final result is printed, answering the query in the identified writing style.

## Customization

- You can adjust the `chunk_size` and `chunk_overlap` parameters in the `CharacterTextSplitter` to control the size and overlap of the text chunks.
- The `temperature` parameter in `OpenAI` can be modified to control the randomness of the generated responses.
- The writing style identification prompt can be customized to focus on different linguistic features or aspects of writing style.

## License

This project is licensed under the [MIT License](LICENSE).
