import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load the PDF document
loader = PyPDFLoader("Foreign Policy.pdf")
documents = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings for the text chunks
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

# Set up the language model and retrieval chain
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

# Define the writing style identification prompt
prompt = """
Identify the writing style of the given text. To do this you look for these eight things: Sentence length, structure, variation, and position.
Use of sensory details, figurative language, and other literary devices.
Use of sound devices: alliteration, onomatopoeia, rhythm, repetition.
Use of dialogue.
Word choice (diction)
Tone.
Use of local color/culture.
Use of irony.
"""
# Query the retrieval chain with the prompt
result = qa.run(prompt)

print(result)