from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langserve import add_routes
from fastapi import FastAPI
from langchain_core.messages import BaseMessage
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor
from typing import List

import os
os.environ["OPENAI_API_KEY"] = "ENTER-OPENAI-KEY-HERE"
os.environ["PINECONE_API_KEY"] = "ENTER-PINECONE-API-KEY-HERE"

# Load the PDF document
loader = PyPDFLoader("ENTER-DOCUMENT-PATH-HERE")
documents = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create embeddings for the text chunks
index_name = "PINECONE-INDEX-NAME-HERE"
embeddings = OpenAIEmbeddings()
# If you would like to switch back to FAISS
# docsearch = FAISS.from_documents(texts, embeddings)
docsearch = PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)

# Set up the language model and retrieval chain
llm = OpenAI(temperature=0, max_tokens=1500)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

# Define the writing style identification prompt
style_prompt = """
Analyze the provided text and identify the key characteristics of its writing style. Provide a summarization of the writing style in one sentence. You should NOT provide what the text is about but simply the writing style.
"""

# Query the retrieval chain with the style prompt
style_result = qa.run(style_prompt)
print(style_result)

# Define the query answering prompt template
query_prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Answer the following query in the writing style of {style}:\n\nQuery: {{query}}".format(style=style_result)
)

# Set up the query answering chain
query_chain = LLMChain(llm=llm, prompt=query_prompt_template)

# Get user input for the query
user_query = input("Enter your query: ")

# Run the query chain with the user query
final_result = query_chain.run(user_query)

print(final_result)
