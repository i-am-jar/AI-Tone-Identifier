from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

import os
os.environ["OPENAI_API_KEY"] = "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"

# Load the PDF document
loader = PyPDFLoader("J. Acker Hierarchies, Jobs, Bodies -- A Theory of Gendered Organizations.pdf")
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
style_prompt = """
Identify the writing style of the given text. To do this you look for these eight things: Sentence length, structure, variation, and position.
Use of sensory details, figurative language, and other literary devices.
Use of sound devices: alliteration, onomatopoeia, rhythm, repetition.
Use of dialogue.
Word choice (diction)
Tone.
Use of irony.
"""

# Query the retrieval chain with the style prompt
style_result = qa.run(style_prompt)

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