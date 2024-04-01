from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, File, UploadFile
from langchain.llms import LangServeLLM

app = FastAPI()

# Set up the LLM
llm = LangServeLLM(endpoint_url="http://localhost:8080")

# Set up the document loader and text splitter
loader = UnstructuredFileLoader()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Set up the embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# Set up the prompt template
prompt_template = """
Identify the writing tone of the given text.

Text: {text}

Tone:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Load and split the document
    document = loader.load(file.file.read())
    texts = text_splitter.split_documents(document)

    # Create a vector store from the texts
    vectorstore.add_texts(texts)

    # Set up the retrieval QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), prompt=prompt)

    # Identify the writing tone
    tone = qa.run(texts[0].page_content)

    return {"tone": tone}

@app.post("/generate")
async def generate_text(tone: str, prompt: str):
    # Set up the prompt template for text generation
    generate_prompt_template = """
    Write a {tone} response to the following prompt:

    Prompt: {prompt}

    Response:"""

    generate_prompt = PromptTemplate(template=generate_prompt_template, input_variables=["tone", "prompt"])

    # Generate text using the identified tone
    response = llm(generate_prompt.format(tone=tone, prompt=prompt))

    return {"response": response}