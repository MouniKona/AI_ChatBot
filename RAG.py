import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_milvus import Milvus
# from langchain_community.document_loaders import WebBaseLoader,RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, utility

# MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_API_KEY = "jO2CzNIJphdGbwgqbL5QVvXSlkNQzlPk"

MILVUS_URI = "milvus/milvus_vector.db"
# MODEL_NAME = "open-mistral-7b"
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
file_path = "python-basics-sample-chapters.pdf"


# Step 1: Load the PDF file and extract text
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# Step 2: Split the text into smaller chunks
def split_documents(documents):

    # Create a text splitter to split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Split the text into chunks of 1000 characters
        chunk_overlap=300,  # Overlap the chunks by 300 characters
        is_separator_regex=False,  # Don't split on regex
    )
    # Split the documents into chunks
    docs = text_splitter.split_documents(documents)
    return docs


# Step 3: Generate embeddings for the text chunks
def get_embedding_function():

    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return embedding_function


# Creating vector storage
def create_vector_store(docs, embeddings, uri):

    # Create the directory if it does not exist
    head = os.path.split(uri)
    os.makedirs(head[0], exist_ok=True)

    # Connect to the Milvus database
    connections.connect("default", uri=uri)

    # Create a new vector store and drop any existing one
    vector_store = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="Python_chat_bot",
        connection_args={"uri": uri},
        drop_old=True,
    )
    print("Vector Store Created")

    # # Check if the collection already exists
    # if utility.has_collection("Python_chat_bot"):
    #     print("Collection already exists. Loading existing Vector Store.")
    #     # loading the existing vector store
    #     vector_store = Milvus(
    #         collection_name="Python_chat_bot",
    #         embedding_function=get_embedding_function(),
    #         connection_args={"uri": uri}
    #     )
    # else:
    #     # Create a new vector store and drop any existing one
    #     vector_store = Milvus.from_documents(
    #         documents=docs,
    #         embedding=embeddings,
    #         collection_name="Python_chat_bot",
    #         connection_args={"uri": uri},
    #         drop_old=True,
    #     )
    #     print("Vector Store Created")
    return vector_store


def initialize_milvus(uri: str=MILVUS_URI):
    """
    Initialize the vector store for the RAG model

    Args:
        uri (str, optional): Path to the local milvus db. Defaults to
MILVUS_URI.

    Returns:
        vector_store: The vector store created
    """
    embeddings = get_embedding_function()
    print("Embeddings Loaded")
    documents = load_pdf(file_path)
    print("Documents Loaded")

    # Split the documents into chunks
    docs = split_documents(documents=documents)
    print("Documents Splitting completed")

    vector_store = create_vector_store(docs, embeddings, uri)

    return vector_store


# Step 4: Create a vector store to save embeddings
def load_exisiting_vector_storage(uri=MILVUS_URI):
    # Load an existing vector store
    vector_store = Milvus(
        collection_name="Python_chat_bot",
        embedding_function=get_embedding_function(),
        connection_args={"uri": uri},
    )
    print("Vector Store Loaded")
    return vector_store


# Step 4: Create a prompt
def create_prompt():
    """
    Create a prompt template for the RAG model

    Returns:
        PromptTemplate: The prompt template for the RAG model
    """
    # Define the prompt template
    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    Only use the information provided in the <context> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    <question>
    {input}
    </question>

    The response should be specific and use statistics or numbers when possible.

    Assistant:"""

    # Create a PromptTemplate instance with the defined template and input variables
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
    print("Prompt Created")
    return prompt


def query_rag(query):

    # Define the model
    model = ChatMistralAI(model='open-mistral-7b', api_key=MISTRAL_API_KEY)
    print("Model Loaded")

    prompt = create_prompt()

    # Load the vector store and create the retriever
    vector_store = load_exisiting_vector_storage(uri=MILVUS_URI)
    retriever = vector_store.as_retriever()

    document_chain = create_stuff_documents_chain(model, prompt)
    print("Document Chain Created")

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("Retrieval Chain Created")

    # Generate a response to the query
    repsonse = retrieval_chain.invoke({"input": f"{query}"})
    print("Response Generated")

    return repsonse["answer"]
