from langchain.document_loaders import DirectoryLoader
# import pinecone
# from langchain.vectorstores import Pinecone
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import os
# from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
import streamlit as st
import openai
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

folder = r"AOMSI\AOMSI_book"

# directory = '/content/data'
openai.api_key = os.getenv('OPENAI_API_KEY')
model = SentenceTransformer('all-MiniLM-L6-v2')
# pinecone.init(api_key=os.getenv('pinecone_api_key'), environment='us-west1-gcp-free')
# index = pinecone.Index('medical-chatbot')
# index_name = "medical-chatbot"
    # return index_name

def load_docs(folder):
  loader = DirectoryLoader(folder)
  documents = loader.load()
  return documents



def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_File_Object = file

    # Here, we will creat a pdf reader object
        pdf_Reader = PyPDF2.PdfReader(pdf_File_Object)

    # Now we will print number of pages in pdf file
        print("No. of pages in the given PDF file: ", len(pdf_Reader.pages))
        num_pages = len(pdf_Reader.pages)
        for j in range(num_pages):
    # Here, create a page object
          page_Object = pdf_Reader.pages[j]

    # Now, we will extract text from page
          text = text + page_Object.extract_text()

    # At last, close the pdf file object
        pdf_File_Object.close()

    return text

def load_multiple_pdfs(folder_path):
    pdf_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_texts[filename] = extract_text_from_pdf(pdf_path)
    return pdf_texts
# documents = load_docs(folder)
# len(documents)



# def split_docs(documents,chunk_size=500,chunk_overlap=20):
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#   docs = text_splitter.split_documents(documents)
#   return docs

def split(raw_text):
  text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
  )
  texts = text_splitter.split_text(raw_text)
  return texts
  


documents = load_multiple_pdfs(folder)
# docs = split_docs(documents)
raw_text=""
for i in documents.keys():
  raw_text = raw_text + documents[i]
docs = split(raw_text)
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings()

# index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
# index = Pinecone.from_texts(docs, embeddings, index_name=index_name)
docsearch = FAISS.from_texts(docs, embeddings)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})


def get_conversation_string():
   conversation_string = ""
   for i in range(len(st.session_state['responses'])-1):
      conversation_string +="User: "+st.session_state['requests'][i] + "\n"
      conversation_string +="Bot: "+ st.session_state['responses'][i+1] + "\n"
      return conversation_string
   
def query_refiner(conversation, query):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

# def find_match(input):
#     input_em = model.encode(input).tolist()
#     result = index.query(input_em, top_k=10, includeMetadata=True)
#     return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

