import logging
import uuid
import os
import pathlib
import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
#from langchain.vectorstores import Chroma
from langchain.vectorstores import Pinecone
import pinecone    

load_dotenv()

import pinecone      

pinecone.init(      
	api_key=os.getenv("PINECONE_API_KEY"),      
	environment='gcp-starter'      
)      
index = pinecone.Index('portfolio-ashwin')

logging.basicConfig(level=logging.INFO, format='=========== %(asctime)s :: %(levelname)s :: %(message)s')
MetadataFilter = Dict[str, Union[str, int, bool]]

class FaissIndex():
    def __init__(self):
        self.COLLECTION_NAME = "vector_store" 
        self.index_name = "portfolio-ashwin"
        logging.info(f"Loading Embedding model: all-mpnet-base-v2")
        #self.embeddings = HuggingFaceEmbeddings(model_name = "all-mpnet-base-v2")
        self.embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")
        #self.vector_store = FAISS.load_local("vector_store",self.embeddings)
        #self.vector_store = Chroma(persist_directory="chroma_db", embedding_function=self.embeddings)
        self.vector_store = Pinecone.from_existing_index(index_name=self.index_name, embedding=self.embeddings)
        logging.info(f"Loaded Vector Store: {self.COLLECTION_NAME}")
        self.template = """You are an AI assistant tailored for Ashwin Rachha. Your capabilities include:
                        - Providing insights and details about Ashwin Rachha's past experiences and achievements.
                        - Sharing information regarding professional endeavors and projects.
                        - Offering advice or recommendations based on Ashwin's preferences and interests.
                        - Sharing anecdotes or stories from Ashwin Rachha's life that are relevant to the question asked.
                        - Answering any question related to Ashwin Rachha's professional or personal life.
                        - Only answer things that are relevant to the question.
                        - Answer it all in first person
                        Question = {question}
                        {context}
                        """
        self.prompt = PromptTemplate(template = self.template, input_variables=['context', 'question'])
        self.chain_type_kwargs = {"prompt": self.prompt}
        logging.info(f"Initializing LLM")
        self.llm = ChatOpenAI(model_name = "gpt-4", temperature = 0)
        logging.info(f"Initializing Retrieval QA Chain")
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever = self.vector_store.as_retriever(), chain_type = "stuff", chain_type_kwargs=self.chain_type_kwargs)


