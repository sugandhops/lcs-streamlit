import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import pinecone

class ChatBot():
    def __init__(self):
        load_dotenv()
        self.loader = TextLoader('./horoscope.txt')
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)
        self.embeddings = HuggingFaceEmbeddings()
        
        # Initialize Pinecone client
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
        
        # Define Index Name
        self.index_name = "langchain-demo"
        
        # Checking Index
        if self.index_name not in pinecone.list_indexes():
            # Create new Index
            pinecone.create_index(name=self.index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, self.index_name)
        else:
            # Link to the existing index
            self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

        # Define the repo ID and connect to Mixtral model on Huggingface
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=repo_id, 
            model_kwargs={"temperature": 0.8, "top_k": 50}, 
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )
        
        # Create the prompt template
        template = """
        You are a fortune teller. These Human will ask you a questions about their life. 
        Use following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer: 

        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Chain it all together
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(),  "question": RunnablePassthrough()} 
            | self.prompt 
            | self.llm
            | StrOutputParser()
        )

# Test the ChatBot class
if __name__ == "__main__":
    bot = ChatBot()
    user_input = input("Ask me anything: ")
    result = bot.rag_chain.invoke(user_input)
    print(result)
