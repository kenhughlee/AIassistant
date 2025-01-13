import os
import torch
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
# embeddings = None
# tokenizer = None
# model = None

Watsonx_API = "JsTvsspcCYqoVErlVXbwE6oLPtI4njwG-hTUrT1Yjq_l"
Project_id= "88defe2d-a2f3-46c5-a83c-69ced659ae04" 


# Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, tokenizer, model
    
    params = {
        GenParams.MAX_NEW_TOKENS: 500, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }

    watsonx_llm = WatsonxLLM(
        model_id="meta-llama/llama-3-8b-instruct",
        url="https://us-south.ml.cloud.ibm.com",
        apikey=Watsonx_API,
        project_id=Project_id,
        params=params
    )

    my_credentials = {
        "apikey" : Watsonx_API,
        "url" : "https://us-south.ml.cloud.ibm.com"
    }


    # LLAMA2_model = Model(
    #     model_id= 'meta-llama/llama-3-8b-instruct', 
    #     credentials=my_credentials,
    #     params=params,
    #     project_id=Project_id  
    #     )

    # llm_hub = WatsonxLLM(model=LLAMA2_model)
    llm_hub = watsonx_llm 

    #Initialize embeddings using a pre-trained model to represent the text data.
    # embeddings =  HuggingFaceInstructEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2", 
    #     model_kwargs={"device": DEVICE}
    # )

    # Load the tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
    except Exception as e:
        print(f"Error initializing tokenizer or model: {e}")
        raise
        

# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain
    
    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split the document into chunks, set chunk_size=1024, and chunk_overlap=64. assign it to variable text_splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    
    # Create an embeddings database using Chroma from the split text chunks.
    # db = Chroma.from_documents(texts, embedding=embeddings)
    
    # Create embeddings using the model and tokenizer
    def create_embeddings(texts):
        print (f"tokenizer: {tokenizer}")
        print (f"automodel: {model}")
        if tokenizer is None or model is None:
            raise ValueError("Tokenizer or model is not initialized")
        inputs = tokenizer([doc.page_content for doc in texts], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    embeddings = create_embeddings(texts)

    # Create an embeddings database using Chroma
    db = Chroma.from_documents(texts, embedding=embeddings)
    
    # Build the QA chain, which utilizes the LLM and retriever for answering questions.
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever= db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False
    )


# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    # Pass the prompt and the chat history to the conversation_retrieval_chain object
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    
    answer =  output["result"]
    
    # Update the chat history
    chat_history.append((prompt, answer))

    # Return the model's response
    return answer
    

# Initialize the language model
init_llm()
