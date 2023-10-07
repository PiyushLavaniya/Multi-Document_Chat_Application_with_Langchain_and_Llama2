import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate, CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

##let's initialize the session-state, i.e let's create lists-> 'history', 'past', 'generated'
def create_session_state():
    if 'history' not in st.session_state:
        st.session_state['history']= []  ##This is an empty list that will contain the chatting history of user and the bot.
    
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ['Ask away...']  ##Here we also created an empty list that will contain the responses from the bot. We initialize it with 'ask away..', i.e. the first message that will be shown from the bot's side.

    if 'past' not in st.session_state: ##This is will also create an empty list tha will contain the messages from the users.
        st.session_state['past'] = ['Hey....']
        
        
        
##Now let's create our coversation, i.e querying the model and getting the response from it
def create_chat(query, chain, history):
    result = chain({'question': query, 'chat_history': history})   ##Here we are passing the query (user message) to the chain and passing these values to the chain to get the response from the LLM.
    history.append((query, result['answer']))    ##'chat_history' is the key that we defined when we created 'memory' and value 'history' is the parameter that we passed and the list that we created in the 'create_session_state' function.
    return result['answer']



##Now let's create a function to display the chat history
def display_chat_history(chain):
    ##We will make two containers that will contain the chat history or the stuff that we are wanting to display.
    reply_container = st.container()  ##This is for the reply.
    conversation_container = st.container()  ##This is for the normal conversation.
    
    with conversation_container:  ##Now we have opened this container and add things that we want this container to contain.
        ##First we will create a form, that will take input from the user
        with st.form(key = 'my_form', clear_on_submit = True):
            ##Now we are within this form
            user_input = st.text_input("Question: ", placeholder = 'Enter your text here.....', key = 'input')
            submit_button = st.form_submit_button(label = 'Send')
            
        if submit_button and user_input:
            with st.spinner('Typing...'):    ##We created an apinner to be shown when there is an input and the 'Send' button is clicked.
                ##Now within this spinner, we will perform tasks that will take place while the user will only see an spinner spinning
                output = create_chat(user_input, chain, st.session_state['history'])
                st.session_state['past'].append(user_input)   ##appending the 'past' list with the user_input
                st.session_state['generated'].append(output)  ##appending the 'generated' list with the bot's reponse.
                ##This will all take place when the user will hit the 'Send' button and an spinner will be displayed while all these tasks will be happening in the background.
    ##Now we will show the chat history in the 'reply_container'
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user = True, key = str(i) + '_user', avatar_style='human')
                message(st.session_state['generated'][i], key = str(i) + '_bot', avatar_style = 'robot')
                
##Let's create a chain for conversation
def create_conversational_chain(vectorstore):
    load_dotenv()
    ##Creating the LLM   
    #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        #streaming=True, 
                        #callbacks=[StreamingStdOutCallbackHandler()],
                       #model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    llm = Replicate(
        model = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e",   ##We get the LLM from 'Replicate', we can run it on internet, we do not need to download it.
        callbacks=[StreamingStdOutCallbackHandler()],
        input={'temperature': 1, 'max_length': 500, 'top_p': 1}
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_kwargs = {'k': 2}),
        memory = memory
    )
    return chain

                
def main():
    
    load_dotenv()
    
    create_session_state()   ##We are initializing the 'session states' here.
    
    st.title("Chat with Multiple Documents, :books:")
    st.sidebar.title("Upload your Documents here.")
    uploaded_files = st.sidebar.file_uploader("Click here to upload your Files", accept_multiple_files=True)
    
    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete = False) as temp_file:  ##making the temporary file for use.
                temp_file.write(file.read())   ##all the files we upload will be read into the temporary file
                temp_file_path = temp_file.name    ##getting the temp_file path for layer
        
        loader = None
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
            
        elif file_extension == '.docx' or '.doc':
            loader = Docx2txtLoader(temp_file_path)
            
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)

        if loader:
            text.extend(loader.load())  ##loading the file into the 'text' list
            os.remove(temp_file_path)  ##after work, we are deleting the 'temp_file_path'
            
            
        text_splitter = CharacterTextSplitter(separator = '\n', chunk_size = 1000, chunk_overlap = 1000, length_function = len)
        chunks = text_splitter.split_documents(text)
    
        ##Creating the embeddings
        embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device': 'cuda'})
    
        ##Creating the vectorstore
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    
        ##Let's create the chain using the function we created above
        chain = create_conversational_chain(vectorstore)
    
        ##Let's initialize our display_history function here
        display_chat_history(chain)
        
        
        
        
        
    
    
if __name__ == '__main__':
    main()
    