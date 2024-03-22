import streamlit as st
from dotenv import load_dotenv
import os
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import SimpleDirectoryReader
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core.callbacks import CallbackManager

# # Initialize LlamaIndexCallbackHandler
# langfuse_callback_handler = LlamaIndexCallbackHandler(
#     public_key='pk-lf-3b89f8c1-e7c0-4cef-a8de-5a75a1821c10',
#     secret_key='sk-lf-8a2208e8-1ed2-4ba7-9206-37076211a7e0',
#     host="https://cloud.langfuse.com"
# )

# # Set the callback manager
# Settings.callback_manager = CallbackManager([langfuse_callback_handler])

# Set up environment variables and initialize models
def setup_environment_and_models():
    load_dotenv()
    os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets['LLAMA_CLOUD_API_KEY']
    os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
    os.environ['ANTHROPIC_API_KEY'] = st.secrets['ANTHROPIC_API_KEY']
    # Set up OpenAI LLM and embedding model
    llm = Anthropic(model="claude-3-sonnet-20240229", temperature=0.0)
    Settings.llm = llm
    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# Load data from files and create vector store indices
def create_index():
    parser = LlamaParse(api_key=os.getenv('LLAMA_CLOUD_API_KEY'), result_type="markdown", verbose=True)
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader("./wfu_docs", file_extractor=file_extractor).load_data()
    node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0125", temperature=0.0), num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    recursive_index = VectorStoreIndex(nodes=base_nodes+objects)
    return recursive_index

# Set up reranker and recursive query engine
def setup_query_engine(recursive_index):
    reranker = FlagEmbeddingReranker(top_n=5, model="BAAI/bge-reranker-large")
    recursive_query_engine = recursive_index.as_query_engine(
        similarity_top_k=15, node_postprocessors=[reranker], verbose=True
    )
    return recursive_query_engine

# Streamlit app
def main():
    st.set_page_config(page_title="MSBA Program Exploration", layout="centered", initial_sidebar_state="collapsed")
    st.title("🎩 WFU Master in Business Analytics Program Exploration")

    # Initialize environment and models if not already done
    if 'setup_done' not in st.session_state:
        setup_environment_and_models()
        st.session_state['setup_done'] = True

    # Check if recursive_query_engine exists in session_state, if not, create it
    if 'recursive_query_engine' not in st.session_state:
        recursive_index = create_index()
        st.session_state['recursive_query_engine'] = setup_query_engine(recursive_index)

    # Streamlit UI components
    query = st.text_input("👉 Enter your query:", placeholder="Ask me anything about the MSBA program...")
    if st.button("Search"):
        with st.spinner("Searching..."):
            # try:
            response = st.session_state['recursive_query_engine'].query(query).response
            st.markdown(f"**Answer:**\n\n{response}")
                # Flush the callback handler after displaying the response
            #     langfuse_callback_handler.flush()
            # except Exception as e:
            #     st.error(f"An error occurred during the search: {e}")
            #     # Log the error for further investigation
            #     print(f"Error during search: {e}")

if __name__ == "__main__":
    main()
