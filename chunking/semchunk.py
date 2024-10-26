
import time
import os
import sys
import argparse
# from dotenv import load_dotenv
from helper_functions import *
from langchain.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain.embeddings import HuggingFaceBgeEmbeddings
#from langchain_openai.embeddings import OpenAIEmbeddings
import os
import glob as glob
import torch
from math import ceil
from langchain_core.documents import Document
from uuid import uuid4

# Add the parent directory to the path since we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Load environment variables from a .env file (e.g., OpenAI API key)
# load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"



# Function to run semantic chunking and return chunking and retrieval times
class SemanticChunkingRAG:
    """
    A class to handle the Semantic Chunking RAG process for document chunking and query retrieval.
    """

    def __init__(self, path, n_retrieved=4, embeddings=None, breakpoint_type: BreakpointThresholdType = "percentile",
                 breakpoint_amount=90):
        """
        Initializes the SemanticChunkingRAG by encoding the content using a semantic chunker.

        Args:
            path (str): Path to the PDF file to encode.
            n_retrieved (int): Number of chunks to retrieve for each query (default: 2).
            embeddings: Embedding model to use.
            breakpoint_type (str): Type of semantic breakpoint threshold.
            breakpoint_amount (float): Amount for the semantic breakpoint threshold.
        """
        print("\n--- Initializing Semantic Chunking RAG ---")

        documents = []
        for f in glob.glob(os.path.join(path, '*.json')):
            with open(f, 'r') as js:
                data = json.load(js) 
                doc = Document(page_content= data["content"], 
                       metadata={"source": data["source"], "title": data["title"], "language": data["language"]})
                documents.append(doc)

        # content = load_json_docs(path)
        # # print([doc.page_content for doc in content][0])
        # texts = [doc.page_content for doc in content]
        # metadatas = [doc.metadata for doc in content]
        local_model_path = "/teamspace/studios/this_studio/test_model/"

        model_kwargs = {'device': 'cuda'}  # Use 'cpu' if you don't have a GPU
        encode_kwargs = {'normalize_embeddings': True}

        # Initialize the embedding model
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=local_model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.vector_store = Chroma(
            collection_name="semantic_chunking_collection",
            embedding_function=self.embeddings,
            persist_directory="db"  # Replace with actual storage path
        )
        
        # self.embeddings = embeddings if embeddings else OpenAIEmbeddings(model="text-embedding-3-large")
        self.semantic_chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount
        )
        # Batch settings
        batch_size = 2  # Adjust batch size based on your GPU memory capacity
        num_batches = ceil(len(documents) / batch_size)
        uuids = [str(uuid4()) for _ in range(len(documents))]

        start_time = time.time()

        # Initialize an empty FAISS vector store
        faiss_index = None

        # Process in batches
        start_time = time.time()
        for i in range(num_batches):
            with torch.no_grad():
                batch_texts = documents[i * batch_size: (i + 1) * batch_size]
                ids = uuids[i * batch_size: (i + 1) * batch_size]
                # batch_metadatas = metadatas[i * batch_size: (i + 1) * batch_size]


                # Add documents to the Chroma vector store
                self.vector_store.add_documents(documents=batch_texts, ids=ids)
                print(f"Processed batch {i + 1}/{num_batches}")
                self.vector_store.persist()

                # Clear CUDA cache and delete batch docs to manage memory
                # del batch_docs
                torch.cuda.empty_cache()

        # Record the time taken for chunking
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Semantic Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # Persist the vector store
        self.vector_store.persist()

    def run(self, query):
        """
        Retrieves and displays the context for the given query.

        Args:
            query (str): The query to retrieve context for.

        Returns:
            tuple: The retrieval time.
        """
        # Measure time for semantic retrieval
        start_time = time.time()
        semantic_context = retrieve_context_per_question(query, self.semantic_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Semantic Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Display the retrieved context
        show_context(semantic_context)
        return self.time_records


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a PDF document with semantic chunking RAG.")
    parser.add_argument("--path", type=str, default="/teamspace/studios/this_studio/dataset/cleandataset/",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--breakpoint_threshold_type", type=str,
                        choices=["percentile", "standard_deviation", "interquartile", "gradient"],
                        default="percentile",
                        help="Type of breakpoint threshold to use for chunking (default: percentile).")
    parser.add_argument("--breakpoint_threshold_amount", type=float, default=90,
                        help="Amount of the breakpoint threshold to use (default: 90).")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk in simple chunking (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks in simple chunking (default: 200).")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="Query to test the retriever (default: 'What is the main cause of climate change?').")
    parser.add_argument("--experiment", action="store_true",
                        help="Run the experiment to compare performance between semantic chunking and simple chunking.")

    return parser.parse_args()


# Main function to process PDF, chunk text, and test retriever
def main(args):
    # Initialize SemanticChunkingRAG
    semantic_rag = SemanticChunkingRAG(
        path=args.path,
        n_retrieved=args.n_retrieved,
        breakpoint_type=args.breakpoint_threshold_type,
        breakpoint_amount=args.breakpoint_threshold_amount
    )

    # Run a query
    semantic_rag.run(args.query)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())