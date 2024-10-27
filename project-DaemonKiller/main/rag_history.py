import os
import sys
import argparse
from langchain.vectorstores import Chroma
from helper_functions import *
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryBufferMemory 
import yaml
with open("config.yml") as config:
    api_key = config["OPENAI_API_KEY"]
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
os.environ["OPENAI_API_KEY"] = "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"


class RAG:
    def __init__(self, n_retrieved=4, local_model_path="/teamspace/studios/this_studio/test_model/", model_kwargs={'device': 'cuda'}, encode_kwargs={'normalize_embeddings': True}):
        self.n_retrieved = n_retrieved
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=local_model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        # Initialize memory for conversation history summarization
        self.memory = ConversationSummaryBufferMemory(
            llm=OpenAI(),  # Use OpenAI as the summarization model
            max_token_limit=200  # Set a max token limit for the summary
        )

    def query(self, query):
        vectorstore = Chroma(
            collection_name="semantic_chunking_collection",
            embedding_function=self.embeddings,
            persist_directory="db"  # Directory where the vector store is saved
        )
        try:
            # Retrieve relevant documents with ensemble retriever
            semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": self.n_retrieved})
            documents = vectorstore.similarity_search(query, k=self.n_retrieved)
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = self.n_retrieved
            ensemble_retriever = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever], weights=[0.6, 0.4])
            ensemble_context = ensemble_retriever.get_relevant_documents(query)
        except:
            ensemble_context = []

        # Run the query through a RetrievalQA chain with memory for history
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=ensemble_retriever,
            memory=self.memory,  # Attach memory to the QA chain
            return_source_documents=True
        )

        # Get response and update memory with conversation context
        response = qa_chain({"query": query})
        return response['result'], response['source_documents']  # Return response and source documents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a PDF document with semantic chunking RAG.")
    parser.add_argument("--n_retrieved", type=int, default=10, help="Number of chunks to retrieve for each query.")
    parser.add_argument("--query", type=str, default="What gaming-related services and features does Swisscom offer, and what is the Swisscom Hero League?")
    return parser.parse_args()


def main(args):
    semantic_rag = RAG(n_retrieved=args.n_retrieved)
    result, sources = semantic_rag.query(args.query)
    print("Response:", result)
    print("Sources:", sources)


if __name__ == "__main__":
    main(parse_args())
