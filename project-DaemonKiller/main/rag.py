import os
import sys
import argparse
from langchain.vectorstores import Chroma
from helper_functions import *
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.llms import OpenAI
from generation import generate
import yaml
from langchain.chains import RetrievalQA
import logging
logging.basicConfig(
    filename='app.log',          # Log file name
    level=logging.INFO,           # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)
with open('/teamspace/studios/this_studio/project-DaemonKiller/main/config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
class RAG:
    def __init__(self, n_retrieved=4, local_model_path = "/teamspace/studios/this_studio/test_model/", model_kwargs = {'device': 'cuda'}, encode_kwargs = {'normalize_embeddings': True}) -> None:
        self.n_retrieved = n_retrieved
        self.embeddings = OpenAIEmbeddings()
        # self.embeddings = HuggingFaceBgeEmbeddings(
        #     model_name=local_model_path,
        #     model_kwargs=model_kwargs,
        #     encode_kwargs=encode_kwargs
        # )

    def query(self, query):

        vectorstore = Chroma(
            collection_name="semantic_chunking_collection",
            embedding_function=self.embeddings,
            persist_directory="db"  # The directory where the vector store is saved
        )
        try:
            with open("history.json", "r") as json_file:
                hist = json.load(json_file)

            summary_text = hist["summary"][0]["summary_text"]
            semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": self.n_retrieved})
            semantic_context = retrieve_context_per_question(query+summary_text, semantic_retriever)
            documents = vectorstore.similarity_search(query+summary_text, k=self.n_retrieved) # k=vectorstore.index.ntotal
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = self.n_retrieved
            # bm25_context = bm25_retriever.retrieve_context_per_question(query, semantic_retriever)
            ensemble_retriever = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever],weights=[0.6, 0.4])
            ensemble_context = ensemble_retriever.get_relevant_documents(query+summary_text)
            logging.info(ensemble_context)
        except:
            ensemble_context = ""

        return ensemble_context
        # generate(query, ensemble_context)

        # Create a RetrievalQA chain
        # qa_chain = RetrievalQA.from_chain_type(llm=llm,
        # chain_type="stuff",
        # retriever=ensemble_retriever,
        # return_source_documents=True)
        # print("Semantic context:\n")
        # # show_context(semantic_context)
        # print('+++++++++++++++++++++++')
        # print("RRF:\n")
        # print("\n".join(f"Document {i}: {doc.page_content}\nMetadata: {doc.metadata}" for i, doc in enumerate(ensemble_context, 1)))
        # print('+++++++++++++++++++++++')
        # print(qa_chain({"query": query}))


        # show_context(semantic_context)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a PDF document with semantic chunking RAG.")
    # parser.add_argument("--path", type=str, default="/teamspace/studios/this_studio/dataset/rag_test/",
    #                     help="Path to the PDF file to encode.")
    parser.add_argument("--n_retrieved", type=int, default=10,
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
    parser.add_argument("--query", type=str, default="What gaming-related services and features does Swisscom offer, and what is the Swisscom Hero League?")
    parser.add_argument("--experiment", action="store_true",
                        help="Run the experiment to compare performance between semantic chunking and simple chunking.")
    return parser.parse_args()

def main(args):
   semantic_rag = RAG()
   semantic_rag.query(args.query)


if __name__ == "__main__":
    main(parse_args())