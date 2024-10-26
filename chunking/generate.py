import json
from llama_index.llms.openai import OpenAI
import json

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.readers.json import JSONReader
import glob
from torch.utils.data import DataLoader, Dataset
import torch
import json
from llama_index.finetuning import generate_qa_embedding_pairs

def generate_partitioned_qa_embedding_pairs(llm, nodes, base_output_path, partition_size=1000):
    """
    Generates QA embedding pairs and saves them to partitioned JSON files.
    
    Args:
        llm (OpenAI): The language model instance.
        nodes (list): List of nodes for generating embedding pairs.
        base_output_path (str): Base path for the output JSON files.
        partition_size (int): Number of nodes per partition file.
    """
    # Partition nodes into chunks
    for i in range(0, len(nodes), partition_size):
        chunk_nodes = nodes[i:i + partition_size]
        output_path = f"{base_output_path}_part_{i // partition_size + 1}.json"
        
        # Generate QA embedding pairs for the current chunk
        dataset_chunk = generate_qa_embedding_pairs(
            llm=llm,
            nodes=chunk_nodes,
            output_path=output_path
        )
        
        # Save chunk to file
        with open(output_path, 'w') as f:
            json.dump(dataset_chunk, f)
        print(f"Saved partitioned dataset to {output_path}")

# Use the function for both training and validation datasets
llm = OpenAI(model="gpt-3.5-turbo")

TRAIN_CORPUS_FPATH = "/teamspace/studios/this_studio/dataset/rag_test/text"
VAL_CORPUS_FPATH = "/teamspace/studios/this_studio/dataset/rag_test/text_val"
def load_corpus(files, verbose=False):
        if verbose:
            print(f"Loading files {files}")

        reader = SimpleDirectoryReader(input_dir=files)
        docs = reader.load_data()
        if verbose:
            print(f"Loaded {len(docs)} docs")

        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

        if verbose:
            print(f"Parsed {len(nodes)} nodes")

        return nodes

train_nodes = load_corpus(TRAIN_CORPUS_FPATH, verbose=True)
val_nodes = load_corpus(VAL_CORPUS_FPATH, verbose=True)

generate_partitioned_qa_embedding_pairs(
    llm=llm,
    nodes=train_nodes,
    base_output_path="train_dataset",
    partition_size=25  # Adjust as needed
)

generate_partitioned_qa_embedding_pairs(
    llm=llm,
    nodes=val_nodes,
    base_output_path="val_dataset",
    partition_size=25  # Adjust as needed
)
