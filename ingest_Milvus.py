#!/usr/bin/env python3
import csv
import re
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

load_dotenv()


# Load environment variables
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
list_files = os.environ.get('LIST_FILES')
remote_path = os.environ.get('REMOTE_PATH')
collection_name = os.environ.get('MILVUS_COLLECTION_NAME')
milvus_h = os.environ.get('MILVUS_HOST')
milvus_p = os.environ.get('MILVUS_PORT')
docs_batch_size = int(os.environ.get('MAX_DOCS_BATCH_SIZE'))
chunk_size = 500
chunk_overlap = 50


def extract_project_metadata(filepath):
    """
    Extract metadata from a project filepath.
    :param filepath: str, the filepath of the project directory
    :return: dict, a dictionary containing the extracted metadata
    """

    # Initialize an empty dictionary to store the extracted metadata
    metadata = {}

    # Use regular expressions to extract the project number and folder names
    project_folder = r'^.*[\\\/]03 - Projets en cours[\\\/]'
    project_name = r'([0-9]{4})\s*-\s*([\sa-zA-Z0-9_-]+)'
    subfolder = r'[\\\/]([\sa-zA-Z0-9_-]+)[\\\/].*$'
    s_str = project_folder + project_name + subfolder

    match = re.search(r'^(.*[\\\/])?(\.?.*?)(\.[^.]*?|)$', filepath)
    if match:
        match2 = re.search(s_str, match.group(1)) 
        if match2:
            project_number, project_name, folder_name = match2.groups()
            metadata['project_number'] = project_number
            metadata['project_name'] = project_name
            metadata['folder_name'] = folder_name
    return metadata

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    #".csv": (CSVLoader, {}),
    #".docx": (Docx2txtLoader, {}),
    #".doc": (UnstructuredWordDocumentLoader, {}),
    #".docx": (UnstructuredWordDocumentLoader, {}),
    #".enex": (EverNoteLoader, {}),
    #".eml": (MyElmLoader, {}),
    #".epub": (UnstructuredEPubLoader, {}),
    #".html": (UnstructuredHTMLLoader, {}),
    #".md": (UnstructuredMarkdownLoader, {}),
    #".msg": (MyElmLoader, {}),
    #".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    #".ppt": (UnstructuredPowerPointLoader, {}),
    #".pptx": (UnstructuredPowerPointLoader, {}),
    #".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        try:
            loader = loader_class(file_path, **loader_args)
            data = loader.load()
            prj_metadata = extract_project_metadata(file_path)
            for doc in data:
                doc.metadata.update(prj_metadata)
            return data
        except Exception as e:
            # Add file_path to exception message
            print(f"Problem loading {file_path} skipping document: {e}")
            return ""
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    if os.path.isfile(list_files):
        with open(list_files, "r") as csvfile:
            reader_csv = csv.reader(csvfile, delimiter=";")
            next(reader_csv, None)  # skip the headers
            for row in reader_csv:
                # 3rd value is the extension of the file
                if row[2] in LOADER_MAPPING:
                    #Path is 2nd value and replace mount path
                    my_path = row[1]
                    my_path = my_path.replace(remote_path,source_dir)
                    my_path =  my_path.replace("\\","/")
                    all_files.append(my_path)                
    else:
        for ext in LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    #To debug
    #print(f"All len = {len(all_files)}")
    #print(f"Ignored len = {len(ignored_files)}")
    #print(f"Filtered len = {len(filtered_files)}")
    #return None
    
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()
    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def main():
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    milvus_store = Milvus(embedding_function=embeddings,
        collection_name=collection_name,
        drop_old = False,
        connection_args={"host": milvus_h, "port": milvus_p})
    
    if milvus_store.col:
        # Update and store new files if any
        all_res = []
        #Milvus has a max for collection rpc msg
        # Work around to extract all data without iterator of version 2.3
        res = milvus_store.col.query(
            expr = "pk >= 0",
            output_fields = ["pk"])
        for i in range(0,len(res),docs_batch_size):
            my_str = [x["pk"] for x in res[i:i+docs_batch_size]]
            my_str = ','.join(map(str,my_str))
            res_id = milvus_store.col.query(
                expr = "pk in [" + my_str + "]",
                output_fields = ["pk", "source"])
            all_res.extend(res_id)
        #remove file duplicates since used text_splitter
        all_src = list(set([metadata['source'] for metadata in all_res]))
        
        print(f"Appending to existing Milvus vectorstore collection {collection_name}")
        texts = process_documents(all_src)
        
        print(f"Creating embeddings. May take some minutes...")
        #Milvus seems to manage batch by itself, used with Chroma
        for i in range(0,len(texts),docs_batch_size):
            milvus_store.add_documents(texts[i:i+docs_batch_size])
    else:
        # Create and store locally vectorstore
        print("Creating new Milvus vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        # Set up a vector store used to save the vector embeddings. Here we use Milvus as the vector store.
        for i in range(0,len(texts),docs_batch_size):
            milvus_store.add_documents(texts[i:i+docs_batch_size])

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

if __name__ == "__main__":
    main()
