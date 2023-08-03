#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Milvus
from langchain.llms import GPT4All, LlamaCpp, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import argparse
import time
import torch


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
collection_name = os.environ.get('MILVUS_COLLECTION_NAME')
milvus_h = os.environ.get('MILVUS_HOST')
milvus_p = os.environ.get('MILVUS_PORT')
model_n_ctx = int(os.environ.get('MODEL_N_CTX'))
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    milvus_store = Milvus(embedding_function=embeddings,
        collection_name=collection_name,
        drop_old = False,
        connection_args={"host": milvus_h, "port": milvus_p})

    retriever = milvus_store.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    if model_type == "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, 
                max_tokens=model_n_ctx, 
                n_batch=model_n_batch, 
                callbacks=callbacks, 
                verbose=False
                )
    elif model_type == "GPT4All":
        llm = GPT4All(model=model_path, 
                max_tokens=model_n_ctx, 
                backend='gptj', 
                n_batch=model_n_batch, 
                callbacks=callbacks, 
                verbose=False
                )
    elif model_type == "HuggingFace":
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                torch_dtype=torch.float16,load_in_8bit=True
                )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        generate_text = pipeline(model=model, 
                tokenizer=tokenizer,
                return_full_text=True,  # langchain expects the full text
                task='text-generation',
                # we pass model parameters here too
                #stopping_criteria=stopping_criteria,  # without this model rambles during chat
                temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                max_new_tokens=model_n_ctx,  # mex number of tokens to generate in the output
                repetition_penalty=1.1  # without this output begins repeating
                )
        llm = HuggingFacePipeline(pipeline=generate_text)
    else:
        # raise exception if model_type is not supported
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    
# For chain type info, see https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
# options are : stuff, map_reduce, refine or map-rerank
    qa = RetrievalQA.from_chain_type(llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents= not args.hide_source
            )
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()


if __name__ == "__main__":
    main()
