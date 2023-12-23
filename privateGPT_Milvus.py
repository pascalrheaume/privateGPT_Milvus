#!/usr/bin/env python3
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Milvus
from langchain.llms import GPT4All, LlamaCpp, HuggingFacePipeline, OpenLLM

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig, pipeline
import os
import argparse
import time
import torch

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
openllm_server_host = os.environ.get('OPENLLM_SERVER_HOST')
openllm_server_port = os.environ.get('OPENLLM_SERVER_PORT')
collection_name = os.environ.get('MILVUS_COLLECTION_NAME')
milvus_h = os.environ.get('MILVUS_HOST')
milvus_p = os.environ.get('MILVUS_PORT')
model_n_ctx = int(os.environ.get('MODEL_N_CTX'))
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
score_threshold  = float(os.environ.get('SCORE_THRESHOLD',0.8))
fetch_k = int(os.environ.get('FETCH_K',20))
lambda_mult = float(os.environ.get('LAMBDA_MULT',0.25))

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


def main():
    # Parse the command line arguments
    args = parse_arguments()
    # To select model see: https://huggingface.co/spaces/mteb/leaderboard
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    milvus_store = Milvus(embedding_function=embeddings,
        collection_name=collection_name,
        drop_old = False,
        connection_args={"host": milvus_h, "port": milvus_p},
        search_params={"HNSW": {"metric_type": "L2", "params": {"ef": fetch_k}}})

#Use Similarity
#    retriever = milvus_store.as_retriever(
#            search_kwargs={"k": target_source_chunks,'score_threshold': score_threshold}
#           )

#Use MMR when many similar documents (e.g. many versions of same documents)

    my_retriever = milvus_store.as_retriever(search_type="mmr",
                   search_kwargs={"k": target_source_chunks,
                                   'fetch_k':fetch_k, 
                                   'lambda_mult': lambda_mult}
               )

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
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config)
        tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        generation_config = GenerationConfig.from_pretrained(model_path)
        generation_config.max_new_tokens = model_n_ctx
        generation_config.temperature = 0.0001
        generation_config.top_p = 0.95
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15

        generate_text = pipeline(model=model, 
                tokenizer=tokenizer,
                return_full_text=True,  # langchain expects the full text
                task='text-generation',
                generation_config=generation_config,
                )
        llm = HuggingFacePipeline(pipeline=generate_text)
    elif model_type == "OpenLLM":
        openllm_url = "http://" + openllm_server_host + ":" + openllm_server_port
        print(openllm_url)
        #llm = OpenLLM(server_url=openllm_url, server_type='grpc')
        llm = OpenLLM(server_url=openllm_url, server_type='http')
    else:
        # raise exception if model_type is not supported
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    
    # Prompt
    # Optionally, pull from the Hub
    # from langchain import hub
    # prompt = hub.pull("rlm/rag-prompt")
    # Or, define your own:
#    template = """Answer the question based only on the following context:
#    {context}
#
#    Question: {question}
#    """
#    prompt = ChatPromptTemplate.from_template(template)

#    combine_template = "Write a summary of the following text:\n\n{summaries}"
#    combine_prompt_template = PromptTemplate.from_template(template=combine_template)
#
#    question_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
#    {context}
#    Question: {question}
#    Helpful Answer:"""
#    question_prompt_template = PromptTemplate.from_template(template=question_template)
#

    # For chain type info, see https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
    # options are : stuff, map_reduce, refine or map-rerank
    # See also https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
    qa = RetrievalQA.from_chain_type(llm=llm, 
            chain_type="stuff", 
#            chain_type="map_reduce",
#            chain_type="refine",
            retriever=my_retriever, 
            return_source_documents= not args.hide_source,
#            chain_type_kwargs={"question_prompt": question_prompt_template,
#                               "combine_prompt": combine_prompt_template}

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
