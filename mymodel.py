from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

documents=SimpleDirectoryReader("/kaggle/working/LLM/data").load_data()

#####################

system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")



llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index=VectorStoreIndex.from_documents(documents,service_context=service_context)


query_engine=index.as_query_engine()



import argparse

def main():
    parser = argparse.ArgumentParser(description="A simple example of argparse.")
    
    # Add a string argument
    parser.add_argument('-s', '--string', type=str, required=True, help='A string argument')

    # Parse the arguments
    args = parser.parse_args()

    # Access the string argument
    input_string = args.string
    print(f'The input string is: {input_string}')

    response = query_engine.query(input_string)

    print(response)


if __name__ == "__main__":
    main()




   
    
    
    
    
    
    
    
    
    
