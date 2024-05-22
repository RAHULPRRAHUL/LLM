from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
import gradio as gr

def process_text(input_text):



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

    response=query_engine.query(input_text)

    return response

iface = gr.Interface(
    fn=process_text,  # Function to process input
    inputs="text",    # Input type
    outputs="text"    # Output type
)

iface.launch()
   
    
    
    
    
    
    
    
    
    
