import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

documents = SimpleDirectoryReader("/kaggle/working/LLM/data_2").load_data()



system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
## Default format supportable by LLama2
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

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
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
)

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()




def process_text(input_string):
    response = query_engine.query(input_string)
    return response

# Define the Gradio interface
iface = gr.Interface(
    fn=process_text,  # Function to process the input text
    inputs=gr.Textbox(lines=5, placeholder="Enter your text here..."),  # Text input
    outputs=gr.Textbox(lines=5),  # Text output
    title="Text Input and Output Example",  # Title of the Gradio interface
    description="Enter text in the input box and see the processed text in the output box."  # Description
)

# Launch the Gradio interface
iface.launch()

