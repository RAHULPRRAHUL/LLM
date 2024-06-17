import gradio as gr

def process_text(input_text):
    # In this example, we just return the input text. You can add any text processing here.
    return input_text

# Define the Gradio interface
iface = gr.Interface(
    fn=process_text,  # Function to process the input text
    inputs=gr.Textbox(lines=5, placeholder="Enter your text here..."),  # Text input
    outputs=gr.Textbox(),  # Text output
    title="Text Input and Output Example",  # Title of the Gradio interface
    description="Enter text in the input box and see the processed text in the output box."  # Description
)

# Launch the Gradio interface
iface.launch()

