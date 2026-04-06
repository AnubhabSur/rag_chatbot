import os
import tempfile
import gradio as gr
from dotenv import load_dotenv
from ingest import ingest_pdf
from rag import build_chain

load_dotenv()
google_api_key= os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]= google_api_key

chain = build_chain()

def upload_pdf(file):
    if file is None:
        return "No file uploaded."
    try:
        ingest_pdf(file.name)
        return f"'{os.path.basename(file.name)}' ingested successfully!"
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}"

def chat(message, history):
    try:
        result = chain.invoke(message)
        return result
    except Exception as e:
        return f"Error processing query: {str(e)}. Please try again in a moment."

with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("## RAG Chatbot\nUpload a PDF, then ask questions about it.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Ingest PDF")
            status = gr.Textbox(label="Status", interactive=False)
            upload_btn.click(upload_pdf, inputs=pdf_input, outputs=status)

        with gr.Column(scale=2):
            gr.ChatInterface(fn=chat)

if __name == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

demo.launch()