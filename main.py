import os
import gradio as gr
import whisper

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Global state
qa_chain = None

# Load Whisper model once
whisper_model = whisper.load_model("base")  # Or use "small", "medium", etc.

# 1. Process PDF
def process_pdf(file):
    global qa_chain

    # Load the PDF
    loader = PyPDFLoader(file)
    pages = loader.load()

    if not pages:
        return "❌ PDF is empty or could not be read."

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    # Embed using Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Load Ollama LLM
    llm = Ollama(model="llama3.1", base_url="http://localhost:11434")

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return "✅ PDF processed! You can now ask questions."

# 2. Ask a question (text or edited transcription)
def ask_question(query):
    global qa_chain
    if qa_chain is None:
        return "❌ Please upload and process a PDF first.", ""

    result = qa_chain(query)
    answer = result["result"]
    sources = result["source_documents"]

    # Format source excerpts
    excerpts = "\n\n---\n\n".join(
        [f"📄 **Source Excerpt {i+1}:**\n{doc.page_content.strip()}" for i, doc in enumerate(sources)]
    )

    return answer, excerpts

# 3. Transcribe audio only
def transcribe_audio(audio_path):
    if not audio_path:
        return "❌ No audio file received."

    try:
        result = whisper_model.transcribe(audio_path)
        question = result["text"]
    except Exception as e:
        return f"❌ Whisper failed: {str(e)}"

    if not question.strip():
        return "❌ Could not detect any speech."

    return question

# 4. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧾 Rag for PDF using LangChain + Ollama + Whisper")
    gr.Markdown("Upload a PDF, then ask questions via text or voice. All models run locally.")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        upload_button = gr.Button("Process PDF")

    status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("💬 Ask by Text"):
        question = gr.Textbox(label="Ask a question")
        submit_button = gr.Button("Submit Question")

    with gr.Tab("🎤 Ask by Voice"):
        audio_input = gr.Audio(type="filepath", label="🎤 Record your question", interactive=True)
        transcribed_question = gr.Textbox(label="📝 Transcribed Text ", lines=2)
        transcribe_button = gr.Button("🎧 Transcribe Audio")
        submit_voice_question = gr.Button("❓ Ask Question")

    answer = gr.Textbox(label="Answer", lines=4, interactive=False)
    sources = gr.Markdown(label="📚 Source Excerpts")

    # Wiring up buttons
    upload_button.click(process_pdf, inputs=pdf_file, outputs=status)
    submit_button.click(ask_question, inputs=question, outputs=[answer, sources])
    transcribe_button.click(transcribe_audio, inputs=audio_input, outputs=transcribed_question)
    submit_voice_question.click(ask_question, inputs=transcribed_question, outputs=[answer, sources])

# 5. Launch the app
if __name__ == "__main__":
    demo.launch()
