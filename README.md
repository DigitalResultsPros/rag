# RAG PDF Assistant

A powerful Retrieval-Augmented Generation (RAG) system that allows you to ask questions about PDF documents using both text and voice input. All AI models run locally for privacy and security.

## Features

- **📄 PDF Processing**: Upload and process PDF documents for question answering
- **💬 Text Queries**: Ask questions about your documents via text input
- **🎤 Voice Queries**: Use voice input with automatic transcription via OpenAI Whisper
- **🔍 Source Attribution**: View exact excerpts from the PDF that support each answer
- **🏠 Local AI Models**: All processing done locally using Ollama and Whisper
- **🌐 Web Interface**: User-friendly Gradio interface for easy interaction

## Technical Requirements

- Python 3.8+
- Ollama or LM Studio installed and running locally
- Sufficient RAM for local AI models (recommended 8GB+)

## Installation

1. **Install Ollama**: 
   ```bash
   # Visit https://ollama.ai/download and install Ollama
   # Or use curl on Linux/Mac:
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull Required Models**:
   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

3. **Set Up Python Environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Application**:
   ```bash
   python main.py
   ```

2. **Access the Web Interface**: 
   - The application will launch at `http://localhost:7860`
   - Upload a PDF file using the file uploader
   - Click "Process PDF" to prepare the document for questioning

3. **Ask Questions**:
   - Use the "Ask by Text" tab for typed questions
   - Use the "Ask by Voice" tab to record and transcribe voice questions
   - View answers with source citations from your PDF

## Project Structure

```
.
├── main.py              # Main application code
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .venv/              # Python virtual environment
```

## Dependencies

- `gradio`: Web interface framework
- `langchain`: RAG framework and document processing
- `langchain-community`: Community integrations for LangChain
- `langchain-ollama`: Ollama integration for LangChain
- `faiss-cpu`: Vector database for document retrieval
- `openai-whisper`: Speech-to-text transcription

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   - Ensure Ollama is running: `ollama serve`
   - Check if models are downloaded: `ollama list`

2. **Whisper Model Loading**:
   - First run may download Whisper models automatically
   - Ensure sufficient disk space for model files

3. **PDF Processing Errors**:
   - Try with simpler PDFs first
   - Ensure PDF is not password protected

4. **Memory Issues**:
   - Use smaller Whisper models ("base" instead of "medium")
   - Process smaller PDFs or use more RAM

### Port Already in Use

If port 7860 is occupied, Gradio will automatically use another port. Check the terminal output for the actual URL.

## Development

To modify or extend this project:

1. The main logic is in `main.py`
2. Model configurations can be adjusted in the code:
   - Whisper model size (line 16)
   - Ollama models (lines 34 and 38)
   - Text splitting parameters (line 30)

## License

This project is open source and available under the MIT License.