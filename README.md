# AutoRAG - Multi-Source Document Retrieval System

AutoRAG is a powerful Python-based Retrieval-Augmented Generation (RAG) system that combines AutoGen's multi-agent framework with Google's Gemini 2.0 Flash model. It enables intelligent question-answering by retrieving relevant context from multiple document sources including local files, online repositories, and specialized datasets.

## 🚀 Features

- **Multiple RAG Agents**: Choose from 4 different specialized agents
- **Local Document Support**: Process your own documents (PDF, Word, text, code files, etc.)
- **Online Repository Integration**: Access FLAML documentation and research datasets
- **Multi-hop Reasoning**: Handle complex questions requiring multiple pieces of information
- **Interactive CLI**: User-friendly command-line interface
- **Persistent Storage**: ChromaDB for efficient document indexing and retrieval
- **Flexible Configuration**: Easy setup with environment variables

## 📋 Supported Agent Types

### 1. FLAML Documentation Agent
- **Purpose**: Code generation and FLAML-related questions
- **Source**: Microsoft FLAML GitHub repository documentation
- **Best for**: AutoML, hyperparameter optimization, machine learning workflows

### 2. Natural Questions Agent
- **Purpose**: General knowledge questions
- **Source**: Natural Questions dataset from Google Research
- **Best for**: Factual questions, general knowledge queries

### 3. Multi-hop QA Agent
- **Purpose**: Complex reasoning questions
- **Source**: 2WikiMultihopQA dataset
- **Best for**: Questions requiring information from multiple sources

### 4. Local Documents Agent
- **Purpose**: Your own document collections
- **Source**: Local files and directories
- **Best for**: Custom knowledge bases, personal documents, company docs

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- Google AI API key (for Gemini 2.0 Flash)

### Step 1: Clone the Repository
```bash
git clone https://github.com/naakaarafr/AutoRAG.git
cd AutoRAG
```

### Step 2: Install Dependencies
```bash
pip install google-generativeai pyautogen chromadb python-dotenv openai
```

### Step 3: Set Up Environment Variables
Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

To get a Google AI API key:
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

## 🚀 Quick Start

### Basic Usage
```bash
python app.py
```

Follow the interactive prompts to:
1. Choose your RAG agent type (1-4)
2. Enter your question or problem
3. Get AI-powered answers with retrieved context

### Example Workflow

```
=== Interactive RAG Agent ===
Choose a RAG agent type:
1. FLAML Documentation (for code generation and FLAML-related questions)
2. Natural Questions (for general knowledge questions)
3. Multi-hop QA (for complex reasoning questions)
4. Local Documents (for your own documents)
5. Exit

Enter your choice (1-5): 4

Enter the path to your documents directory or file:
~/Documents/my_research_papers

Enter a collection name (or press Enter for 'local_docs'): research_papers

Creating RAG agent for path: /home/user/Documents/my_research_papers
This may take a moment to process and index your documents...

✅ Agent created successfully!

Enter your question or problem:
What are the main findings about neural networks in my research papers?
```

## 📁 Supported File Types

The Local Documents agent supports various file formats:

- **Text files**: `.txt`, `.md`
- **Documents**: `.pdf`, `.doc`, `.docx`
- **Code files**: `.py`, `.js`, `.html`, `.css`, `.json`, `.xml`
- **Data files**: `.csv`

## ⚙️ Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google AI API key (required)

### Model Configuration
The system uses Gemini 2.0 Flash by default with these settings:
- Temperature: 0.1 (for consistent responses)
- Timeout: 600 seconds
- Chunk size: 1500-2000 tokens
- Retrieval results: 5 most relevant chunks

### ChromaDB Storage
- Default location: `/tmp/chromadb`
- Persistent storage across sessions
- Automatic collection management

## 🔧 Advanced Usage

### Custom Document Processing
```python
from app import create_local_docs_rag_agent, chat_with_rag_agent

# Create agent for specific document collection
rag_agent = create_local_docs_rag_agent(
    docs_path="/path/to/documents",
    collection_name="my_custom_collection"
)

# Process questions programmatically
result = chat_with_rag_agent(rag_agent, assistant, "Your question here")
```

### Multi-line Input
For complex questions, you can use multi-line input:
```
Enter your question or problem:
How can I implement a machine learning pipeline that:
1. Preprocesses text data
2. Performs feature extraction
3. Trains multiple models
4. Evaluates performance
END
```

## 🔍 How It Works

1. **Document Ingestion**: Documents are processed and split into chunks
2. **Embedding Generation**: Text chunks are converted to embeddings using sentence transformers
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient retrieval
4. **Query Processing**: User questions are embedded and matched against stored documents
5. **Context Retrieval**: Most relevant document chunks are retrieved
6. **Response Generation**: Gemini 2.0 Flash generates answers using retrieved context

## 🔧 Troubleshooting

### Common Issues

**API Key Error**
```
ValueError: GOOGLE_API_KEY not found in environment variables
```
- Ensure your `.env` file contains the correct API key
- Verify the file is in the project root directory

**No Documents Found**
```
Warning: No supported document files found
```
- Check that your document path exists
- Verify files have supported extensions
- Ensure you have read permissions

**ChromaDB Errors**
- The system automatically creates `/tmp/chromadb` directory
- On Windows, this might cause issues - modify the path in the code if needed

**Network Issues**
- Online agents (FLAML, Natural Questions, Multi-hop) require internet access
- Firewall settings might block document downloads

### Performance Tips

- **Large Document Collections**: Processing many documents takes time on first run
- **Chunk Size**: Larger chunks (2000 tokens) provide more context but slower retrieval
- **Collection Names**: Use unique names to avoid conflicts between different document sets

## 📊 Performance Metrics

- **Retrieval Speed**: ~1-3 seconds for most queries
- **Accuracy**: Depends on document quality and question complexity
- **Memory Usage**: Scales with document collection size
- **Storage**: ChromaDB collections are compressed and efficient

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional file format support
- Enhanced preprocessing pipelines
- Alternative embedding models
- UI improvements
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [AutoGen](https://github.com/microsoft/autogen) - Multi-agent conversation framework
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database for embeddings
- [FLAML](https://github.com/microsoft/FLAML) - Fast library for automated machine learning
- [Google AI](https://ai.google.dev/) - Gemini API documentation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the code comments for detailed explanations
3. Create an issue in the repository
4. Ensure your environment meets all prerequisites
