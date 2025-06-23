import json
import os
import chromadb
import autogen
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.retrieve_utils import TEXT_FORMATS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# First, install required packages:
# pip install google-generativeai pyautogen chromadb python-dotenv openai

# Configuration for Gemini 2.0 Flash
config_list = [
    {
        "model": "gemini-2.0-flash-exp",
        "api_key": os.getenv("GOOGLE_API_KEY"),  # Load from .env file
        "api_type": "google",
    }
]

# Validate API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

# Create AssistantAgent instance with better system message
assistant = AssistantAgent(
    name="assistant",
    system_message="""You are a helpful assistant that provides accurate information based on retrieved context. 
    When given context from documents, use that information to answer questions accurately and cite the relevant parts.
    Always use the provided context to answer questions. If you receive retrieved context, base your response on it.
    If no relevant context is provided, state that you cannot find relevant information in the documents.""",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "temperature": 0.1,  # Lower temperature for more consistent responses
    },
)

def create_flaml_rag_agent():
    """Create RAG agent for FLAML documentation"""
    # Use more reliable documentation sources
    docs_path = [
        "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
        "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
        "https://raw.githubusercontent.com/microsoft/FLAML/main/README.md",
    ]
    
    # Check if local docs exist, otherwise use remote only
    local_docs = os.path.join(os.path.abspath(""), "..", "website", "docs")
    if os.path.exists(local_docs):
        docs_path.append(local_docs)
    
    return RetrieveUserProxyAgent(
        name="flaml_ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: True,  # Terminate after getting response
        retrieve_config={
            "task": "code",
            "docs_path": docs_path,
            "custom_text_types": ["mdx"],
            "chunk_token_size": 1500,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="/tmp/chromadb"),
            "collection_name": "flaml_docs",
            "get_or_create": True,
        },
        code_execution_config={"use_docker": False}
    )

def create_natural_questions_rag_agent():
    """Create RAG agent for Natural Questions dataset"""
    corpus_file = "https://huggingface.co/datasets/thinkall/NaturalQuestionsQA/resolve/main/corpus.txt"
    
    return RetrieveUserProxyAgent(
        name="nq_ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        is_termination_msg=lambda x: True,  # Terminate after getting response
        retrieve_config={
            "task": "qa",
            "docs_path": corpus_file,
            "chunk_token_size": 1500,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="/tmp/chromadb"),
            "collection_name": "natural-questions",
            "chunk_mode": "one_line",
            "get_or_create": True,
        },
        code_execution_config={"use_docker": False}
    )

def create_multihop_rag_agent():
    """Create RAG agent for multi-hop QA with customized prompt"""
    PROMPT_MULTIHOP = """You are a retrieve augmented chatbot. Answer questions based on the provided context and your knowledge. Think step-by-step.

Learn from these examples:

Context: Kurram Garhi is a village near Bannu, Pakistan. Trojkrsti is a village in Macedonia.
Q: Are both villages in the same country?
A: Kurram Garhi is in Pakistan. Trojkrsti is in Macedonia. They are not in the same country. Answer: no.

Context: Early Side of Later (2004) by Matt Goss reached No. 78 UK. What's Inside is Joan Armatrading's 14th album.
Q: Which was released earlier, What's Inside or Cassandra's Dream?
A: What's Inside was released in 1995. Cassandra's Dream was released in 2008. What's Inside was earlier. Answer: What's Inside.

Now answer this question:

Context: {input_context}
Q: {input_question}
A:"""

    corpus_file_2wiki = "https://huggingface.co/datasets/thinkall/2WikiMultihopQA/resolve/main/corpus.txt"
    
    return RetrieveUserProxyAgent(
        name="multihop_ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: True,  # Terminate after getting response
        retrieve_config={
            "task": "qa",
            "docs_path": corpus_file_2wiki,
            "chunk_token_size": 1500,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="/tmp/chromadb"),
            "collection_name": "2wikimultihopqa",
            "chunk_mode": "one_line",
            "customized_prompt": PROMPT_MULTIHOP,
            "customized_answer_prefix": "the answer is",
            "get_or_create": True,
        },
        code_execution_config={"use_docker": False}
    )

def create_local_docs_rag_agent(docs_path, collection_name="local_docs"):
    """Create RAG agent for local documents with improved configuration"""
    # Expand user path (handles ~ for home directory)
    docs_path = os.path.expanduser(docs_path)
    docs_path = os.path.abspath(docs_path)
    
    if not docs_path or not os.path.exists(docs_path):
        print(f"Error: Path '{docs_path}' does not exist.")
        return None
    
    # Check if path contains any supported file types
    supported_extensions = ['.txt', '.md', '.pdf', '.doc', '.docx', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']
    has_files = False
    found_files = []
    
    if os.path.isfile(docs_path):
        # If it's a single file
        if any(docs_path.lower().endswith(ext) for ext in supported_extensions):
            has_files = True
            found_files.append(docs_path)
    else:
        # If it's a directory
        for root, dirs, files in os.walk(docs_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    has_files = True
                    found_files.append(os.path.join(root, file))
                if len(found_files) >= 10:  # Limit for display
                    break
            if has_files and len(found_files) >= 10:
                break
    
    if not has_files:
        print(f"Warning: No supported document files found in '{docs_path}'")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return None
    else:
        print(f"Found {len(found_files)} supported files (showing first 10):")
        for file in found_files[:10]:
            print(f"  - {file}")
        if len(found_files) > 10:
            print(f"  ... and {len(found_files) - 10} more files")
    
    try:
        # Create unique collection name to avoid conflicts
        import time
        unique_collection_name = f"{collection_name}_{int(time.time()) % 10000}"
        
        print(f"Creating RAG agent with collection: {unique_collection_name}")
        
        # Ensure ChromaDB directory exists
        chromadb_path = "/tmp/chromadb"
        os.makedirs(chromadb_path, exist_ok=True)
        
        # Create the RAG agent with improved configuration
        rag_agent = RetrieveUserProxyAgent(
            name="local_ragproxyagent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
            retrieve_config={
                "task": "qa",
                "docs_path": [docs_path] if isinstance(docs_path, str) else docs_path,
                "chunk_token_size": 2000,  # Larger chunks for better context
                "model": config_list[0]["model"],
                "client": chromadb.PersistentClient(path=chromadb_path),
                "collection_name": unique_collection_name,
                "get_or_create": True,
                "custom_text_types": ["txt", "md", "py", "js", "html", "css", "json", "xml", "csv"],
                "embedding_model": "all-MiniLM-L6-v2",
                "n_results": 5,  # Retrieve more relevant chunks
                "distance_threshold": 0.7,  # Similarity threshold
                "overwrite": False,  # Don't overwrite existing collections
            },
            code_execution_config=False  # Disable code execution for safety
        )
        
        return rag_agent
        
    except Exception as e:
        print(f"Error creating local docs RAG agent: {e}")
        import traceback
        traceback.print_exc()
        return None

def initialize_rag_agent(rag_agent):
    """Properly initialize the RAG agent and ensure documents are processed"""
    print("Initializing RAG agent and processing documents...")
    
    try:
        # Force document processing by triggering the internal initialization
        if hasattr(rag_agent, '_init_db'):
            print("Initializing database and processing documents...")
            rag_agent._init_db()
            print("✅ Database initialized successfully.")
            
            # Now try to retrieve some documents to verify
            if hasattr(rag_agent, 'retrieve_docs'):
                print("Testing document retrieval...")
                test_docs = rag_agent.retrieve_docs("test", n_results=1)
                
                if test_docs and len(test_docs) > 0:
                    print(f"✅ Successfully retrieved {len(test_docs)} test documents.")
                    return True
                else:
                    print("⚠️ No documents were retrieved during test.")
                    return False
            else:
                print("✅ Database initialized but no retrieve_docs method found.")
                return True
                
        else:
            print("⚠️ Agent doesn't have _init_db method")
            return False
            
    except Exception as e:
        print(f"Error during initialization: {e}")
        print("Trying alternative initialization method...")
        
        # Alternative: try to force initialization through a simple chat
        try:
            # This will force the agent to initialize its database
            if hasattr(rag_agent, 'initiate_chat'):
                print("Attempting initialization through chat...")
                # Create a temporary simple assistant for initialization
                temp_assistant = AssistantAgent(
                    name="temp_init_assistant",
                    system_message="You are a temporary assistant for initialization.",
                    llm_config={"config_list": config_list}
                )
                
                # This should trigger document processing
                rag_agent.initiate_chat(
                    temp_assistant,
                    message="Initialize",
                    max_turns=1,
                    clear_history=True
                )
                print("✅ Alternative initialization completed.")
                return True
            else:
                print("⚠️ No suitable initialization method found.")
                return False
                
        except Exception as alt_e:
            print(f"Alternative initialization also failed: {alt_e}")
            return False

def get_user_input():
    """Get multi-line user input"""
    print("\nEnter your question or problem:")
    print("(Press Enter twice for single-line input, or type 'END' on a new line for multi-line input)")
    
    lines = []
    empty_line_count = 0
    
    while True:
        try:
            line = input()
            
            # Check if user typed END to finish multi-line input
            if line.strip().upper() == "END":
                break
            
            # Handle empty lines for single-line input detection
            if line.strip() == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
                lines.append(line)
            else:
                empty_line_count = 0
                lines.append(line)
                
        except EOFError:
            break
    
    # Remove trailing empty lines
    while lines and lines[-1].strip() == "":
        lines.pop()
    
    result = "\n".join(lines).strip()
    return result

def chat_with_rag_agent(rag_agent, assistant, user_problem, max_rounds=3):
    """Enhanced chat function that ensures document retrieval"""
    print(f"Processing query: '{user_problem[:50]}...' if longer")
    
    try:
        # Reset the assistant's conversation history first
        assistant.reset()
        
        print("Starting conversation with document context...")
        print("-" * 50)
        
        # Use initiate_chat directly - let the RAG agent handle document retrieval internally
        result = rag_agent.initiate_chat(
            assistant, 
            message=user_problem,
            max_turns=max_rounds,
            clear_history=True
        )
        
        return result
        
    except Exception as e:
        print(f"Error during chat: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a fallback approach
        print("Attempting fallback chat method...")
        try:
            # Simple fallback without manual document retrieval
            assistant.reset()
            
            # Try a more direct approach
            response = rag_agent.generate_reply(
                messages=[{"role": "user", "content": user_problem}],
                sender=assistant
            )
            
            if response:
                print("Assistant response:", response)
                return response
            else:
                print("No response generated")
                return None
                
        except Exception as fallback_e:
            print(f"Fallback method also failed: {fallback_e}")
            return None

def main():
    """Main interactive function"""
    print("=== Interactive RAG Agent ===")
    print("Choose a RAG agent type:")
    print("1. FLAML Documentation (for code generation and FLAML-related questions)")
    print("2. Natural Questions (for general knowledge questions)")
    print("3. Multi-hop QA (for complex reasoning questions)")
    print("4. Local Documents (for your own documents)")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "5":
                print("Goodbye!")
                break
            elif choice not in ["1", "2", "3", "4"]:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
                continue
            
            # Handle local documents case
            if choice == "4":
                print("Using Local Documents RAG Agent...")
                print("Enter the path to your documents directory or file:")
                print("(You can use ~ for home directory, e.g., ~/Documents/my_docs)")
                docs_path = input().strip()
                
                if not docs_path:
                    print("No path entered. Please try again.")
                    continue
                    
                collection_name = input("Enter a collection name (or press Enter for 'local_docs'): ").strip() or "local_docs"
                
                print(f"\nCreating RAG agent for path: {docs_path}")
                print("This may take a moment to process and index your documents...")
                
                try:
                    rag_agent = create_local_docs_rag_agent(docs_path, collection_name)
                    if rag_agent is None:
                        continue
                    
                    # Initialize the agent properly
                    if not initialize_rag_agent(rag_agent):
                        print("⚠️ Agent initialization had issues, but continuing anyway...")
                    
                    print("✅ Agent created successfully!")
                    
                    # Get user's problem/question
                    user_problem = get_user_input()
                    
                    if not user_problem:
                        print("No problem entered. Please try again.")
                        continue
                    
                    print(f"\nProcessing your request...")
                    print("=" * 50)
                    
                    # Use enhanced chat function
                    chat_result = chat_with_rag_agent(rag_agent, assistant, user_problem)
                    
                except Exception as e:
                    print(f"Error with Local Documents RAG agent: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            else:
                # For all other choices, get user problem first
                user_problem = get_user_input()
                
                if not user_problem:
                    print("No problem entered. Please try again.")
                    continue
                
                print(f"\nProcessing your request...")
                print("=" * 50)
                
                # Create appropriate RAG agent based on choice
                rag_agent = None
                
                if choice == "1":
                    print("Using FLAML Documentation RAG Agent...")
                    try:
                        rag_agent = create_flaml_rag_agent()
                        
                        # Initialize properly
                        initialize_rag_agent(rag_agent)
                        
                        print("Agent created successfully. Starting chat...")
                        chat_result = chat_with_rag_agent(rag_agent, assistant, user_problem)
                        
                    except Exception as e:
                        print(f"Error creating FLAML RAG agent: {e}")
                        continue
                        
                elif choice == "2":
                    print("Using Natural Questions RAG Agent...")
                    print("Initializing agent and downloading corpus (this may take a moment)...")
                    try:
                        rag_agent = create_natural_questions_rag_agent()
                        
                        # Initialize properly
                        initialize_rag_agent(rag_agent)
                        
                        print("Agent created successfully. Starting chat...")
                        chat_result = chat_with_rag_agent(rag_agent, assistant, user_problem)
                        
                    except Exception as e:
                        print(f"Error creating Natural Questions RAG agent: {e}")
                        print("This might be due to network issues or API limits. Please try again.")
                        continue
                    
                elif choice == "3":
                    print("Using Multi-hop QA RAG Agent...")
                    try:
                        rag_agent = create_multihop_rag_agent()
                        
                        # Initialize properly
                        initialize_rag_agent(rag_agent)
                        
                        print("Agent created successfully. Starting chat...")
                        chat_result = chat_with_rag_agent(rag_agent, assistant, user_problem)
                        
                    except Exception as e:
                        print(f"Error creating Multi-hop QA RAG agent: {e}")
                        continue
            
            print("\n" + "=" * 50)
            print("Question processed successfully!")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            print("Please try again.")

if __name__ == "__main__":
    main()