import streamlit as st
import os
import shutil
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Load environment variables
load_dotenv()

# -----------------------------
# Configuration from .env
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Clean up uploads folder
def cleanup_uploads():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)

cleanup_uploads()

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_resource
def load_llm():
    return AzureChatOpenAI(
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0
    )

@st.cache_resource
def load_embeddings():
    return AzureOpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION
    )

@st.cache_resource
def create_vector_db(files):
    documents = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        ext = os.path.splitext(file.name)[-1].lower()

        # Save locally
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {ext}")
            continue

        doc = loader.load()
        documents.extend(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = load_embeddings()

    # Use Chroma 
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory=".chroma_db")
    return vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    """Format retrieved documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(messages):
    """Format chat history for context"""
    formatted = ""
    for msg in messages[:-1]:  # Exclude current message
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    return formatted

# -----------------------------
# RAGAS Evaluation Functions (FIXED)
# -----------------------------
def setup_ragas_llm_and_embeddings():
    """Setup RAGAS with Azure OpenAI LLM and embeddings"""
    try:
        # Create Azure OpenAI LLM for RAGAS
        azure_llm = AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            temperature=0
        )
        
        # Create Azure OpenAI Embeddings for RAGAS
        azure_embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=AZURE_OPENAI_API_VERSION
        )
        
        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(azure_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)
        
        return ragas_llm, ragas_embeddings
    
    except Exception as e:
        st.error(f"Error setting up RAGAS components: {str(e)}")
        return None, None

def create_ragas_dataset(questions, answers, contexts, ground_truths=None):
    """Create a dataset for RAGAS evaluation with better error handling"""
    try:
        # Ensure all inputs are lists and have same length
        questions = list(questions)
        answers = list(answers) 
        contexts = list(contexts)
        
        # Validate contexts format - RAGAS expects list of lists
        processed_contexts = []
        for ctx in contexts:
            if isinstance(ctx, str):
                processed_contexts.append([ctx])  # Wrap single string in list
            elif isinstance(ctx, list):
                processed_contexts.append(ctx)
            else:
                processed_contexts.append([str(ctx)])  # Convert to string and wrap
        
        data = {
            "question": questions,
            "answer": answers,
            "contexts": processed_contexts,
        }
        
        if ground_truths:
            ground_truths = list(ground_truths)
            # Ensure ground truths list matches questions length
            while len(ground_truths) < len(questions):
                ground_truths.append("")
            data["ground_truth"] = ground_truths[:len(questions)]
        
        # Debug info
        st.write(f"Debug - Creating dataset with {len(questions)} items")
        st.write(f"Debug - Sample question: {questions[0] if questions else 'None'}")
        st.write(f"Debug - Sample contexts type: {type(processed_contexts[0]) if processed_contexts else 'None'}")
        
        return Dataset.from_dict(data)
        
    except Exception as e:
        st.error(f"Error creating RAGAS dataset: {str(e)}")
        raise e

def evaluate_with_ragas(dataset, include_ground_truth_metrics=False):
    """Evaluate RAG system using RAGAS metrics"""
    try:
        # Setup RAGAS with Azure OpenAI
        ragas_llm, ragas_embeddings = setup_ragas_llm_and_embeddings()
        
        if ragas_llm is None or ragas_embeddings is None:
            st.error("Failed to setup RAGAS components")
            return None
        
        # Base metrics that don't require ground truth
        base_metrics = [
            faithfulness,
            answer_relevancy,
        ]
        
        # Metrics that require ground truth
        ground_truth_metrics = [
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity
        ]
        
        # Choose metrics based on data availability
        if include_ground_truth_metrics and "ground_truth" in dataset.column_names:
            metrics = base_metrics + ground_truth_metrics
            st.info("Running evaluation with ground truth metrics...")
        else:
            metrics = base_metrics
            st.info("Running evaluation with base metrics only (faithfulness, answer_relevancy)...")
        
        # Configure metrics with Azure OpenAI
        for metric in metrics:
            if hasattr(metric, 'llm'):
                metric.llm = ragas_llm
            if hasattr(metric, 'embeddings'):
                metric.embeddings = ragas_embeddings
        
        # Debug: Show dataset info
        st.write("Debug - Dataset columns:", list(dataset.column_names))
        st.write("Debug - Dataset size:", len(dataset))
        
        # Run evaluation
        st.info("Running evaluation... This may take a few minutes.")
        result = evaluate(dataset, metrics=metrics)
        
        return result
    
    except Exception as e:
        st.error(f"Error during RAGAS evaluation: {str(e)}")
        st.exception(e)
        return None

def display_ragas_results(result):
    """Display RAGAS evaluation results in Streamlit"""
    if result is None:
        st.error("No results to display - evaluation result is None")
        return
    
    st.subheader("📊 RAGAS Evaluation Results")
    
    try:
        metrics_data = {}
        
        # Handle different RAGAS result formats
        if hasattr(result, 'to_pandas'):
            try:
                # Convert to pandas and get mean values
                df = result.to_pandas()
                st.write("Debug - DataFrame shape:", df.shape)
                st.write("Debug - DataFrame columns:", df.columns.tolist())
                
                # Calculate mean for numeric columns only
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    metrics_data = df[numeric_cols].mean().to_dict()
                    st.write("Debug - Numeric metrics extracted:", metrics_data)
                else:
                    st.warning("No numeric columns found in evaluation results")
                    # Try to extract metrics from the dataframe manually
                    for col in df.columns:
                        try:
                            # Try to convert column to numeric
                            numeric_series = pd.to_numeric(df[col], errors='coerce')
                            if not numeric_series.isna().all():
                                metrics_data[col] = numeric_series.mean()
                        except:
                            continue
                
            except Exception as e:
                st.error(f"Error processing pandas dataframe: {e}")
                # Fallback: try to access result as dict
                if isinstance(result, dict):
                    metrics_data = {k: v for k, v in result.items() if isinstance(v, (int, float))}
        
        # If still no metrics, try alternative approaches
        if not metrics_data:
            # Try accessing as dictionary
            if isinstance(result, dict):
                # Filter for numeric values only
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        metrics_data[key] = value
            
            # Try accessing common RAGAS metric attributes
            metric_names = ['faithfulness', 'answer_relevancy', 'context_precision', 
                          'context_recall', 'answer_correctness', 'answer_similarity']
            
            for metric_name in metric_names:
                if hasattr(result, metric_name):
                    value = getattr(result, metric_name)
                    if isinstance(value, (int, float)):
                        metrics_data[metric_name] = value
                    elif hasattr(value, 'mean') and callable(value.mean):
                        try:
                            metrics_data[metric_name] = float(value.mean())
                        except:
                            pass
        
        if not metrics_data:
            st.error("Could not extract numeric metrics from RAGAS result")
            st.write("Raw result type:", type(result))
            st.write("Raw result content:", str(result)[:500] + "..." if len(str(result)) > 500 else str(result))
            return
        
        # Display metrics in a clean format
        st.success("✅ Evaluation completed successfully!")
        
        # Create columns for better layout
        cols = st.columns(min(3, len(metrics_data)))
        
        metric_descriptions = {
            'faithfulness': 'Factual Accuracy',
            'answer_relevancy': 'Answer Relevance', 
            'context_precision': 'Context Precision',
            'context_recall': 'Context Recall',
            'answer_correctness': 'Answer Correctness',
            'answer_similarity': 'Answer Similarity'
        }
        
        for i, (metric, value) in enumerate(metrics_data.items()):
            col_idx = i % len(cols)
            with cols[col_idx]:
                display_name = metric_descriptions.get(metric, metric.replace('_', ' ').title())
                
                # Color code based on score
                if value >= 0.8:
                    color = "green"
                elif value >= 0.6:
                    color = "orange"
                else:
                    color = "red"
                
                st.metric(
                    label=display_name,
                    value=f"{float(value):.3f}",
                    help=f"Score: {float(value):.3f} (Higher is better)"
                )
        
        # Show detailed explanation
        with st.expander("📋 Metrics Explanation"):
            st.markdown("""
            **Faithfulness (0-1)**: Measures factual consistency with the provided context.
            - Higher scores indicate answers are more factually grounded.
            
            **Answer Relevancy (0-1)**: Measures how well the answer addresses the question.
            - Higher scores indicate more relevant and on-topic answers.
            
            **Context Precision (0-1)**: Measures relevance of retrieved context.
            - Higher scores indicate better context retrieval.
            
            **Context Recall (0-1)**: Measures completeness of context retrieval.
            - Higher scores indicate more comprehensive context.
            
            **Answer Correctness (0-1)**: Measures factual similarity to ground truth.
            - Requires ground truth answers for comparison.
            
            **Answer Similarity (0-1)**: Measures semantic similarity to ground truth.
            - Requires ground truth answers for comparison.
            
            **Score Interpretation:**
            - 🟢 0.8-1.0: Excellent
            - 🟡 0.6-0.8: Good  
            - 🔴 0.0-0.6: Needs Improvement
            """)
        
        # Store results for download
        st.session_state.metrics_data = metrics_data
        
        # Summary insight
        avg_score = sum(metrics_data.values()) / len(metrics_data)
        if avg_score >= 0.8:
            st.success(f"🎉 Excellent performance! Average score: {avg_score:.3f}")
        elif avg_score >= 0.6:
            st.info(f"👍 Good performance! Average score: {avg_score:.3f}")
        else:
            st.warning(f"⚠️ Performance needs improvement. Average score: {avg_score:.3f}")
        
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        st.exception(e)

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="🧠 Document Q&A with RAGAS Evaluation", page_icon="📚")
st.title("🧠 Chat with Your Documents + RAGAS Evaluation")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "evaluation_data" not in st.session_state:
    st.session_state.evaluation_data = []

if "ground_truths" not in st.session_state:
    st.session_state.ground_truths = []

if "ragas_results" not in st.session_state:
    st.session_state.ragas_results = None

# Sidebar for RAGAS configuration
with st.sidebar:
    st.header("⚙️ RAGAS Configuration")
    
    enable_ragas = st.checkbox("Enable RAGAS Evaluation", value=False)
    
    if enable_ragas:
        st.info("RAGAS evaluation will track your Q&A interactions for assessment.")
        
        # Show current evaluation data count
        data_count = len(st.session_state.evaluation_data)
        st.info(f"Current Q&A pairs collected: {data_count}")
        
        # Option to add ground truth answers
        use_ground_truth = st.checkbox("Include Ground Truth Answers", value=False)
        
        if use_ground_truth:
            # Fix: Properly count ground truth answers
            gt_count = len([gt for gt in st.session_state.ground_truths if gt.strip()])  # Count non-empty ground truths
            st.info(f"Ground truth answers: {gt_count}/{data_count}")
            
            # Show ground truth input section immediately if there are Q&A pairs without ground truth
            if data_count > gt_count:
                st.write("**Add Ground Truth Answers:**")
                
                # Find the first Q&A pair without ground truth
                for i, item in enumerate(st.session_state.evaluation_data):
                    if i >= len(st.session_state.ground_truths) or not st.session_state.ground_truths[i].strip():
                        st.write(f"**Q{i+1}:** {item['question'][:50]}...")
                        ground_truth = st.text_area(
                            f"Ground truth for Q{i+1}:", 
                            key=f"gt_input_{i}",
                            height=80
                        )
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button(f"Save GT {i+1}", key=f"save_gt_{i}"):
                                # Ensure ground_truths list is long enough
                                while len(st.session_state.ground_truths) <= i:
                                    st.session_state.ground_truths.append("")
                                
                                st.session_state.ground_truths[i] = ground_truth
                                st.success(f"Ground truth {i+1} saved!")
                                st.rerun()
                        
                        with col2:
                            if st.button(f"Skip Q{i+1}", key=f"skip_gt_{i}"):
                                # Ensure ground_truths list is long enough
                                while len(st.session_state.ground_truths) <= i:
                                    st.session_state.ground_truths.append("")
                                
                                st.session_state.ground_truths[i] = ""  # Empty string for skipped
                                st.info(f"Skipped question {i+1}")
                                st.rerun()
                        break
    
    # Button to run evaluation
    if st.button("🔍 Run RAGAS Evaluation") and enable_ragas:
        if st.session_state.evaluation_data:
            # Clear previous results
            st.session_state.ragas_results = None
            
            with st.spinner("Running RAGAS evaluation..."):
                try:
                    # Prepare data for evaluation
                    questions = [item["question"] for item in st.session_state.evaluation_data]
                    answers = [item["answer"] for item in st.session_state.evaluation_data]
                    contexts = [item["contexts"] for item in st.session_state.evaluation_data]
                    
                    # Create dataset with proper ground truth handling
                    if (use_ground_truth and 
                        len(st.session_state.ground_truths) >= len(questions) and
                        any(gt.strip() for gt in st.session_state.ground_truths[:len(questions)])):
                        
                        # Filter out empty ground truths and corresponding Q&A pairs
                        filtered_data = []
                        for i, (q, a, c) in enumerate(zip(questions, answers, contexts)):
                            if i < len(st.session_state.ground_truths) and st.session_state.ground_truths[i].strip():
                                filtered_data.append((q, a, c, st.session_state.ground_truths[i]))
                        
                        if filtered_data:
                            f_questions, f_answers, f_contexts, f_ground_truths = zip(*filtered_data)
                            dataset = create_ragas_dataset(f_questions, f_answers, f_contexts, f_ground_truths)
                            result = evaluate_with_ragas(dataset, include_ground_truth_metrics=True)
                        else:
                            st.warning("No valid ground truth answers found!")
                            dataset = create_ragas_dataset(questions, answers, contexts)
                            result = evaluate_with_ragas(dataset, include_ground_truth_metrics=False)
                    else:
                        dataset = create_ragas_dataset(questions, answers, contexts)
                        result = evaluate_with_ragas(dataset, include_ground_truth_metrics=False)
                    
                    if result:
                        st.session_state.ragas_results = result
                        st.success("✅ RAGAS evaluation completed!")
                        st.rerun()
                    else:
                        st.error("❌ RAGAS evaluation failed!")
                        
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
                    st.exception(e)
        else:
            st.warning("No evaluation data available. Ask some questions first!")
    
    # Show evaluation status
    if enable_ragas:
        if st.session_state.evaluation_data:
            st.success(f"Ready to evaluate {len(st.session_state.evaluation_data)} Q&A pairs")
        else:
            st.info("Ask questions to collect evaluation data")
    
    # Clear evaluation data button
    if st.button("🗑️ Clear Evaluation Data"):
        st.session_state.evaluation_data = []
        st.session_state.ground_truths = []
        st.session_state.ragas_results = None
        st.success("Evaluation data cleared!")
        st.rerun()

# Display RAGAS results at the top if available
if st.session_state.ragas_results is not None:
    display_ragas_results(st.session_state.ragas_results)
    
    # Option to download results
    if "metrics_data" in st.session_state:
        if st.button("📥 Download RAGAS Results"):
            results_df = pd.DataFrame([st.session_state.metrics_data])
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="ragas_evaluation_results.csv",
                mime="text/csv"
            )

# Main app
uploaded_files = st.file_uploader("Upload PDF/DOCX/TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    try:
        retriever = create_vector_db(uploaded_files)
        llm = load_llm()

        # Define prompt template
        prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {query}

Answer:"""

        # Create custom chain with proper variable handling
        def create_qa_chain():
            prompt = PromptTemplate.from_template(prompt_template)
            
            def get_context_and_query(inputs):
                query = inputs["query"]
                docs = retriever.get_relevant_documents(query)
                context = format_docs(docs)
                
                return {
                    "context": context,
                    "query": query,
                    "source_documents": docs
                }
            
            def run_llm(inputs):
                response = llm.invoke([
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt.format(
                        context=inputs["context"],
                        query=inputs["query"]
                    )}
                ])
                
                return {
                    "result": response.content,
                    "source_documents": inputs["source_documents"],
                    "context": inputs["context"]
                }
            
            return get_context_and_query, run_llm

        context_retriever, llm_runner = create_qa_chain()
        st.session_state.qa_components = (context_retriever, llm_runner)

        st.success("✅ Ready! Ask a question about your documents.")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                # Get context and run LLM
                context_retriever, llm_runner = st.session_state.qa_components
                
                context_result = context_retriever({"query": prompt})
                result = llm_runner(context_result)
                
                answer = result["result"]
                source_docs = result.get("source_documents", [])
                context = result.get("context", "")

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if source_docs:
                        sources = list(set(doc.metadata.get('source', 'Unknown') for doc in source_docs))
                        st.caption("📄 Sources:")
                        for src in sources:
                            st.caption(f"- {os.path.basename(src)}")

                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Store data for RAGAS evaluation if enabled
                if enable_ragas:
                    evaluation_item = {
                        "question": prompt,
                        "answer": answer,
                        "contexts": context  # Store as string, will be converted to list in create_ragas_dataset
                    }
                    st.session_state.evaluation_data.append(evaluation_item)

            except Exception as e:
                st.error(f"⚠️ Error processing question: {str(e)}")

    except Exception as e:
        st.error(f"⚠️ An error occurred during setup: {str(e)}")

# Instructions
with st.expander("ℹ️ How to use RAGAS evaluation"):
    st.markdown("""
    1. **Upload your documents** and start asking questions
    2. **Enable RAGAS Evaluation** in the sidebar
    3. **Ask multiple questions** to build evaluation data
    4. **Optionally provide ground truth answers** for more comprehensive evaluation
    5. **Click "Run RAGAS Evaluation"** to assess your RAG system performance
    6. **Review the metrics** to understand system quality:
        - Higher scores (closer to 1.0) indicate better performance
        - Focus on improving low-scoring metrics
    """)

# Required packages note
st.markdown("---")
st.caption("📦 Required packages: `pip install ragas datasets`")
st.caption("🔑 Make sure your .env file contains Azure OpenAI credentials:")
st.code("""
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
""", language="bash")