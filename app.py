"""
Market Research Assistant
A Streamlit application that generates market research reports using Wikipedia data.
Designed for business analysts.
"""

import os
import json
import warnings
from typing import Tuple, List, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

def get_openai_api_key():
    """
    Retrieve OpenAI API key from multiple sources in order of priority:
    1. Streamlit secrets (for deployment)
    2. Environment variable (loaded via dotenv or system env)
    3. Sidebar input (for local testing)
    
    Returns:
        Tuple of (api_key, source) where source indicates where the key came from
    """
    # Try Streamlit secrets first (with error handling)
    try:
        # Check if secrets attribute exists and has the key
        if hasattr(st, 'secrets'):
            # Use getattr with default to avoid triggering FileNotFoundError
            secrets_dict = getattr(st, 'secrets', {})
            if secrets_dict and "OPENAI_API_KEY" in secrets_dict:
                key = secrets_dict["OPENAI_API_KEY"]
                if key:
                    return key, "Streamlit Secrets"
    except (FileNotFoundError, AttributeError, KeyError):
        # Secrets file doesn't exist or key not found, continue to next option
        pass
    except Exception:
        # Any other error with secrets, continue to next option
        pass
    
    # Try environment variable (loaded via dotenv or system env)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key, "Environment Variable (.env or system)"
    
    # No key found from secrets or env
    return None, None


def initialize_llm(api_key: str):
    """Initialize the ChatOpenAI LLM with the provided API key.
    
    GPT-5 series are reasoning models and require the Responses API.
    They use 'reasoning_effort' instead of 'temperature'.
    """
    return ChatOpenAI(
        model="gpt-5-mini-2025-08-07",
        reasoning_effort="low",
        use_responses_api=True,
        openai_api_key=api_key
    )


def extract_text_from_content(content) -> str:
    """
    Extract plain text from an LLM response content.
    
    The Responses API returns content as a list of blocks:
      [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}]
    whereas the Chat Completions API returns a plain string.
    This helper handles both formats.
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    
    return str(content)


# ============================================================================
# Q1: USER INPUT VALIDATION (LLM-Powered)
# ============================================================================

def validate_industry_input(user_input: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Q1: Use an LLM to intelligently validate and classify the user's industry input.

    The LLM evaluates the input and returns a JSON object with:
      - status: "valid", "ambiguous", or "invalid"
      - message: A short feedback message
      - clarification_options: A list of 3 specific options if status is "ambiguous"

    Args:
        user_input: The raw industry string entered by the user
        llm: An initialized ChatOpenAI instance

    Returns:
        A dict with keys: status, message, clarification_options
    """
    # Fast-fail for empty input (no need to call the LLM)
    if not user_input or not user_input.strip():
        return {
            "status": "invalid",
            "message": "⚠️ Please enter an industry name to generate a report.",
            "clarification_options": []
        }

    # ── LLM Validation Chain ──────────────────────────────────────────────
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert industry-classification assistant.
Your job is to evaluate a user's text input and decide whether it clearly
refers to a specific, searchable industry or market sector.

Return ONLY a valid JSON object (no markdown, no extra text) with these keys:

{{
  "status": "valid" | "ambiguous" | "invalid",
  "message": "<short feedback message>",
  "clarification_options": ["<question 1>", "<question 2>", "<question 3>"]
}}

Classification rules:
• "valid"   – The input clearly identifies one specific industry or market
              sector that can be researched (e.g., "Electric Vehicles",
              "Semiconductor Industry", "Cloud Computing").
              Set clarification_options to an empty list [].

• "ambiguous" – The input is too vague, is an acronym with multiple
                meanings, or could refer to several different industries
                (e.g., "AI", "pet", "chip", "green").
                Provide exactly 3 clarification_options.
                IMPORTANT: Each option MUST be a full clarifying question,
                NOT a short keyword. Start each question with a phrase like:
                - "Are you referring to..."
                - "Do you mean..."
                - "Are you interested in..."
                Example for input "pet":
                [
                  "Are you referring to pets as animals kept for companionship?",
                  "Do you mean \"PET\" in a scientific or medical context, such as Positron Emission Tomography?",
                  "Are you interested in a specific aspect of pets, like care, types, or behavior?"
                ]

• "invalid"  – The input is gibberish, offensive, a greeting, a question,
               or clearly not related to any industry/market sector
               (e.g., "hello", "asdfgh", "what is this?").
               Set clarification_options to an empty list []."""),
        ("human", "User input: {user_input}")
    ])

    chain = validation_prompt | llm

    try:
        response = chain.invoke({"user_input": user_input.strip()})
        raw_response = extract_text_from_content(response.content)

        # Strip possible markdown code fences the LLM might add
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]  # remove opening fence line
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        result = json.loads(cleaned)

        # Ensure expected keys exist with safe defaults
        return {
            "status": result.get("status", "invalid"),
            "message": result.get("message", ""),
            "clarification_options": result.get("clarification_options", [])
        }

    except (json.JSONDecodeError, Exception) as e:
        # If the LLM response can't be parsed, treat the input as valid
        # so the pipeline can still proceed (fail-open for usability).
        return {
            "status": "valid",
            "message": f"⚠️ Validation returned an unexpected format — proceeding anyway. ({e})",
            "clarification_options": []
        }


# ============================================================================
# Q2: WIKIPEDIA DATA RETRIEVAL
# ============================================================================

def retrieve_wikipedia_documents(industry: str, top_k: int = 5) -> Tuple[List[Document], List[str]]:
    """
    Q2: Retrieve relevant Wikipedia documents for the given industry.
    
    Args:
        industry: The industry name to search for
        top_k: Number of top documents to retrieve (default: 5)
        
    Returns:
        Tuple of (list of Document objects, list of URLs)
    """
    try:
        # Initialize Wikipedia Retriever
        retriever = WikipediaRetriever(
            lang="en",
            top_k_results=top_k,
            doc_content_chars_max=2000  # Limit content per document for efficiency
        )
        
        # Retrieve documents using invoke() method (LangChain v0.1+)
        documents = retriever.invoke(industry)
        
        # Extract URLs from documents
        urls = []
        for doc in documents:
            # WikipediaRetriever stores metadata with 'source' key containing the URL
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                # Try different possible keys for URL
                if 'source' in metadata:
                    urls.append(metadata['source'])
                elif 'url' in metadata:
                    urls.append(metadata['url'])
                elif 'wikipedia_url' in metadata:
                    urls.append(metadata['wikipedia_url'])
                else:
                    # If no URL found, try to construct from title
                    if 'title' in metadata:
                        title = metadata['title'].replace(' ', '_')
                        urls.append(f"https://en.wikipedia.org/wiki/{title}")
        
        return documents, urls
    
    except Exception as e:
        st.error(f"❌ Error retrieving Wikipedia data: {str(e)}")
        return [], []


# ============================================================================
# Q3: REPORT GENERATION
# ============================================================================

def generate_market_research_report(
    llm: ChatOpenAI,
    documents: List[Document],
    industry: str
) -> str:
    """
    Q3: Generate a market research report using the retrieved Wikipedia documents.
    
    The report is:
    - Geared towards business analysts
    - Less than 500 words
    - Based strictly on the provided Wikipedia context
    
    Args:
        llm: Initialized ChatOpenAI instance
        documents: List of retrieved Wikipedia documents
        industry: The industry name
        
    Returns:
        Generated report as a string
    """
    # Combine all document contents into context
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a professional market research analyst. Your task is to create a concise 
market research report based STRICTLY on the provided Wikipedia context. 

IMPORTANT CONSTRAINTS:
- The report must be LESS THAN 500 WORDS
- Base your report ONLY on the information provided in the context
- Do NOT include any information not found in the context
- Write in a professional tone suitable for business analysts
- Focus on market trends, industry overview, key players, and business insights
- Structure the report with clear sections if appropriate"""),
        
        ("human", """Generate a market research report for the following industry: {industry}

Wikipedia Context:
{context}

Please provide a comprehensive but concise market research report (under 500 words) 
based strictly on the information above.""")
    ])
    
    # Format the prompt
    formatted_prompt = prompt_template.format_messages(
        industry=industry,
        context=context
    )
    
    try:
        # Generate the report
        response = llm.invoke(formatted_prompt)
        report = extract_text_from_content(response.content)
        
        return report
    
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        return ""


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Market Research Assistant",
        page_icon="📊",
        layout="wide"
    )
    
    # Title and description
    st.title("📊 Market Research Assistant")
    st.markdown("Generate professional market research reports using Wikipedia data.")
    st.divider()
    
    # Sidebar for API key configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get API key from secrets/env
        api_key_from_config, key_source = get_openai_api_key()
        
        # Display API key status
        st.subheader("🔑 API Key Configuration")
        
        # Initialize session state for API key input if not exists
        if "api_key_input_value" not in st.session_state:
            st.session_state.api_key_input_value = ""
        
        # Initialize session state for saved API key
        if "saved_api_key" not in st.session_state:
            st.session_state.saved_api_key = ""
        
        # Show status if key is loaded from config
        if api_key_from_config:
            masked_key = api_key_from_config[:7] + "..." + api_key_from_config[-4:] if len(api_key_from_config) > 11 else "***"
            st.success(f"✅ API Key loaded from: **{key_source}**")
            st.caption(f"Preview: `{masked_key}`")
            st.caption("💡 You can edit the API key below")
        else:
            st.warning("⚠️ No API Key found in config")
            st.caption("Please enter your API key below")
        
        # Always show input field (allow editing even if key is loaded from config)
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter or edit your OpenAI API key. Click 'Save Key' to save it.",
            placeholder="sk-...",
            value=st.session_state.saved_api_key if st.session_state.saved_api_key else "",
            key="api_key_input"
        )
        
        # Save button
        col1, col2 = st.columns([1, 1])
        with col1:
            save_button = st.button("💾 Save Key", use_container_width=True, type="primary")
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        # Handle save button
        if save_button:
            if api_key_input and api_key_input.strip():
                st.session_state.saved_api_key = api_key_input.strip()
                st.success("✅ API Key saved successfully!")
                st.rerun()
            else:
                st.error("❌ Please enter an API key before saving.")
        
        # Handle clear button
        if clear_button:
            st.session_state.saved_api_key = ""
            st.rerun()
        
        # Determine which key to use: saved key > sidebar input > config
        if st.session_state.saved_api_key:
            # Use saved key
            api_key = st.session_state.saved_api_key
            if api_key_from_config and api_key != api_key_from_config:
                st.info("ℹ️ Using saved API key (overriding config)")
        elif api_key_input and api_key_input.strip():
            # User entered/edited key in sidebar - use that
            api_key = api_key_input.strip()
            if api_key_from_config and api_key != api_key_from_config:
                st.info("ℹ️ Using API key from input (overriding config)")
        elif api_key_from_config:
            # No sidebar input, use config key
            api_key = api_key_from_config
        else:
            # No key at all
            api_key = None
        
        # Show warning if no API key (but don't stop the app)
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key above and click 'Save Key' to generate reports.")
        
        st.divider()
    
    # Main input section
    st.header("Industry Input")
    industry = st.text_input(
        "Enter Industry Name",
        placeholder="e.g., Electric Vehicles, Coffee Shop, Semiconductors",
        help="Enter the industry you want to research"
    )
    
    generate_button = st.button("Generate Report", type="primary", use_container_width=True)
    
    st.divider()
    
    # Process when generate button is clicked
    if generate_button:
        # ── Pre-flight check: API key must exist before any LLM call ──
        if not api_key:
            st.error("❌ OpenAI API key is required. Please configure it in the sidebar.")
            st.stop()

        # Initialize LLM (needed for both validation and report generation)
        try:
            llm = initialize_llm(api_key)
        except Exception as e:
            st.error(f"❌ Error initializing LLM: {str(e)}")
            st.stop()

        # ── Q1: Intelligent Input Validation ──────────────────────────
        with st.spinner("🧠 Analyzing your input..."):
            validation = validate_industry_input(industry, llm)

        if validation["status"] == "invalid":
            st.error(f"🚫 {validation['message']}")
            st.stop()

        elif validation["status"] == "ambiguous":
            st.subheader("I need a bit more detail to avoid pulling the wrong pages")
            questions = validation.get("clarification_options", [])
            questions_text = "\n".join(
                f"{idx}. {q}" for idx, q in enumerate(questions, 1)
            )
            st.info(questions_text)
            st.stop()

        # status == "valid" — proceed
        if validation["message"]:
            st.success(f"✅ {validation['message']}")

        # (LLM already initialized above, skip duplicate init)

        # Q2: Retrieve Wikipedia documents
        with st.spinner("🔍 Retrieving relevant Wikipedia pages..."):
            documents, urls = retrieve_wikipedia_documents(industry, top_k=5)
        
        if not documents:
            st.error("❌ No Wikipedia documents found. Please try a different industry name.")
            st.stop()
        
        # Display retrieved URLs
        st.header("📚 Source References")
        st.markdown("**Top 5 Most Relevant Wikipedia Pages:**")
        
        if urls:
            for i, url in enumerate(urls[:5], 1):
                st.markdown(f"{i}. [{url}]({url})")
        else:
            st.info("URLs not available in document metadata.")
        
        st.divider()
        
        # Q3: Generate report
        with st.spinner("✍️ Generating market research report..."):
            report = generate_market_research_report(llm, documents, industry)
        
        if report:
            st.header("📄 Market Research Report")
            st.markdown(report)
            
            # Display word count
            word_count = len(report.split())
            st.caption(f"*Report length: {word_count} words*")
            
            if word_count > 500:
                st.warning(f"⚠️ Report exceeds 500 words ({word_count} words). Please review.")
        else:
            st.error("❌ Failed to generate report. Please try again.")


if __name__ == "__main__":
    main()

