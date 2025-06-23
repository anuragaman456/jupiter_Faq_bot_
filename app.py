"""
Jupiter FAQ Bot - Streamlit Web Application
Beautiful and interactive web interface for the FAQ bot with Gemini AI integration
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os
import sys
import google.generativeai as genai

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from scraper import JupiterFAQScraper
from preprocessor import FAQPreprocessor
from embeddings import EmbeddingManager
from bot import JupiterFAQBot, FallbackBot
from evaluation import FAQBotEvaluator

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCree6LpEbBjhQOIyldr6edl9F-l4fM_ys"
genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(
    page_title="Jupiter FAQ Bot 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful modern CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Message Bubbles */
    .message-bubble {
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        position: relative;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        border-bottom-right-radius: 5px;
    }
    
    .bot-bubble {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
        border-bottom-left-radius: 5px;
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .confidence-high {
        background: rgba(76, 175, 80, 0.9);
        color: white;
    }
    
    .confidence-medium {
        background: rgba(255, 152, 0, 0.9);
        color: white;
    }
    
    .confidence-low {
        background: rgba(244, 67, 54, 0.9);
        color: white;
    }
    
    /* Suggestion Chips */
    .suggestion-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .suggestion-chip {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .suggestion-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Input Area */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metrics Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading Animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots::after {
        content: '';
        animation: dots 1.5s steps(5, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'bot' not in st.session_state:
    st.session_state.bot = None

if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None

if 'current_language' not in st.session_state:
    st.session_state.current_language = 'en'

if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

if 'suggestion_clicked' not in st.session_state:
    st.session_state.suggestion_clicked = None

def initialize_gemini():
    """Initialize Gemini model"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        st.session_state.gemini_model = model
        return True
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        return False

def initialize_bot():
    """Initialize the FAQ bot"""
    try:
        # Load preprocessed FAQs
        with open('data/preprocessed_faqs.json', 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager()
        embedding_manager.initialize_from_faqs(faqs)
        
        # Initialize bot with Gemini API key
        bot = JupiterFAQBot(embedding_manager, model_type='gemini', api_key=GEMINI_API_KEY)
        
        st.session_state.embedding_manager = embedding_manager
        st.session_state.bot = bot
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize bot: {e}")
        return False

def get_confidence_color(confidence):
    """Get color for confidence badge"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_chat_message(message, is_user=True, confidence=None, suggestions=None, message_id=None):
    """Display a beautiful chat message"""
    if is_user:
        st.markdown(f"""
        <div class="message-bubble user-bubble">
            <strong>You</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display the main message first
        st.markdown(f"""
        <div class="message-bubble bot-bubble">
            <strong>🤖 Jupiter Bot</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
        
        # Display confidence badge separately if available
        if confidence is not None:
            try:
                conf_value = float(confidence)
                confidence_class = get_confidence_color(conf_value)
                st.markdown(f"""
                <div class="confidence-badge {confidence_class}">
                    Confidence: {conf_value:.1%}
                </div>
                """, unsafe_allow_html=True)
            except (ValueError, TypeError):
                # If conversion fails, don't show confidence badge
                pass
        
        # Display suggestions as Streamlit buttons instead of HTML
        if suggestions:
            st.markdown("**💡 Related Questions:**")
            cols = st.columns(len(suggestions[:3]))
            for i, suggestion in enumerate(suggestions[:3]):
                with cols[i]:
                    # Use message_id for unique keys, fallback to timestamp if available
                    unique_key = f"chat_sugg_{message_id}_{i}" if message_id else f"chat_sugg_{i}_{suggestion[:20]}"
                    if st.button(suggestion, key=unique_key):
                        st.session_state.suggestion_clicked = suggestion
                        st.rerun()

def main():
    """Main application"""
    
    # Beautiful Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Jupiter FAQ Bot</h1>
        <p>Your intelligent assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Language Selection
        language = st.selectbox(
            "🌍 Language",
            ["English", "Hindi", "Hinglish"],
            index=0
        )
        
        # Model Selection
        model_type = st.selectbox(
            "🤖 AI Model",
            ["Gemini 2.0 Flash", "OpenAI GPT"],
            index=0
        )
        
        # Search Method
        search_method = st.selectbox(
            "🔍 Search Method",
            ["Hybrid", "Semantic", "Keyword"],
            index=0
        )
        
        # Confidence Threshold
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Statistics")
        if st.session_state.chat_history:
            st.metric("Total Messages", len(st.session_state.chat_history))
            st.metric("Average Confidence", f"{sum([msg.get('confidence', 0) for msg in st.session_state.chat_history if not msg.get('is_user', True)]) / max(1, len([msg for msg in st.session_state.chat_history if not msg.get('is_user', True)])):.1%}")
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Initialize bot if not already done
    if st.session_state.bot is None:
        with st.spinner("🚀 Initializing Jupiter Bot..."):
            if initialize_bot() and initialize_gemini():
                st.success("✅ Bot initialized successfully!")
            else:
                st.error("❌ Failed to initialize bot")
                return
    
    # Main Chat Interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        # Create unique message ID from timestamp
        message_id = message.get('timestamp', i)
        if hasattr(message_id, 'strftime'):
            message_id = message_id.strftime('%Y%m%d%H%M%S%f')
        else:
            message_id = str(i)
            
        display_chat_message(
            message['text'],
            message.get('is_user', True),
            message.get('confidence'),
            message.get('suggestions'),
            message_id
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat Input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "💬 Ask me anything about Jupiter...",
            placeholder="e.g., How do I add money to my Jupiter account?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("🚀 Send", type="primary")
    
    # Handle user input
    if (user_input and send_button) or (user_input and st.session_state.get('auto_send', False)):
        if user_input.strip():
            # Add user message to history
            st.session_state.chat_history.append({
                'text': user_input,
                'is_user': True,
                'timestamp': datetime.now()
            })
            
            # Get bot response
            with st.spinner("🤖 Thinking..."):
                try:
                    # Use Gemini for response generation
                    if st.session_state.gemini_model:
                        # Get relevant context from embeddings
                        context = st.session_state.embedding_manager.semantic_search(user_input, k=3)
                        
                        # Create prompt for Gemini
                        context_text = ""
                        for i, result in enumerate(context, 1):
                            context_text += f"{i}. Q: {result['question']}\nA: {result['answer']}\n\n"
                        
                        prompt = f"""You are Jupiter's helpful FAQ assistant. Based on the following context, provide a friendly and accurate answer to the user's question.

Context:
{context_text}

User Question: {user_input}

Instructions:
- Provide a clear, helpful, and conversational answer
- Use the context information but rephrase it naturally
- If the context doesn't fully answer the question, say so politely
- Keep the response concise but informative (2-3 sentences)
- Be friendly and professional
- Focus on Jupiter-specific information
- If you don't have enough context, suggest related topics

Answer:"""
                        
                        response = st.session_state.gemini_model.generate_content(prompt)
                        bot_response = response.text
                        
                        # Calculate confidence based on context relevance
                        if context:
                            # Use the highest similarity score from search results
                            max_similarity = max([result.get('similarity_score', 0) for result in context])
                            confidence = min(0.95, max_similarity * 1.2)  # Boost confidence slightly
                        else:
                            confidence = 0.3
                        
                        # Generate suggestions
                        suggestions = [
                            "How to complete KYC verification?",
                            "What are the transaction limits?",
                            "How to earn rewards on Jupiter?",
                            "What documents are required?"
                        ]
                        
                    else:
                        # Fallback to regular bot
                        response = st.session_state.bot.ask(user_input)
                        bot_response = response['answer']
                        confidence = response.get('confidence', 0.5)
                        suggestions = response.get('suggestions', [])
                    
                    # Add bot response to history
                    st.session_state.chat_history.append({
                        'text': bot_response,
                        'is_user': False,
                        'confidence': confidence,
                        'suggestions': suggestions,
                        'timestamp': datetime.now()
                    })
                    
                    # Clear auto_send flag
                    st.session_state.auto_send = False
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Quick Suggestions
    st.markdown("### 💡 Quick Suggestions")
    suggestions = [
        "How do I add money to my Jupiter account?",
        "What are Jupiter's security features?",
        "How to contact Jupiter customer support?",
        "What are the transaction limits?",
        "How to set up UPI on Jupiter?"
    ]
    
    cols = st.columns(5)
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"sugg_{i}"):
                st.session_state.suggestion_clicked = suggestion
                st.rerun()
    
    # Handle suggestion clicks
    if st.session_state.suggestion_clicked:
        # Add the suggested question to chat history
        st.session_state.chat_history.append({
            'text': st.session_state.suggestion_clicked,
            'is_user': True,
            'timestamp': datetime.now()
        })
        
        # Get bot response for the suggestion
        with st.spinner("🤖 Thinking..."):
            try:
                if st.session_state.gemini_model:
                    context = st.session_state.embedding_manager.semantic_search(st.session_state.suggestion_clicked, k=3)
                    
                    context_text = ""
                    for i, result in enumerate(context, 1):
                        context_text += f"{i}. Q: {result['question']}\nA: {result['answer']}\n\n"
                    
                    prompt = f"""You are Jupiter's helpful FAQ assistant. Based on the following context, provide a friendly and accurate answer to the user's question.

Context:
{context_text}

User Question: {st.session_state.suggestion_clicked}

Instructions:
- Provide a clear, helpful, and conversational answer
- Use the context information but rephrase it naturally
- If the context doesn't fully answer the question, say so politely
- Keep the response concise but informative (2-3 sentences)
- Be friendly and professional
- Focus on Jupiter-specific information
- If you don't have enough context, suggest related topics

Answer:"""
                    
                    response = st.session_state.gemini_model.generate_content(prompt)
                    bot_response = response.text
                    
                    # Calculate confidence based on context relevance
                    if context:
                        # Use the highest similarity score from search results
                        max_similarity = max([result.get('similarity_score', 0) for result in context])
                        confidence = min(0.95, max_similarity * 1.2)  # Boost confidence slightly
                    else:
                        confidence = 0.3
                    
                    suggestions = [
                        "How to complete KYC verification?",
                        "What are the transaction limits?",
                        "How to earn rewards on Jupiter?",
                        "What documents are required?"
                    ]
                else:
                    response = st.session_state.bot.ask(st.session_state.suggestion_clicked)
                    bot_response = response['answer']
                    confidence = response.get('confidence', 0.5)
                    suggestions = response.get('suggestions', [])
                
                st.session_state.chat_history.append({
                    'text': bot_response,
                    'is_user': False,
                    'confidence': confidence,
                    'suggestions': suggestions,
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        # Clear the suggestion
        st.session_state.suggestion_clicked = None
        st.rerun()

if __name__ == "__main__":
    main() 