"""
Core FAQ Bot Module for Jupiter Help Centre
Integrates LLM capabilities with semantic search for intelligent responses
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
import time
from datetime import datetime

# LLM imports
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_community.llms import OpenAI as LangchainOpenAI
    from langchain_community.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Local imports
from embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

class JupiterFAQBot:
    """Main FAQ Bot class with LLM integration"""
    
    def __init__(self, 
                 embedding_manager: EmbeddingManager,
                 model_type: str = 'openai',
                 model_name: str = 'gpt-3.5-turbo',
                 api_key: str = None,
                 language: str = 'en'):
        
        self.embedding_manager = embedding_manager
        self.model_type = model_type
        self.model_name = model_name
        self.language = language
        self.conversation_history = []
        
        # Initialize LLM
        self.llm = self._initialize_llm(api_key)
        
        # Response templates
        self.templates = self._load_templates()
        
        logger.info(f"Initialized JupiterFAQBot with {model_type} model")
    
    def _initialize_llm(self, api_key: str = None):
        """Initialize the language model"""
        if self.model_type == 'openai':
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI not available, using fallback mode")
                return None
            
            # Get API key from environment or parameter
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found, using fallback mode")
                return None
            
            try:
                client = OpenAI(api_key=api_key)
                # Test the connection
                client.models.list()
                return client
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                return None
        
        elif self.model_type == 'langchain':
            if not LANGCHAIN_AVAILABLE:
                logger.warning("LangChain not available, using fallback mode")
                return None
            
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found, using fallback mode")
                return None
            
            try:
                return ChatOpenAI(
                    model_name=self.model_name,
                    temperature=0.7,
                    openai_api_key=api_key
                )
            except Exception as e:
                logger.error(f"Failed to initialize LangChain: {e}")
                return None
        
        elif self.model_type == 'gemini':
            if not GEMINI_AVAILABLE:
                logger.warning("Gemini not available, using fallback mode")
                return None
            
            # Get API key from environment or parameter
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.warning("Gemini API key not found, using fallback mode")
                return None
            
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                return model
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                return None
        
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using fallback mode")
            return None
    
    def _load_templates(self) -> Dict:
        """Load response templates for different languages"""
        templates = {
            'en': {
                'greeting': "Hello! I'm Jupiter's FAQ assistant. How can I help you today?",
                'no_answer': "I'm sorry, I couldn't find a specific answer to your question. Could you please rephrase it or ask something else?",
                'confidence_low': "I found some information that might be related, but I'm not entirely sure it answers your question:",
                'suggestions': "You might also be interested in:",
                'error': "I'm having trouble processing your request right now. Please try again in a moment.",
                'goodbye': "Thank you for using Jupiter's FAQ assistant. Have a great day!"
            },
            'hi': {
                'greeting': "नमस्ते! मैं Jupiter का FAQ सहायक हूं। आज मैं आपकी कैसे मदद कर सकता हूं?",
                'no_answer': "माफ़ करें, मुझे आपके सवाल का सटीक जवाब नहीं मिला। क्या आप इसे दूसरे तरीके से पूछ सकते हैं?",
                'confidence_low': "मुझे कुछ जानकारी मिली है जो संबंधित हो सकती है, लेकिन मुझे पूरा यकीन नहीं है:",
                'suggestions': "आप इनमें भी रुचि ले सकते हैं:",
                'error': "अभी आपके अनुरोध को संसाधित करने में समस्या आ रही है। कृपया कुछ देर बाद फिर से कोशिश करें।",
                'goodbye': "Jupiter के FAQ सहायक का उपयोग करने के लिए धन्यवाद। आपका दिन शुभ हो!"
            },
            'hinglish': {
                'greeting': "Hello! मैं Jupiter का FAQ assistant हूं। आज मैं आपकी कैसे help कर सकता हूं?",
                'no_answer': "Sorry, मुझे आपके question का exact answer नहीं मिला। क्या आप इसे differently पूछ सकते हैं?",
                'confidence_low': "मुझे कुछ information मिली है जो related हो सकती है, लेकिन मुझे पूरा confidence नहीं है:",
                'suggestions': "आप इनमें भी interest ले सकते हैं:",
                'error': "Right now आपके request को process करने में problem आ रही है। Please कुछ देर बाद try करें।",
                'goodbye': "Jupiter के FAQ assistant का use करने के लिए thank you। Have a great day!"
            }
        }
        return templates
    
    def _get_template(self, key: str) -> str:
        """Get template text for current language"""
        return self.templates.get(self.language, self.templates['en']).get(key, "")
    
    def _create_prompt(self, query: str, context: List[Dict]) -> str:
        """Create prompt for LLM based on query and context"""
        if not context:
            return self._get_template('no_answer')
        
        # Build context from search results
        context_text = ""
        for i, result in enumerate(context[:3], 1):
            context_text += f"{i}. Question: {result['question']}\n"
            context_text += f"   Answer: {result['answer']}\n\n"
        
        # Create prompt based on language
        if self.language == 'en':
            prompt = f"""You are Jupiter's helpful FAQ assistant. Based on the following context, provide a friendly and accurate answer to the user's question.

Context:
{context_text}

User Question: {query}

Instructions:
- Provide a clear, helpful, and conversational answer
- Use the context information but rephrase it naturally
- If the context doesn't fully answer the question, say so politely
- Keep the response concise but informative
- Be friendly and professional

Answer:"""
        
        elif self.language == 'hi':
            prompt = f"""आप Jupiter के मददगार FAQ सहायक हैं। निम्नलिखित संदर्भ के आधार पर, उपयोगकर्ता के प्रश्न का मित्रवत और सटीक उत्तर दें।

संदर्भ:
{context_text}

उपयोगकर्ता का प्रश्न: {query}

निर्देश:
- स्पष्ट, सहायक और बातचीत के अंदाज में उत्तर दें
- संदर्भ जानकारी का उपयोग करें लेकिन इसे स्वाभाविक रूप से पुनः व्यक्त करें
- यदि संदर्भ प्रश्न का पूरा उत्तर नहीं देता, तो विनम्रतापूर्वक कहें
- प्रतिक्रिया संक्षिप्त लेकिन जानकारीपूर्ण रखें
- मित्रवत और पेशेवर रहें

उत्तर:"""
        
        else:  # hinglish
            prompt = f"""आप Jupiter के helpful FAQ assistant हैं। निम्नलिखित context के आधार पर, user के question का friendly और accurate answer दें।

Context:
{context_text}

User Question: {query}

Instructions:
- Clear, helpful और conversational answer दें
- Context information का use करें लेकिन इसे naturally rephrase करें
- अगर context question का पूरा answer नहीं देता, तो politely कहें
- Response concise लेकिन informative रखें
- Friendly और professional रहें

Answer:"""
        
        return prompt
    
    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using LLM"""
        if not self.llm:
            return self._get_template('error')
        
        try:
            if self.model_type == 'openai':
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            
            elif self.model_type == 'langchain':
                response = self.llm.predict(prompt)
                return response.strip()
            
            elif self.model_type == 'gemini':
                response = self.llm.generate_content(prompt)
                return response.text.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._get_template('error')
    
    def _get_suggestions(self, query: str) -> List[str]:
        """Get suggested questions based on user query"""
        try:
            suggestions = self.embedding_manager.get_similar_questions(query, k=3)
            return suggestions
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []
    
    def _format_response(self, answer: str, suggestions: List[str], confidence: float) -> Dict:
        """Format the complete response"""
        response = {
            'answer': answer,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'suggestions': suggestions
        }
        
        # Add confidence warning if needed
        if confidence < 0.7:
            response['warning'] = self._get_template('confidence_low')
        
        return response
    
    def ask(self, query: str, use_llm: bool = True, search_method: str = 'hybrid') -> Dict:
        """Main method to ask a question to the bot"""
        logger.info(f"Processing query: {query}")
        
        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'timestamp': datetime.now().isoformat()
        })
        
        # Perform semantic search
        search_results = self.embedding_manager.semantic_search(
            query, k=5, method=search_method, threshold=0.3
        )
        
        if not search_results:
            return {
                'answer': self._get_template('no_answer'),
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'suggestions': [],
                'search_results': []
            }
        
        # Get best match confidence
        best_confidence = search_results[0]['similarity_score']
        
        # Generate response
        if use_llm and self.llm and best_confidence > 0.3:
            # Use LLM to generate conversational response
            prompt = self._create_prompt(query, search_results)
            answer = self._generate_llm_response(prompt)
        else:
            # Use direct answer from search results
            answer = search_results[0]['answer']
            if best_confidence < 0.5:
                answer = f"{self._get_template('confidence_low')}\n\n{answer}"
        
        # Get suggestions
        suggestions = self._get_suggestions(query)
        
        # Format response
        response = self._format_response(answer, suggestions, best_confidence)
        response['search_results'] = search_results
        
        return response
    
    def get_greeting(self) -> str:
        """Get greeting message"""
        return self._get_template('greeting')
    
    def get_goodbye(self) -> str:
        """Get goodbye message"""
        return self._get_template('goodbye')
    
    def change_language(self, language: str):
        """Change bot language"""
        if language in self.templates:
            self.language = language
            logger.info(f"Changed bot language to: {language}")
        else:
            logger.warning(f"Unsupported language: {language}")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Cleared conversation history")
    
    def get_bot_stats(self) -> Dict:
        """Get bot statistics"""
        embedding_stats = self.embedding_manager.get_embedding_stats()
        
        stats = {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'language': self.language,
            'llm_available': self.llm is not None,
            'conversation_count': len(self.conversation_history),
            'embedding_stats': embedding_stats
        }
        
        return stats

class FallbackBot:
    """Simple fallback bot when LLM is not available"""
    
    def __init__(self, embedding_manager: EmbeddingManager, language: str = 'en'):
        self.embedding_manager = embedding_manager
        self.language = language
        self.templates = {
            'en': {
                'greeting': "Hello! I'm Jupiter's FAQ assistant. How can I help you today?",
                'no_answer': "I'm sorry, I couldn't find a specific answer to your question.",
                'confidence_low': "I found some information that might be related:",
                'suggestions': "You might also be interested in:"
            },
            'hi': {
                'greeting': "नमस्ते! मैं Jupiter का FAQ सहायक हूं।",
                'no_answer': "माफ़ करें, मुझे आपके सवाल का जवाब नहीं मिला।",
                'confidence_low': "मुझे कुछ जानकारी मिली है:",
                'suggestions': "आप इनमें भी रुचि ले सकते हैं:"
            }
        }
    
    def ask(self, query: str) -> Dict:
        """Simple search-based response"""
        results = self.embedding_manager.semantic_search(query, k=3, threshold=0.3)
        
        if not results:
            return {
                'answer': self.templates[self.language]['no_answer'],
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'suggestions': []
            }
        
        best_result = results[0]
        answer = best_result['answer']
        
        if best_result['similarity_score'] < 0.5:
            answer = f"{self.templates[self.language]['confidence_low']}\n\n{answer}"
        
        return {
            'answer': answer,
            'confidence': best_result['similarity_score'],
            'timestamp': datetime.now().isoformat(),
            'suggestions': [r['question'] for r in results[1:3]]
        }

def main():
    """Test the FAQ bot"""
    # Load preprocessed FAQs
    try:
        with open('data/preprocessed_faqs.json', 'r', encoding='utf-8') as f:
            faqs = json.load(f)
    except FileNotFoundError:
        print("Preprocessed FAQ data not found. Please run preprocessor.py first.")
        return
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    embedding_manager.initialize_from_faqs(faqs)
    
    # Initialize bot
    bot = JupiterFAQBot(embedding_manager, model_type='openai')
    
    # Test queries
    test_queries = [
        "How do I complete KYC verification?",
        "What payment methods does Jupiter accept?",
        "How can I earn rewards?",
        "What are the transaction limits?",
        "How do I reset my password?"
    ]
    
    print("Testing Jupiter FAQ Bot...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = bot.ask(query)
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']:.3f}")
        if response['suggestions']:
            print(f"Suggestions: {response['suggestions']}")

if __name__ == "__main__":
    main() 