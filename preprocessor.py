"""
Data Preprocessing Module for Jupiter FAQ Bot
Handles cleaning, normalization, deduplication, and categorization of FAQ data
"""

import pandas as pd
import numpy as np
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import logging
from textblob import TextBlob
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class FAQPreprocessor:
    """Preprocesses FAQ data for the bot"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9][\d]{0,15}', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Lemmatize and remove stop words
        normalized_tokens = []
        for token in tokens:
            if token.lower() not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token.lower())
                normalized_tokens.append(lemmatized)
        
        return ' '.join(normalized_tokens)
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key terms from text"""
        normalized = self.normalize_text(text)
        tokens = normalized.split()
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for token in tokens:
            if len(token) > 3:  # Filter out short words
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def categorize_faq(self, question: str, answer: str) -> str:
        """Categorize FAQ based on content analysis"""
        text = f"{question} {answer}".lower()
        
        # Enhanced category mapping
        categories = {
            'kyc': ['kyc', 'verification', 'identity', 'document', 'pan', 'aadhaar', 'biometric', 'e-kyc'],
            'payments': ['payment', 'upi', 'card', 'transaction', 'transfer', 'bank', 'neft', 'imps', 'rtgs'],
            'rewards': ['reward', 'cashback', 'points', 'bonus', 'offer', 'discount', 'cashback'],
            'limits': ['limit', 'maximum', 'minimum', 'daily', 'monthly', 'yearly', 'threshold'],
            'security': ['security', 'password', 'pin', 'otp', 'fraud', 'secure', 'authentication'],
            'account': ['account', 'profile', 'settings', 'preferences', 'personal'],
            'cards': ['card', 'debit', 'credit', 'virtual', 'physical', 'cardless'],
            'investments': ['investment', 'mutual fund', 'stocks', 'portfolio', 'wealth'],
            'support': ['support', 'help', 'contact', 'customer service', 'complaint']
        }
        
        # Calculate category scores
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                return best_category[0]
        
        return 'general'
    
    def detect_language(self, text: str) -> str:
        """Detect language of the text"""
        try:
            blob = TextBlob(text)
            return blob.detect_language()
        except:
            return 'en'  # Default to English
    
    def find_duplicates(self, faqs: List[Dict], similarity_threshold: float = 0.8) -> List[Tuple[int, int]]:
        """Find duplicate or very similar FAQs"""
        if len(faqs) < 2:
            return []
        
        # Prepare text for vectorization
        questions = [faq['question'] for faq in faqs]
        answers = [faq['answer'] for faq in faqs]
        
        # Combine question and answer for better similarity detection
        combined_texts = [f"{q} {a}" for q, a in zip(questions, answers)]
        
        # Vectorize
        try:
            tfidf_matrix = self.vectorizer.fit_transform(combined_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            duplicates = []
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] >= similarity_threshold:
                        duplicates.append((i, j))
            
            return duplicates
        except Exception as e:
            logger.warning(f"Error in duplicate detection: {e}")
            return []
    
    def remove_duplicates(self, faqs: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """Remove duplicate FAQs"""
        duplicates = self.find_duplicates(faqs, similarity_threshold)
        
        if not duplicates:
            return faqs
        
        # Mark duplicates for removal
        to_remove = set()
        for i, j in duplicates:
            # Keep the one with higher confidence or better quality
            if faqs[i].get('confidence', 0) >= faqs[j].get('confidence', 0):
                to_remove.add(j)
            else:
                to_remove.add(i)
        
        # Remove duplicates
        cleaned_faqs = [faq for idx, faq in enumerate(faqs) if idx not in to_remove]
        
        logger.info(f"Removed {len(to_remove)} duplicate FAQs")
        return cleaned_faqs
    
    def enhance_faqs(self, faqs: List[Dict]) -> List[Dict]:
        """Enhance FAQ data with additional features"""
        enhanced_faqs = []
        
        for faq in faqs:
            enhanced_faq = faq.copy()
            
            # Clean and normalize text
            enhanced_faq['question_clean'] = self.clean_text(faq['question'])
            enhanced_faq['answer_clean'] = self.clean_text(faq['answer'])
            
            # Normalize for search
            enhanced_faq['question_normalized'] = self.normalize_text(faq['question'])
            enhanced_faq['answer_normalized'] = self.normalize_text(faq['answer'])
            
            # Extract keywords
            enhanced_faq['keywords'] = self.extract_keywords(f"{faq['question']} {faq['answer']}")
            
            # Detect language
            enhanced_faq['language'] = self.detect_language(faq['question'])
            
            # Categorize
            enhanced_faq['category'] = self.categorize_faq(faq['question'], faq['answer'])
            
            # Calculate text statistics
            enhanced_faq['question_length'] = len(faq['question'])
            enhanced_faq['answer_length'] = len(faq['answer'])
            enhanced_faq['total_length'] = enhanced_faq['question_length'] + enhanced_faq['answer_length']
            
            # Calculate readability score (simple)
            enhanced_faq['readability_score'] = self.calculate_readability(faq['answer'])
            
            enhanced_faqs.append(enhanced_faq)
        
        return enhanced_faqs
    
    def calculate_readability(self, text: str) -> float:
        """Calculate simple readability score"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple Flesch Reading Ease approximation
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        return max(0, min(100, readability))
    
    def filter_quality_faqs(self, faqs: List[Dict], min_length: int = 20, max_length: int = 2000) -> List[Dict]:
        """Filter FAQs based on quality criteria"""
        quality_faqs = []
        
        for faq in faqs:
            question_len = len(faq.get('question', ''))
            answer_len = len(faq.get('answer', ''))
            
            # Check length constraints
            if question_len < 10 or answer_len < min_length:
                continue
            
            if question_len > 500 or answer_len > max_length:
                continue
            
            # Check for meaningful content
            if not faq.get('question', '').strip() or not faq.get('answer', '').strip():
                continue
            
            # Check confidence score
            if faq.get('confidence', 0) < 0.5:
                continue
            
            quality_faqs.append(faq)
        
        logger.info(f"Filtered {len(faqs)} FAQs to {len(quality_faqs)} quality FAQs")
        return quality_faqs
    
    def preprocess_faqs(self, faqs: List[Dict]) -> List[Dict]:
        """Complete preprocessing pipeline"""
        logger.info(f"Starting preprocessing of {len(faqs)} FAQs")
        
        # Step 1: Clean and enhance
        enhanced_faqs = self.enhance_faqs(faqs)
        
        # Step 2: Remove duplicates
        deduplicated_faqs = self.remove_duplicates(enhanced_faqs)
        
        # Step 3: Filter for quality
        quality_faqs = self.filter_quality_faqs(deduplicated_faqs)
        
        # Step 4: Sort by confidence and quality
        sorted_faqs = sorted(quality_faqs, key=lambda x: (
            x.get('confidence', 0),
            x.get('readability_score', 0),
            x.get('total_length', 0)
        ), reverse=True)
        
        logger.info(f"Preprocessing complete. Final dataset: {len(sorted_faqs)} FAQs")
        return sorted_faqs
    
    def save_preprocessed_data(self, faqs: List[Dict], filename: str = 'data/preprocessed_faqs.json'):
        """Save preprocessed FAQs"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(faqs, f, indent=2, ensure_ascii=False)
        
        # Save summary statistics
        df = pd.DataFrame(faqs)
        summary = {
            'total_faqs': len(faqs),
            'categories': df['category'].value_counts().to_dict(),
            'languages': df['language'].value_counts().to_dict(),
            'avg_confidence': df['confidence'].mean(),
            'avg_readability': df['readability_score'].mean(),
            'avg_length': df['total_length'].mean()
        }
        
        summary_filename = filename.replace('.json', '_summary.json')
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved preprocessed data to {filename}")
        logger.info(f"Summary: {summary}")
        
        return filename

def main():
    """Main function to run preprocessing"""
    # Load raw FAQs
    try:
        with open('data/jupiter_faqs.json', 'r', encoding='utf-8') as f:
            raw_faqs = json.load(f)
    except FileNotFoundError:
        logger.error("Raw FAQ data not found. Please run scraper.py first.")
        return
    
    # Initialize preprocessor
    preprocessor = FAQPreprocessor()
    
    # Preprocess FAQs
    processed_faqs = preprocessor.preprocess_faqs(raw_faqs)
    
    # Save processed data
    preprocessor.save_preprocessed_data(processed_faqs)
    
    print(f"Preprocessing complete! Processed {len(processed_faqs)} FAQs")

if __name__ == "__main__":
    main() 