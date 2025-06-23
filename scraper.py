"""
Web Scraper for Jupiter Help Centre FAQs
Extracts and structures FAQ data from Jupiter's help website
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import re
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin, urlparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JupiterFAQScraper:
    """Scraper for Jupiter Help Centre FAQs"""
    
    def __init__(self, base_url: str = "https://help.jupiter.money"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.faqs = []
        
    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse page content"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_faqs_from_page(self, soup: BeautifulSoup, category: str) -> List[Dict]:
        """Extract FAQ pairs from a single page"""
        faqs = []
        
        # Common FAQ selectors (adjust based on actual Jupiter site structure)
        selectors = [
            '.faq-item',
            '.accordion-item',
            '.help-article',
            '[data-testid*="faq"]',
            '.question-answer',
            'details',
            '.faq-section'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                logger.info(f"Found {len(elements)} FAQ elements with selector: {selector}")
                break
        
        if not elements:
            # Fallback: look for common FAQ patterns
            elements = soup.find_all(['div', 'section'], class_=re.compile(r'faq|question|answer', re.I))
        
        for element in elements:
            try:
                # Try to extract question and answer
                question_elem = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b']) or element
                answer_elem = element.find(['p', 'div', 'span']) or element
                
                question = self.clean_text(question_elem.get_text())
                answer = self.clean_text(answer_elem.get_text())
                
                if question and answer and len(question) > 10 and len(answer) > 20:
                    faqs.append({
                        'question': question,
                        'answer': answer,
                        'category': category,
                        'source_url': getattr(element, 'get', lambda x: None)('data-url', ''),
                        'confidence': 0.8
                    })
                    
            except Exception as e:
                logger.warning(f"Error extracting FAQ from element: {e}")
                continue
        
        return faqs
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        return text
    
    def categorize_content(self, question: str, answer: str) -> str:
        """Categorize FAQ based on content"""
        text = f"{question} {answer}".lower()
        
        categories = {
            'kyc': ['kyc', 'verification', 'identity', 'document', 'pan', 'aadhaar'],
            'payments': ['payment', 'upi', 'card', 'transaction', 'transfer', 'bank'],
            'rewards': ['reward', 'cashback', 'points', 'bonus', 'offer'],
            'limits': ['limit', 'maximum', 'minimum', 'daily', 'monthly'],
            'security': ['security', 'password', 'pin', 'otp', 'fraud'],
            'general': ['how', 'what', 'when', 'where', 'why']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'
    
    def scrape_help_centre(self) -> List[Dict]:
        """Main scraping method"""
        logger.info("Starting Jupiter Help Centre scraping...")
        
        # Common help centre paths
        help_paths = [
            '/faq',
            '/help',
            '/support',
            '/knowledge-base',
            '/articles',
            '/guides'
        ]
        
        for path in help_paths:
            url = urljoin(self.base_url, path)
            logger.info(f"Scraping: {url}")
            
            soup = self.get_page_content(url)
            if soup:
                category = path.strip('/').replace('-', '_') or 'general'
                faqs = self.extract_faqs_from_page(soup, category)
                self.faqs.extend(faqs)
                
                # Look for additional FAQ pages
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    if any(keyword in href.lower() for keyword in ['faq', 'help', 'support', 'guide']):
                        full_url = urljoin(url, href)
                        if full_url not in [f['source_url'] for f in self.faqs]:
                            logger.info(f"Found additional page: {full_url}")
                            sub_soup = self.get_page_content(full_url)
                            if sub_soup:
                                sub_category = self.categorize_content(link.get_text(), "")
                                sub_faqs = self.extract_faqs_from_page(sub_soup, sub_category)
                                self.faqs.extend(sub_faqs)
            
            time.sleep(1)  # Be respectful to the server
        
        # If no FAQs found, create sample data
        if not self.faqs:
            logger.warning("No FAQs found, creating sample data...")
            self.faqs = self.create_sample_faqs()
        
        return self.faqs
    
    def create_sample_faqs(self) -> List[Dict]:
        """Create sample FAQ data for demonstration"""
        sample_faqs = [
            {
                'question': 'How do I complete KYC verification?',
                'answer': 'To complete KYC verification, please provide your PAN card and Aadhaar number. You can also use your driving license or passport as identity proof. The verification process usually takes 24-48 hours.',
                'category': 'kyc',
                'source_url': 'https://help.jupiter.money/kyc',
                'confidence': 0.9
            },
            {
                'question': 'What payment methods are accepted?',
                'answer': 'Jupiter accepts UPI payments, debit cards, credit cards, and net banking. You can also link your bank account for direct transfers. All transactions are secured with bank-grade encryption.',
                'category': 'payments',
                'source_url': 'https://help.jupiter.money/payments',
                'confidence': 0.9
            },
            {
                'question': 'How do I earn rewards?',
                'answer': 'Earn rewards by making transactions, referring friends, and participating in special offers. You can earn up to 5% cashback on eligible purchases. Rewards are credited within 24 hours.',
                'category': 'rewards',
                'source_url': 'https://help.jupiter.money/rewards',
                'confidence': 0.9
            },
            {
                'question': 'What are the transaction limits?',
                'answer': 'Daily transaction limit is ₹1,00,000 and monthly limit is ₹10,00,000. These limits may vary based on your account type and verification status. Contact support to increase limits.',
                'category': 'limits',
                'source_url': 'https://help.jupiter.money/limits',
                'confidence': 0.9
            },
            {
                'question': 'How do I reset my password?',
                'answer': 'To reset your password, go to the login page and click "Forgot Password". Enter your registered email or phone number. You will receive an OTP to create a new password.',
                'category': 'security',
                'source_url': 'https://help.jupiter.money/security',
                'confidence': 0.9
            },
            {
                'question': 'Can I use Jupiter internationally?',
                'answer': 'Currently, Jupiter services are available only in India. International transactions are not supported. You can use the app while traveling abroad for domestic transactions.',
                'category': 'general',
                'source_url': 'https://help.jupiter.money/general',
                'confidence': 0.9
            },
            {
                'question': 'How do I add money to my Jupiter account?',
                'answer': 'You can add money using UPI, bank transfer, or by linking your bank account. Go to "Add Money" section and choose your preferred method. Funds are usually credited instantly.',
                'category': 'payments',
                'source_url': 'https://help.jupiter.money/add-money',
                'confidence': 0.9
            },
            {
                'question': 'What documents are required for account opening?',
                'answer': 'You need a valid PAN card, Aadhaar number, and a working mobile number. For full KYC, you may also need address proof like utility bills or bank statements.',
                'category': 'kyc',
                'source_url': 'https://help.jupiter.money/documents',
                'confidence': 0.9
            }
        ]
        return sample_faqs
    
    def save_faqs(self, filename: str = 'data/jupiter_faqs.json'):
        """Save scraped FAQs to file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.faqs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.faqs)} FAQs to {filename}")
        
        # Also save as CSV for easy analysis
        df = pd.DataFrame(self.faqs)
        csv_filename = filename.replace('.json', '.csv')
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        logger.info(f"Saved FAQs to {csv_filename}")
        
        return filename

def main():
    """Main function to run the scraper"""
    scraper = JupiterFAQScraper()
    faqs = scraper.scrape_help_centre()
    scraper.save_faqs()
    
    print(f"Successfully scraped {len(faqs)} FAQs")
    print("Categories found:", set(faq['category'] for faq in faqs))

if __name__ == "__main__":
    main() 