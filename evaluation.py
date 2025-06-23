"""
Evaluation Module for Jupiter FAQ Bot
Provides comprehensive metrics for assessing bot performance
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

logger = logging.getLogger(__name__)

class FAQBotEvaluator:
    """Evaluates FAQ bot performance"""
    
    def __init__(self, bot, test_data: List[Dict] = None):
        self.bot = bot
        self.test_data = test_data or []
        self.evaluation_results = {}
        
    def load_test_data(self, filename: str = 'data/test_queries.json'):
        """Load test queries and expected answers"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
            logger.info(f"Loaded {len(self.test_data)} test queries")
        except FileNotFoundError:
            logger.warning(f"Test data file {filename} not found, using sample data")
            self.test_data = self._create_sample_test_data()
    
    def _create_sample_test_data(self) -> List[Dict]:
        """Create sample test data for evaluation"""
        return [
            {
                'query': 'How do I complete KYC verification?',
                'expected_answer': 'KYC verification requires PAN card and Aadhaar number',
                'category': 'kyc',
                'difficulty': 'easy'
            },
            {
                'query': 'What payment methods are accepted?',
                'expected_answer': 'Jupiter accepts UPI, cards, and net banking',
                'category': 'payments',
                'difficulty': 'easy'
            },
            {
                'query': 'How do I earn rewards?',
                'expected_answer': 'Earn rewards through transactions and referrals',
                'category': 'rewards',
                'difficulty': 'medium'
            },
            {
                'query': 'What are the transaction limits?',
                'expected_answer': 'Daily limit is ₹1,00,000 and monthly is ₹10,00,000',
                'category': 'limits',
                'difficulty': 'medium'
            },
            {
                'query': 'How do I reset my password?',
                'expected_answer': 'Use forgot password option with OTP verification',
                'category': 'security',
                'difficulty': 'easy'
            },
            {
                'query': 'Can I use Jupiter internationally?',
                'expected_answer': 'Jupiter is only available in India',
                'category': 'general',
                'difficulty': 'easy'
            },
            {
                'query': 'How do I add money to my account?',
                'expected_answer': 'Add money using UPI, bank transfer, or linked account',
                'category': 'payments',
                'difficulty': 'medium'
            },
            {
                'query': 'What documents are required for account opening?',
                'expected_answer': 'PAN card, Aadhaar, and mobile number are required',
                'category': 'kyc',
                'difficulty': 'medium'
            }
        ]
    
    def evaluate_semantic_similarity(self, query: str, response: str, expected: str) -> float:
        """Calculate semantic similarity between response and expected answer"""
        try:
            # Use the bot's embedding model to get embeddings
            embeddings = self.bot.embedding_manager.model.encode([response, expected])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def evaluate_keyword_overlap(self, response: str, expected: str) -> float:
        """Calculate keyword overlap between response and expected answer"""
        try:
            # Simple keyword extraction
            response_words = set(response.lower().split())
            expected_words = set(expected.lower().split())
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            response_keywords = response_words - stop_words
            expected_keywords = expected_words - stop_words
            
            if not expected_keywords:
                return 0.0
            
            overlap = len(response_keywords.intersection(expected_keywords))
            return overlap / len(expected_keywords)
            
        except Exception as e:
            logger.error(f"Error calculating keyword overlap: {e}")
            return 0.0
    
    def evaluate_response_length(self, response: str) -> Dict:
        """Evaluate response length appropriateness"""
        words = response.split()
        sentences = response.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': len(words) / max(1, len([s for s in sentences if s.strip()])),
            'is_appropriate_length': 10 <= len(words) <= 200
        }
    
    def evaluate_confidence_calibration(self, confidence: float, actual_quality: float) -> float:
        """Evaluate how well confidence scores match actual response quality"""
        return 1.0 - abs(confidence - actual_quality)
    
    def run_evaluation(self, test_queries: List[Dict] = None) -> Dict:
        """Run comprehensive evaluation"""
        if test_queries is None:
            test_queries = self.test_data
        
        logger.info(f"Running evaluation on {len(test_queries)} test queries")
        
        results = []
        total_queries = len(test_queries)
        
        for i, test_case in enumerate(test_queries):
            logger.info(f"Evaluating query {i+1}/{total_queries}: {test_case['query']}")
            
            # Get bot response
            bot_response = self.bot.ask(test_case['query'])
            
            # Calculate metrics
            semantic_similarity = self.evaluate_semantic_similarity(
                test_case['query'], 
                bot_response['answer'], 
                test_case['expected_answer']
            )
            
            keyword_overlap = self.evaluate_keyword_overlap(
                bot_response['answer'], 
                test_case['expected_answer']
            )
            
            length_metrics = self.evaluate_response_length(bot_response['answer'])
            
            # Overall quality score (weighted average)
            quality_score = (0.6 * semantic_similarity + 0.4 * keyword_overlap)
            
            confidence_calibration = self.evaluate_confidence_calibration(
                bot_response['confidence'], 
                quality_score
            )
            
            result = {
                'query': test_case['query'],
                'expected_answer': test_case['expected_answer'],
                'bot_answer': bot_response['answer'],
                'category': test_case.get('category', 'general'),
                'difficulty': test_case.get('difficulty', 'medium'),
                'semantic_similarity': semantic_similarity,
                'keyword_overlap': keyword_overlap,
                'quality_score': quality_score,
                'bot_confidence': bot_response['confidence'],
                'confidence_calibration': confidence_calibration,
                'response_length': length_metrics,
                'has_suggestions': len(bot_response.get('suggestions', [])) > 0,
                'suggestion_count': len(bot_response.get('suggestions', []))
            }
            
            results.append(result)
        
        # Calculate aggregate metrics
        self.evaluation_results = self._calculate_aggregate_metrics(results)
        self.evaluation_results['detailed_results'] = results
        
        logger.info("Evaluation completed successfully")
        return self.evaluation_results
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate evaluation metrics"""
        df = pd.DataFrame(results)
        
        metrics = {
            'total_queries': len(results),
            'avg_semantic_similarity': df['semantic_similarity'].mean(),
            'avg_keyword_overlap': df['keyword_overlap'].mean(),
            'avg_quality_score': df['quality_score'].mean(),
            'avg_confidence': df['bot_confidence'].mean(),
            'avg_confidence_calibration': df['confidence_calibration'].mean(),
            'avg_response_length': df['response_length'].apply(lambda x: x['word_count']).mean(),
            'avg_suggestions_per_query': df['suggestion_count'].mean(),
            'queries_with_suggestions': (df['suggestion_count'] > 0).sum(),
            'high_quality_responses': (df['quality_score'] > 0.7).sum(),
            'medium_quality_responses': ((df['quality_score'] > 0.5) & (df['quality_score'] <= 0.7)).sum(),
            'low_quality_responses': (df['quality_score'] <= 0.5).sum()
        }
        
        # Category-wise performance
        category_metrics = {}
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            category_metrics[category] = {
                'count': len(cat_df),
                'avg_quality': cat_df['quality_score'].mean(),
                'avg_confidence': cat_df['bot_confidence'].mean()
            }
        
        metrics['category_performance'] = category_metrics
        
        # Difficulty-wise performance
        difficulty_metrics = {}
        for difficulty in df['difficulty'].unique():
            diff_df = df[df['difficulty'] == difficulty]
            difficulty_metrics[difficulty] = {
                'count': len(diff_df),
                'avg_quality': diff_df['quality_score'].mean(),
                'avg_confidence': diff_df['bot_confidence'].mean()
            }
        
        metrics['difficulty_performance'] = difficulty_metrics
        
        return metrics
    
    def generate_evaluation_report(self, output_file: str = 'evaluation_report.json') -> str:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            logger.error("No evaluation results available. Run evaluation first.")
            return ""
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'bot_configuration': self.bot.get_bot_stats(),
            'metrics': self.evaluation_results,
            'summary': self._generate_summary()
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation report saved to {output_file}")
        return output_file
    
    def _generate_summary(self) -> Dict:
        """Generate evaluation summary"""
        metrics = self.evaluation_results
        
        summary = {
            'overall_performance': 'Good' if metrics['avg_quality_score'] > 0.7 else 'Fair' if metrics['avg_quality_score'] > 0.5 else 'Poor',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Identify strengths
        if metrics['avg_semantic_similarity'] > 0.7:
            summary['strengths'].append("High semantic understanding")
        if metrics['avg_confidence_calibration'] > 0.8:
            summary['strengths'].append("Good confidence calibration")
        if metrics['queries_with_suggestions'] > metrics['total_queries'] * 0.7:
            summary['strengths'].append("Good suggestion generation")
        
        # Identify weaknesses
        if metrics['avg_quality_score'] < 0.6:
            summary['weaknesses'].append("Low overall response quality")
        if metrics['avg_confidence_calibration'] < 0.6:
            summary['weaknesses'].append("Poor confidence calibration")
        if metrics['low_quality_responses'] > metrics['total_queries'] * 0.3:
            summary['weaknesses'].append("Too many low-quality responses")
        
        # Generate recommendations
        if metrics['avg_quality_score'] < 0.7:
            summary['recommendations'].append("Improve semantic search accuracy")
        if metrics['avg_confidence_calibration'] < 0.7:
            summary['recommendations'].append("Better confidence scoring")
        if metrics['queries_with_suggestions'] < metrics['total_queries'] * 0.5:
            summary['recommendations'].append("Increase suggestion generation")
        
        return summary
    
    def create_visualizations(self, output_dir: str = 'evaluation_plots'):
        """Create evaluation visualizations"""
        if not self.evaluation_results:
            logger.error("No evaluation results available. Run evaluation first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(self.evaluation_results['detailed_results'])
        metrics = self.evaluation_results
        
        # 1. Quality Score Distribution
        fig = px.histogram(df, x='quality_score', nbins=20, 
                          title='Distribution of Response Quality Scores',
                          labels={'quality_score': 'Quality Score', 'count': 'Number of Responses'})
        fig.write_html(f"{output_dir}/quality_distribution.html")
        
        # 2. Confidence vs Quality
        fig = px.scatter(df, x='bot_confidence', y='quality_score', 
                        title='Confidence vs Quality Score',
                        labels={'bot_confidence': 'Bot Confidence', 'quality_score': 'Quality Score'})
        fig.write_html(f"{output_dir}/confidence_vs_quality.html")
        
        # 3. Category Performance
        cat_df = pd.DataFrame(metrics['category_performance']).T
        fig = px.bar(cat_df, y='avg_quality', title='Average Quality by Category',
                    labels={'avg_quality': 'Average Quality Score', 'index': 'Category'})
        fig.write_html(f"{output_dir}/category_performance.html")
        
        # 4. Difficulty Performance
        diff_df = pd.DataFrame(metrics['difficulty_performance']).T
        fig = px.bar(diff_df, y='avg_quality', title='Average Quality by Difficulty',
                    labels={'avg_quality': 'Average Quality Score', 'index': 'Difficulty'})
        fig.write_html(f"{output_dir}/difficulty_performance.html")
        
        # 5. Comprehensive Dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality Distribution', 'Confidence vs Quality', 
                          'Category Performance', 'Difficulty Performance'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Quality distribution
        fig.add_trace(go.Histogram(x=df['quality_score'], name='Quality'), row=1, col=1)
        
        # Confidence vs Quality
        fig.add_trace(go.Scatter(x=df['bot_confidence'], y=df['quality_score'], 
                                mode='markers', name='Confidence vs Quality'), row=1, col=2)
        
        # Category performance
        fig.add_trace(go.Bar(x=list(cat_df.index), y=cat_df['avg_quality'], 
                            name='Category Quality'), row=2, col=1)
        
        # Difficulty performance
        fig.add_trace(go.Bar(x=list(diff_df.index), y=diff_df['avg_quality'], 
                            name='Difficulty Quality'), row=2, col=2)
        
        fig.update_layout(height=800, title_text="FAQ Bot Evaluation Dashboard")
        fig.write_html(f"{output_dir}/evaluation_dashboard.html")
        
        logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Test the evaluator"""
    # This would typically be run after the bot is set up
    print("FAQ Bot Evaluator - Run this after setting up the bot")
    print("Example usage:")
    print("1. Initialize bot with embeddings")
    print("2. Create evaluator: evaluator = FAQBotEvaluator(bot)")
    print("3. Run evaluation: results = evaluator.run_evaluation()")
    print("4. Generate report: evaluator.generate_evaluation_report()")
    print("5. Create visualizations: evaluator.create_visualizations()")

if __name__ == "__main__":
    main() 