"""
Embeddings and Semantic Search Module for Jupiter FAQ Bot
Implements vector search using FAISS and Chroma with multiple embedding models
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
from typing import List, Dict, Tuple, Optional, Union
import logging
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import time

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embeddings and semantic search for FAQs"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embeddings = None
        self.faqs = None
        self.index = None
        
        # Chroma client for alternative search
        self.chroma_client = None
        self.chroma_collection = None
        
        logger.info(f"Initialized EmbeddingManager with model: {model_name} on {self.device}")
    
    def create_embeddings(self, faqs: List[Dict]) -> np.ndarray:
        """Create embeddings for FAQ data"""
        logger.info(f"Creating embeddings for {len(faqs)} FAQs")
        
        # Prepare texts for embedding
        texts = []
        for faq in faqs:
            # Combine question and answer for better semantic understanding
            combined_text = f"{faq['question']} {faq['answer']}"
            texts.append(combined_text)
        
        # Generate embeddings
        start_time = time.time()
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        end_time = time.time()
        
        logger.info(f"Generated embeddings in {end_time - start_time:.2f} seconds")
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, index_type: str = 'Flat') -> faiss.Index:
        """Build FAISS index for fast similarity search"""
        logger.info(f"Building FAISS index with type: {index_type}")
        
        dimension = embeddings.shape[1]
        
        if index_type == 'IVF' and embeddings.shape[0] > 100:
            # IVF index for large datasets only
            try:
                nlist = min(50, embeddings.shape[0] // 20)  # Reduced number of clusters
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Train the index
                index.train(embeddings)
                index.add(embeddings)
                logger.info(f"FAISS IVF index built with {index.ntotal} vectors")
                return index
            except Exception as e:
                logger.warning(f"IVF index failed, falling back to Flat: {e}")
                # Fall back to flat index
        
        elif index_type == 'HNSW':
            # HNSW index for high-quality search
            try:
                index = faiss.IndexHNSWFlat(dimension, 16)  # Reduced neighbors
                index.hnsw.efConstruction = 100  # Reduced construction
                index.hnsw.efSearch = 50  # Reduced search
                index.add(embeddings)
                logger.info(f"FAISS HNSW index built with {index.ntotal} vectors")
                return index
            except Exception as e:
                logger.warning(f"HNSW index failed, falling back to Flat: {e}")
                # Fall back to flat index
        
        # Simple flat index (most stable)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        logger.info(f"FAISS Flat index built with {index.ntotal} vectors")
        return index
    
    def setup_chroma(self, faqs: List[Dict], embeddings: np.ndarray):
        """Setup Chroma for alternative vector search"""
        try:
            # Initialize Chroma client
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            collection_name = "jupiter_faqs"
            try:
                self.chroma_collection = self.chroma_client.get_collection(collection_name)
                logger.info("Using existing Chroma collection")
                
                # Check if collection is empty and add data if needed
                if self.chroma_collection.count() == 0:
                    logger.info("Chroma collection is empty, adding FAQ data")
                    self._add_faqs_to_chroma(faqs, embeddings)
                else:
                    logger.info(f"Chroma collection has {self.chroma_collection.count()} documents")
                    
            except Exception as e:
                logger.info(f"Creating new Chroma collection: {e}")
                self.chroma_collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Jupiter FAQ embeddings"}
                )
                logger.info("Created new Chroma collection")
                self._add_faqs_to_chroma(faqs, embeddings)
            
        except Exception as e:
            logger.warning(f"Chroma setup failed: {e}")
            self.chroma_client = None
            self.chroma_collection = None
    
    def _add_faqs_to_chroma(self, faqs: List[Dict], embeddings: np.ndarray):
        """Add FAQ data to Chroma collection"""
        try:
            # Add documents to collection
            documents = []
            metadatas = []
            ids = []
            
            for i, faq in enumerate(faqs):
                documents.append(f"{faq['question']} {faq['answer']}")
                metadatas.append({
                    'category': faq.get('category', 'general'),
                    'confidence': faq.get('confidence', 0.8),
                    'source_url': faq.get('source_url', ''),
                    'question': faq['question'],
                    'answer': faq['answer']
                })
                ids.append(f"faq_{i}")
            
            # Add to collection
            self.chroma_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings.tolist()
            )
            
            logger.info(f"Added {len(faqs)} documents to Chroma collection")
            
        except Exception as e:
            logger.error(f"Failed to add FAQs to Chroma: {e}")
            raise
    
    def search_faiss(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Search using FAISS index"""
        if self.index is None:
            logger.warning("FAISS index not available, using simple search")
            return self.search_simple(query, k, threshold)
        
        if self.faqs is None:
            logger.error("FAQs not initialized")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.faqs):
                result = self.faqs[idx].copy()
                result['similarity_score'] = float(score)
                result['rank'] = len(results) + 1
                results.append(result)
        
        return results
    
    def search_chroma(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Search using Chroma"""
        if self.chroma_collection is None:
            logger.error("Chroma collection not initialized")
            return []
        
        try:
            # Search in Chroma
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=k,
                include=['metadatas', 'distances']
            )
            
            formatted_results = []
            for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                # Convert distance to similarity score (Chroma uses L2 distance)
                similarity_score = 1.0 / (1.0 + distance)
                
                if similarity_score >= threshold:
                    result = metadata.copy()
                    result['similarity_score'] = similarity_score
                    result['rank'] = i + 1
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Chroma search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 5, threshold: float = 0.5, 
                     faiss_weight: float = 0.7) -> List[Dict]:
        """Hybrid search combining FAISS and Chroma results"""
        faiss_results = self.search_faiss(query, k, threshold)
        chroma_results = self.search_chroma(query, k, threshold)
        
        # Combine and re-rank results
        combined_results = {}
        
        # Add FAISS results
        for result in faiss_results:
            key = result.get('question', '')
            combined_results[key] = {
                'result': result,
                'faiss_score': result['similarity_score'],
                'chroma_score': 0.0
            }
        
        # Add Chroma results
        for result in chroma_results:
            key = result.get('question', '')
            if key in combined_results:
                combined_results[key]['chroma_score'] = result['similarity_score']
            else:
                combined_results[key] = {
                    'result': result,
                    'faiss_score': 0.0,
                    'chroma_score': result['similarity_score']
                }
        
        # Calculate weighted scores
        final_results = []
        for key, data in combined_results.items():
            weighted_score = (faiss_weight * data['faiss_score'] + 
                            (1 - faiss_weight) * data['chroma_score'])
            
            result = data['result'].copy()
            result['weighted_score'] = weighted_score
            final_results.append(result)
        
        # Sort by weighted score
        final_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Re-rank
        for i, result in enumerate(final_results[:k]):
            result['rank'] = i + 1
        
        return final_results[:k]
    
    def semantic_search(self, query: str, k: int = 5, method: str = 'hybrid', 
                       threshold: float = 0.5) -> List[Dict]:
        """Main semantic search method"""
        logger.info(f"Performing semantic search for: '{query}'")
        
        try:
            if method == 'faiss':
                results = self.search_faiss(query, k, threshold)
            elif method == 'chroma':
                results = self.search_chroma(query, k, threshold)
            else:  # hybrid
                results = self.hybrid_search(query, k, threshold)
        except Exception as e:
            logger.warning(f"Search method {method} failed, using simple search: {e}")
            results = self.search_simple(query, k, threshold)
        
        logger.info(f"Found {len(results)} relevant results")
        return results
    
    def get_similar_questions(self, question: str, k: int = 3) -> List[str]:
        """Get similar questions for suggestions"""
        results = self.semantic_search(question, k, method='faiss')
        return [result['question'] for result in results if result['question'] != question]
    
    def load_embeddings(self, faqs: List[Dict], embeddings_file: str = 'models/faq_embeddings.pkl'):
        """Load pre-computed embeddings"""
        try:
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.faqs = data['faqs']
                self.index = data['index']
            
            logger.info(f"Loaded embeddings from {embeddings_file}")
            return True
            
        except FileNotFoundError:
            logger.info("No pre-computed embeddings found, creating new ones...")
            return False
    
    def save_embeddings(self, embeddings_file: str = 'models/faq_embeddings.pkl'):
        """Save embeddings and index"""
        if self.embeddings is None or self.faqs is None or self.index is None:
            logger.error("No embeddings to save")
            return
        
        os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
        
        data = {
            'embeddings': self.embeddings,
            'faqs': self.faqs,
            'index': self.index,
            'model_name': self.model_name
        }
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved embeddings to {embeddings_file}")
    
    def initialize_from_faqs(self, faqs: List[Dict], embeddings_file: str = 'models/faq_embeddings.pkl'):
        """Initialize embeddings from FAQ data"""
        # Try to load existing embeddings
        if self.load_embeddings(faqs, embeddings_file):
            return
        
        # Create new embeddings
        self.faqs = faqs
        self.embeddings = self.create_embeddings(faqs)
        
        # Build FAISS index with fallback
        try:
            self.index = self.build_faiss_index(self.embeddings)
        except Exception as e:
            logger.warning(f"FAISS index building failed: {e}")
            self.index = None
        
        # Setup Chroma
        self.setup_chroma(faqs, self.embeddings)
        
        # Save embeddings (only if FAISS index was built successfully)
        if self.index is not None:
            self.save_embeddings(embeddings_file)
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about the embeddings"""
        if self.embeddings is None:
            return {}
        
        stats = {
            'total_faqs': len(self.faqs) if self.faqs else 0,
            'embedding_dimension': self.embeddings.shape[1],
            'model_name': self.model_name,
            'device': self.device,
            'index_type': type(self.index).__name__ if self.index else None,
            'chroma_available': self.chroma_collection is not None
        }
        
        if self.faqs:
            categories = [faq.get('category', 'general') for faq in self.faqs]
            stats['categories'] = pd.Series(categories).value_counts().to_dict()
        
        return stats
    
    def search_simple(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Simple search using cosine similarity without FAISS"""
        if self.embeddings is None or self.faqs is None:
            logger.error("Embeddings not initialized")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                result = self.faqs[idx].copy()
                result['similarity_score'] = float(similarity)
                result['rank'] = len(results) + 1
                results.append(result)
        
        return results

def main():
    """Main function to test embeddings"""
    # Load preprocessed FAQs
    try:
        with open('data/preprocessed_faqs.json', 'r', encoding='utf-8') as f:
            faqs = json.load(f)
    except FileNotFoundError:
        logger.error("Preprocessed FAQ data not found. Please run preprocessor.py first.")
        return
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Initialize embeddings
    embedding_manager.initialize_from_faqs(faqs)
    
    # Test search
    test_queries = [
        "How do I complete KYC?",
        "What payment methods are accepted?",
        "How do I earn rewards?",
        "What are the transaction limits?",
        "How do I reset my password?"
    ]
    
    print("Testing semantic search...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = embedding_manager.semantic_search(query, k=3)
        for result in results:
            print(f"  Score: {result['similarity_score']:.3f} - {result['question']}")
    
    # Print stats
    stats = embedding_manager.get_embedding_stats()
    print(f"\nEmbedding Statistics: {stats}")

if __name__ == "__main__":
    main() 