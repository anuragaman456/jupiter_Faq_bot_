# Jupiter FAQ Bot 🤖

An intelligent, conversational FAQ bot built from Jupiter's Help Centre data with beautiful UI and advanced NLP capabilities.

## 🌟 Features

- **Web Scraping**: Automated extraction of FAQs from Jupiter's help centre
- **Semantic Search**: Advanced embedding-based search using FAISS/Chroma
- **Multi-Model Support**: OpenAI, Mistral, and local LLM options
- **Beautiful UI**: Interactive Streamlit interface with modern design
- **Multilingual Support**: Hindi/Hinglish language support
- **Smart Suggestions**: Related query recommendations
- **Evaluation Tools**: Comprehensive accuracy and performance metrics

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export MISTRAL_API_KEY="your_mistral_api_key"  # Optional
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
jupiter_assignment/
├── app.py                 # Main Streamlit application
├── scraper.py            # Web scraping module
├── preprocessor.py       # Data cleaning and preprocessing
├── bot.py               # Core FAQ bot logic
├── embeddings.py        # Semantic search implementation
├── evaluation.py        # Performance evaluation tools
├── data/               # Scraped and processed data
├── models/             # Saved models and embeddings
└── notebooks/          # Jupyter notebooks for analysis
```

## 🎯 Usage

1. **Data Collection**: Run the scraper to collect FAQs
2. **Preprocessing**: Clean and structure the data
3. **Model Training**: Generate embeddings and train models
4. **Interactive Chat**: Use the web interface to chat with the bot

## 🔧 Configuration

- **Models**: Choose between OpenAI GPT, Mistral, or local models
- **Languages**: Support for English, Hindi, and Hinglish
- **Search**: FAISS or Chroma for semantic search
- **UI Theme**: Customizable dark/light themes

## 📊 Evaluation

The bot includes comprehensive evaluation metrics:
- Semantic similarity scores
- Response relevance
- User satisfaction metrics
- Performance benchmarks

## 🤝 Contributing

Feel free to contribute by:
- Adding new language support
- Improving the UI/UX
- Enhancing the evaluation metrics
- Adding new features

## 📄 License

MIT License - feel free to use this project for educational and commercial purposes. 