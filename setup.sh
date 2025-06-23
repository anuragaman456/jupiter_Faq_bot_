#!/bin/bash

# Jupiter FAQ Bot Setup Script
echo "🚀 Setting up Jupiter FAQ Bot..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

# Create virtual environment (optional but recommended)
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📖 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p models
mkdir -p logs

# Set up environment variables
echo "🔐 Setting up environment variables..."
if [ ! -f .env ]; then
    cat > .env << EOF
# Jupiter FAQ Bot Environment Variables
# Add your API keys here

# OpenAI API Key (optional)
# OPENAI_API_KEY=your_openai_api_key_here

# Gemini API Key (optional)
# GEMINI_API_KEY=your_gemini_api_key_here

# Logging
LOG_LEVEL=INFO
EOF
    echo "✅ Created .env file. Please add your API keys if needed."
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the scraper: python scraper.py"
echo "3. Run the web app: streamlit run app.py"
echo ""
echo "📖 For more information, check the README.md file." 