# README.md
# 🔍 Hybrid Keyword Clustering Tool

A powerful tool that combines term frequency analysis with semantic clustering using Large Language Models (LLMs) to intelligently group keywords by both word patterns and user intent.

## 🎯 What It Does

This tool implements a **hybrid clustering approach** that outperforms traditional keyword grouping methods:

1. **Term Frequency Clustering**: Groups keywords that share common words (your original idea)
2. **Semantic Clustering**: Uses LLMs to understand user intent and meaning
3. **Hybrid Fusion**: Combines both approaches for optimal results

## 🚀 Key Features

- **Token-Efficient**: Preview samples before processing to save API costs
- **Interactive UI**: Tune parameters in real-time with immediate feedback
- **Multiple LLM Support**: OpenAI GPT models (extensible to others)
- **Export Results**: Download clustered keywords as CSV
- **Docker Ready**: One-command setup with Docker Compose

## 📊 How It Works

### Your Original Method (Enhanced)
```
Keywords: "international shipping", "shipping costs", "fast delivery"
→ Splits: ["international", "shipping"], ["shipping", "costs"], ["fast", "delivery"]
→ Frequency: shipping=2, international=1, costs=1, fast=1, delivery=1
→ Clusters: shipping_cluster=["international shipping", "shipping costs"]
```

### Semantic Enhancement
```
LLM analyzes: "international shipping", "shipping costs", "fast delivery"
→ Intent Groups: logistics_costs=["international shipping", "shipping costs"]
                delivery_speed=["fast delivery"]
```

### Hybrid Result
Combines frequency patterns with semantic understanding for more accurate grouping.

## 🛠️ Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key

### Setup
```bash
# 1. Clone or create project folder
mkdir keyword-clustering && cd keyword-clustering

# 2. Add all project files (app.py, Dockerfile, etc.)

# 3. Configure API key
cp .env.example .env
# Edit .env with your OpenAI API key

# 4. Run
chmod +x run.sh
./run.sh
```

### Access
Open `http://localhost:8501` in your browser

## 💡 Usage Tips

### For Testing & Iteration
1. **Start Small**: Upload 20-50 keywords first
2. **Preview Sample**: Check what gets sent to LLM before running
3. **Tune Parameters**: Adjust frequency thresholds and semantic weights
4. **Validate Results**: Review clusters before processing full dataset

### For Production
1. **Scale Gradually**: Test with samples, then increase batch sizes
2. **Monitor Costs**: Use `gpt-4o-mini` for cost efficiency
3. **Save Results**: Export to CSV for further analysis

## ⚙️ Configuration

### Environment Variables (.env)
```env
OPENAI_API_KEY=your_api_key_here
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_SAMPLE_SIZE=20
DEFAULT_MIN_FREQUENCY=2
DEFAULT_SEMANTIC_WEIGHT=0.7
```

### UI Parameters
- **Min Term Frequency**: How many times a word must appear to create a cluster
- **LLM Sample Size**: Number of keywords to analyze semantically (affects cost)
- **Semantic Weight**: Balance between frequency (0.0) and semantic (1.0) clustering

## 📈 Benefits Over Simple Clustering

| Method | Pros | Cons |
|--------|------|------|
| **Frequency Only** | Fast, transparent, finds obvious patterns | Misses semantic relationships, ignores intent |
| **Semantic Only** | Understands meaning and intent | Expensive, slower, may miss word patterns |
| **Hybrid** | Best of both worlds, more accurate | Slightly more complex |

## 🔧 Advanced Features

### Clustering Quality Metrics
- **Clustering Rate**: Percentage of keywords successfully clustered
- **Cluster Distribution**: Size and balance of clusters
- **Intent Alignment**: How well clusters match user search intent

### Extensibility
- **Multi-LLM Support**: Easy to add Anthropic, Google, or other APIs
- **Custom Algorithms**: Pluggable clustering methods
- **Export Formats**: CSV, JSON, Excel support

## 🐛 Troubleshooting

### Common Issues
- **"No API key"**: Check .env file configuration
- **"Docker not running"**: Start Docker Desktop
- **"Port 8501 in use"**: Change port in docker-compose.yml
- **High API costs**: Reduce sample size or use gpt-4o-mini

### Performance Tips
- Use smaller samples for testing (10-20 keywords)
- Increase batch size gradually for production
- Monitor token usage in OpenAI dashboard

## 📁 Project Structure
```
keyword-clustering/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── run.sh               # Quick start script
├── .env                 # Environment variables
├── data/                # Input CSV files
├── output/              # Generated results
└── README.md            # This file
```

## 🤝 Contributing

Feel free to extend this tool with:
- Additional LLM providers
- Advanced clustering algorithms
- SERP-based clustering
- Performance optimizations

## 📄 License

MIT License - feel free to use and modify for your projects.

---

*Built for SEO professionals who need intelligent keyword clustering that understands both patterns and intent.*
