import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import openai
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import json
import time

# Configure page
st.set_page_config(page_title="Hybrid Keyword Clustering", layout="wide")

# Initialize session state
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'keywords_df' not in st.session_state:
    st.session_state.keywords_df = None

def term_frequency_clustering(keywords, min_freq=2):
    """Your original term frequency approach"""
    word_counts = Counter()
    keyword_words = {}
    
    for keyword in keywords:
        words = keyword.lower().split()
        keyword_words[keyword] = words
        word_counts.update(words)
    
    # Get high-frequency terms
    common_terms = {word: count for word, count in word_counts.items() 
                   if count >= min_freq}
    
    # Group keywords by their most frequent term
    clusters = {}
    unclustered = []
    
    for keyword in keywords:
        words = keyword_words[keyword]
        best_term = None
        best_count = 0
        
        for word in words:
            if word in common_terms and common_terms[word] > best_count:
                best_term = word
                best_count = common_terms[word]
        
        if best_term:
            if best_term not in clusters:
                clusters[best_term] = []
            clusters[best_term].append(keyword)
        else:
            unclustered.append(keyword)
    
    return clusters, unclustered, common_terms

def semantic_clustering_with_llm(keywords_sample, api_key, model="gpt-4o-mini"):
    """Use LLM for semantic clustering of a sample"""
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    Analyze these keywords and identify semantic themes/intents. Group them into clusters.
    
    Keywords: {', '.join(keywords_sample)}
    
    Return a JSON object where keys are cluster names and values are lists of keywords.
    Focus on user intent and semantic meaning, not just word matching.
    
    Example format:
    {{
        "product_research": ["best running shoes", "top rated sneakers"],
        "purchase_intent": ["buy running shoes", "running shoes for sale"],
        "comparison": ["nike vs adidas shoes", "shoe comparison guide"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {}
    except Exception as e:
        st.error(f"LLM clustering error: {e}")
        return {}

def hybrid_clustering(keywords, tf_clusters, semantic_clusters, hybrid_weight=0.7):
    """Combine term frequency and semantic clustering"""
    final_clusters = {}
    
    # Start with semantic clusters as base
    for cluster_name, cluster_keywords in semantic_clusters.items():
        final_clusters[cluster_name] = set(cluster_keywords)
    
    # Add term frequency insights
    for tf_term, tf_keywords in tf_clusters.items():
        # Find best matching semantic cluster
        best_match = None
        best_overlap = 0
        
        for sem_name, sem_keywords in final_clusters.items():
            overlap = len(set(tf_keywords) & sem_keywords)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = sem_name
        
        if best_match and best_overlap > 0:
            # Merge into existing semantic cluster
            final_clusters[best_match].update(tf_keywords)
        else:
            # Create new cluster based on term frequency
            final_clusters[f"tf_{tf_term}"] = set(tf_keywords)
    
    # Convert sets back to lists
    return {k: list(v) for k, v in final_clusters.items()}

# Streamlit UI
st.title("🔍 Hybrid Keyword Clustering Tool")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password")
    
    # Model selection
    model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    
    # Clustering parameters
    st.subheader("Clustering Parameters")
    min_freq = st.slider("Min Term Frequency", 1, 10, 2)
    sample_size = st.slider("LLM Sample Size", 10, 100, 20)
    hybrid_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.7)

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Input Keywords")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        keyword_column = st.selectbox("Select keyword column", df.columns)
        keywords = df[keyword_column].dropna().tolist()
        st.success(f"Loaded {len(keywords)} keywords")
    else:
        # Manual input
        keywords_text = st.text_area("Or paste keywords (one per line)", 
                                   placeholder="running shoes\nbest sneakers\nshoe comparison\n...")
        keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
    
    # Preview sample for LLM
    if keywords and len(keywords) > 0:
        st.subheader("🎯 Sample Preview")
        sample_keywords = keywords[:sample_size] if len(keywords) > sample_size else keywords
        st.write(f"Will analyze {len(sample_keywords)} keywords with LLM")
        
        with st.expander("View sample"):
            st.write(sample_keywords)

with col2:
    st.subheader("📊 Clustering Results")
    
    if keywords and len(keywords) > 0:
        if st.button("🚀 Run Clustering", type="primary"):
            with st.spinner("Running hybrid clustering..."):
                # Step 1: Term Frequency Clustering
                with st.status("Step 1: Term Frequency Analysis") as status:
                    tf_clusters, unclustered, common_terms = term_frequency_clustering(keywords, min_freq)
                    st.write(f"Found {len(tf_clusters)} frequency-based clusters")
                    status.update(label="Term frequency analysis complete", state="complete")
                
                # Step 2: Semantic Clustering (sample)
                if api_key:
                    with st.status("Step 2: Semantic Analysis") as status:
                        sample_keywords = keywords[:sample_size] if len(keywords) > sample_size else keywords
                        semantic_clusters = semantic_clustering_with_llm(sample_keywords, api_key, model)
                        st.write(f"Found {len(semantic_clusters)} semantic clusters")
                        status.update(label="Semantic analysis complete", state="complete")
                else:
                    semantic_clusters = {}
                    st.warning("No API key provided - using only term frequency clustering")
                
                # Step 3: Hybrid Clustering
                with st.status("Step 3: Hybrid Clustering") as status:
                    if semantic_clusters:
                        final_clusters = hybrid_clustering(keywords, tf_clusters, semantic_clusters, hybrid_weight)
                    else:
                        final_clusters = tf_clusters
                    
                    st.session_state.clusters = final_clusters
                    status.update(label="Hybrid clustering complete", state="complete")
        
        # Display results
        if st.session_state.clusters:
            st.subheader("🎯 Final Clusters")
            
            # Cluster statistics
            total_clustered = sum(len(keywords) for keywords in st.session_state.clusters.values())
            st.metric("Total Clusters", len(st.session_state.clusters))
            st.metric("Keywords Clustered", total_clustered)
            st.metric("Clustering Rate", f"{total_clustered/len(keywords)*100:.1f}%")
            
            # Display clusters
            for cluster_name, cluster_keywords in st.session_state.clusters.items():
                with st.expander(f"📂 {cluster_name} ({len(cluster_keywords)} keywords)"):
                    st.write(cluster_keywords)
            
            # Download results
            results_df = []
            for cluster_name, cluster_keywords in st.session_state.clusters.items():
                for keyword in cluster_keywords:
                    results_df.append({"keyword": keyword, "cluster": cluster_name})
            
            results_df = pd.DataFrame(results_df)
            csv = results_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="keyword_clusters.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("💡 **Tips**: Start with a small sample to test, then scale up. Adjust the semantic weight to balance between frequency and meaning-based clustering.")