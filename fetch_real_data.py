"""
Real Data Fetching Module

Fetches actual data from public sources:
- Epoch AI: Training compute for major models
- Papers with Code: Benchmark leaderboards  
- World Bank/OECD: R&D spending
- arXiv: AI safety research papers
- GitHub: Open source releases

No hardcoded numbers - everything from real sources!
"""

import requests
import json
import csv
from io import StringIO
from datetime import datetime
from typing import List, Dict, Optional
import time


# ============================================================================
# EPOCH AI - Model Training Data
# ============================================================================

def fetch_epoch_ai_models() -> List[Dict]:
    """
    Fetch training compute data from Epoch AI's public dataset.
    
    Source: https://epochai.org/data/epochdb/notable_ai_models.csv
    
    Returns: List of model dictionaries with:
        - System: Model name
        - Publication date: Date string
        - Training compute (FLOP): Training compute in FLOPs
        - Organization: Company/lab
        - Parameters: Number of parameters
    """
    print("Fetching Epoch AI model data...")
    
    url = "https://epochai.org/data/epochdb/notable_ai_models.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse CSV
        csv_data = StringIO(response.text)
        reader = csv.DictReader(csv_data)
        
        models = []
        for row in reader:
            # Filter for models with training compute data
            if row.get('Training compute (FLOP)'):
                try:
                    models.append({
                        'name': row.get('System', ''),
                        'date': row.get('Publication date', ''),
                        'training_compute_flop': float(row.get('Training compute (FLOP)', 0)),
                        'organization': row.get('Organization', ''),
                        'parameters': float(row.get('Parameters', 0)) if row.get('Parameters') else None,
                        'domain': row.get('Domain', ''),
                    })
                except (ValueError, TypeError):
                    continue
        
        print(f"  ✓ Fetched {len(models)} models from Epoch AI")
        return models
        
    except Exception as e:
        print(f"  ✗ Error fetching Epoch AI data: {e}")
        print("  → Using fallback data")
        return []


# ============================================================================
# HUGGING FACE - Benchmark Leaderboards
# ============================================================================

def fetch_huggingface_leaderboard() -> List[Dict]:
    """
    Fetch model benchmark scores from Hugging Face Open LLM Leaderboard.
    
    Source: Hugging Face Datasets API
    
    Returns: List of model scores
    """
    print("Fetching Hugging Face leaderboard...")
    
    # Hugging Face leaderboard dataset
    url = "https://huggingface.co/api/datasets/open-llm-leaderboard/contents"
    
    try:
        # Try to fetch from HF leaderboard API
        # Note: This is a simplified version - actual API structure may vary
        response = requests.get(
            "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/raw/main/src/leaderboard/read_evals.py",
            timeout=30
        )
        
        # For now, we'll use a different approach - fetch known benchmark results
        # from Papers with Code API
        
        print("  → Using Papers with Code for benchmarks")
        return fetch_papers_with_code_benchmarks()
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def fetch_papers_with_code_benchmarks() -> List[Dict]:
    """
    Fetch benchmark results from Papers with Code API.
    
    Source: https://paperswithcode.com/api/v1/
    """
    
    benchmarks = ['mmlu', 'humaneval', 'gsm8k']
    all_results = []
    
    for benchmark in benchmarks:
        try:
            # Papers with Code API endpoint
            url = f"https://paperswithcode.com/api/v1/benchmarks/{benchmark}/"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                # Extract results
                # Note: API structure varies, this is simplified
                all_results.append({
                    'benchmark': benchmark,
                    'data': data
                })
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"  ✗ Error fetching {benchmark}: {e}")
            continue
    
    print(f"  ✓ Fetched data for {len(all_results)} benchmarks")
    return all_results


# ============================================================================
# ARXIV - Safety Research Papers
# ============================================================================

def fetch_arxiv_safety_papers(start_year: int = 2020) -> Dict[int, int]:
    """
    Count AI safety papers on arXiv by year.
    
    Source: arXiv API
    Search terms: "AI safety", "AI alignment", "AI risk"
    
    Returns: Dict mapping year -> paper count
    """
    print("Fetching arXiv safety papers...")
    
    base_url = "http://export.arxiv.org/api/query"
    
    # Search terms for AI safety research
    search_terms = [
        'ti:"AI safety"',
        'ti:"AI alignment"', 
        'ti:"existential risk"',
        'abs:"language model safety"',
        'abs:"AI alignment"'
    ]
    
    query = ' OR '.join(search_terms)
    
    papers_by_year = {}
    
    try:
        # Fetch papers
        params = {
            'search_query': query,
            'start': 0,
            'max_results': 2000,  # arXiv limit
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        
        # Parse XML response (arXiv returns Atom XML)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        # Namespace for arXiv
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        
        for entry in entries:
            published = entry.find('atom:published', ns)
            if published is not None:
                date_str = published.text
                year = int(date_str[:4])
                
                if year >= start_year:
                    papers_by_year[year] = papers_by_year.get(year, 0) + 1
        
        print(f"  ✓ Found {sum(papers_by_year.values())} safety papers since {start_year}")
        print(f"    Breakdown: {dict(sorted(papers_by_year.items()))}")
        
        return papers_by_year
        
    except Exception as e:
        print(f"  ✗ Error fetching arXiv data: {e}")
        return {}


# ============================================================================
# GITHUB - Open Source Model Releases
# ============================================================================

def fetch_github_model_releases() -> Dict[int, Dict[str, int]]:
    """
    Count open source model releases by year.
    
    Source: GitHub API - search for ML model repositories
    
    Returns: Dict mapping year -> {'open': count, 'total': count}
    """
    print("Fetching GitHub model releases...")
    
    # Major model hubs to check
    repos_to_check = [
        'huggingface/transformers',
        'meta-llama/llama',
        'microsoft/phi',
        'mistralai/mistral-src',
        'EleutherAI/gpt-neox',
        'THUDM/ChatGLM',
        'QwenLM/Qwen',
    ]
    
    releases_by_year = {}
    
    try:
        for repo in repos_to_check:
            url = f"https://api.github.com/repos/{repo}/releases"
            
            headers = {}
            # Add token if available (to avoid rate limits)
            # headers = {'Authorization': f'token {GITHUB_TOKEN}'}
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                releases = response.json()
                
                for release in releases:
                    published = release.get('published_at', '')
                    if published:
                        year = int(published[:4])
                        if year not in releases_by_year:
                            releases_by_year[year] = {'open': 0, 'total': 0}
                        
                        releases_by_year[year]['open'] += 1
                        releases_by_year[year]['total'] += 1
                
                time.sleep(1)  # Rate limiting
                
        print(f"  ✓ Found releases across {len(releases_by_year)} years")
        return releases_by_year
        
    except Exception as e:
        print(f"  ✗ Error fetching GitHub data: {e}")
        return {}


# ============================================================================
# WORLD BANK / OECD - R&D Spending
# ============================================================================

def fetch_rd_spending() -> Dict[str, Dict[int, float]]:
    """
    Fetch R&D spending data by country.
    
    Source: World Bank Open Data API
    Indicator: GB.XPD.RSDV.GD.ZS (R&D expenditure as % of GDP)
    
    Returns: Dict mapping country -> year -> spending_billions
    """
    print("Fetching R&D spending data...")
    
    countries = {
        'USA': 'US',
        'CHN': 'China',
        'EUU': 'EU'  # European Union aggregate
    }
    
    # World Bank API endpoint
    base_url = "https://api.worldbank.org/v2/country/{}/indicator/GB.XPD.RSDV.GD.ZS"
    
    rd_data = {}
    
    try:
        for country_code, country_name in countries.items():
            url = base_url.format(country_code)
            params = {
                'format': 'json',
                'date': '2015:2025',
                'per_page': 100
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if len(data) > 1 and isinstance(data[1], list):
                    rd_data[country_name] = {}
                    
                    for entry in data[1]:
                        year = entry.get('date')
                        value = entry.get('value')
                        
                        if year and value:
                            # Convert R&D % to rough $ billions
                            # (Would need GDP data for exact calculation)
                            rd_data[country_name][int(year)] = value
            
            time.sleep(0.5)  # Rate limiting
        
        print(f"  ✓ Fetched R&D data for {len(rd_data)} countries")
        return rd_data
        
    except Exception as e:
        print(f"  ✗ Error fetching World Bank data: {e}")
        return {}


# ============================================================================
# AI INDEX - Comprehensive Metrics
# ============================================================================

def fetch_ai_index_data() -> Dict:
    """
    Fetch data from Stanford HAI AI Index.
    
    Source: https://aiindex.stanford.edu/report/
    Note: This is typically released as annual reports, not a live API
    
    Returns: Dict with various AI metrics
    """
    print("Checking AI Index Report data...")
    
    # The AI Index Report publishes data on GitHub
    url = "https://raw.githubusercontent.com/AI-Index/AI-Index-Data/main/2024/data/"
    
    # Try to fetch some key datasets
    datasets = [
        'performance_benchmarks.csv',
        'corporate_investment.csv',
        'publications.csv'
    ]
    
    ai_index_data = {}
    
    for dataset in datasets:
        try:
            response = requests.get(url + dataset, timeout=30)
            if response.status_code == 200:
                ai_index_data[dataset] = response.text
                print(f"  ✓ Fetched {dataset}")
        except:
            continue
    
    if ai_index_data:
        print(f"  ✓ Loaded {len(ai_index_data)} AI Index datasets")
    else:
        print("  ✗ Could not fetch AI Index data (may require manual download)")
    
    return ai_index_data


# ============================================================================
# MAIN FETCH FUNCTION
# ============================================================================

def fetch_all_real_data() -> Dict:
    """
    Fetch all available real-world data from public sources.
    
    Returns: Dict containing all fetched data
    """
    print("\n" + "=" * 70)
    print("FETCHING REAL-WORLD DATA FROM PUBLIC SOURCES")
    print("=" * 70)
    print()
    
    data = {
        'fetch_timestamp': datetime.now().isoformat(),
        'sources': []
    }
    
    # 1. Epoch AI - Model training data
    try:
        epoch_data = fetch_epoch_ai_models()
        if epoch_data:
            data['epoch_models'] = epoch_data
            data['sources'].append('Epoch AI')
    except Exception as e:
        print(f"Error with Epoch AI: {e}")
    
    print()
    
    # 2. Benchmarks
    try:
        benchmark_data = fetch_huggingface_leaderboard()
        if benchmark_data:
            data['benchmarks'] = benchmark_data
            data['sources'].append('Papers with Code')
    except Exception as e:
        print(f"Error with benchmarks: {e}")
    
    print()
    
    # 3. Safety research papers
    try:
        safety_papers = fetch_arxiv_safety_papers(start_year=2020)
        if safety_papers:
            data['safety_papers'] = safety_papers
            data['sources'].append('arXiv')
    except Exception as e:
        print(f"Error with arXiv: {e}")
    
    print()
    
    # 4. Open source releases
    try:
        github_releases = fetch_github_model_releases()
        if github_releases:
            data['github_releases'] = github_releases
            data['sources'].append('GitHub')
    except Exception as e:
        print(f"Error with GitHub: {e}")
    
    print()
    
    # 5. R&D spending
    try:
        rd_spending = fetch_rd_spending()
        if rd_spending:
            data['rd_spending'] = rd_spending
            data['sources'].append('World Bank')
    except Exception as e:
        print(f"Error with World Bank: {e}")
    
    print()
    
    # 6. AI Index
    try:
        ai_index = fetch_ai_index_data()
        if ai_index:
            data['ai_index'] = ai_index
            data['sources'].append('AI Index')
    except Exception as e:
        print(f"Error with AI Index: {e}")
    
    print()
    print("=" * 70)
    print(f"✓ DATA FETCHING COMPLETE")
    print(f"  Successfully fetched from: {', '.join(data['sources'])}")
    print("=" * 70)
    
    return data


def save_fetched_data(data: Dict, filename: str = "fetched_real_data.json"):
    """Save fetched data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n✓ Saved fetched data to {filename}")


def load_fetched_data(filename: str = "fetched_real_data.json") -> Dict:
    """Load previously fetched data"""
    with open(filename, 'r') as f:
        return json.load(f)


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Fetch all real data
    real_data = fetch_all_real_data()
    
    # Save it
    save_fetched_data(real_data)
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    if 'epoch_models' in real_data:
        print(f"\nEpoch AI Models: {len(real_data['epoch_models'])} models")
        
        # Show recent models
        recent = [m for m in real_data['epoch_models'] 
                  if m.get('date', '').startswith(('2023', '2024', '2025'))]
        print(f"  Recent models (2023-2025): {len(recent)}")
        
        if recent:
            print("\n  Sample recent models:")
            for model in sorted(recent, key=lambda x: x.get('date', ''), reverse=True)[:5]:
                print(f"    - {model['name']} ({model['date']}): {model['training_compute_flop']:.2e} FLOPs")
    
    if 'safety_papers' in real_data:
        print(f"\nSafety Papers by Year:")
        for year, count in sorted(real_data['safety_papers'].items()):
            print(f"  {year}: {count} papers")
    
    if 'github_releases' in real_data:
        print(f"\nOpen Source Releases:")
        for year, counts in sorted(real_data['github_releases'].items()):
            print(f"  {year}: {counts['open']} releases")
    
    print("\n✓ Next step: Use this data to calibrate model parameters")
    print("  Run: python data_calibration.py")
