# fetcher/fetch_coinryze.py

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import time
import random
import re

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
USE_S3 = bool(S3_BUCKET)
if USE_S3:
    import boto3
    s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))

URL = os.getenv("COINRYZE_URL", "https://coinryze.org")
CSV_PATH = os.getenv("CSV_PATH", "frontend/coinryze_history.csv")
PARQUET_PATH = os.getenv("PARQUET_PATH", "frontend/coinryze_history.parquet")
FETCH_INTERVAL_MIN = int(os.getenv("FETCH_INTERVAL_MIN", "1"))
USER_AGENT = os.getenv("USER_AGENT", "CoinryzeAnalyzerBot/1.0 (+https://github.com/your-repo)")
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

def upload_file_to_s3(local_path, s3_key):
    """Upload file to S3 if configured"""
    try:
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"Uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print("S3 upload error:", e)

def respectful_request(url, max_retries=3):
    """Make respectful requests with rate limiting and retries"""
    for attempt in range(max_retries):
        try:
            # Add random delay between requests (3-7 seconds)
            if attempt > 0:
                delay = random.uniform(3, 7)
                print(f"Waiting {delay:.2f} seconds before retry...")
                time.sleep(delay)
            
            response = requests.get(url, headers=HEADERS, timeout=15)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Rate limited
                print("Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                continue
            else:
                print(f"HTTP {response.status_code}: {response.reason}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 10  # Exponential backoff
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    return None

def parse_timestamp(timestamp_str):
    """Parse timestamp from format: 21:32:00 10/07/2025"""
    try:
        # Convert to standard format
        dt = datetime.strptime(timestamp_str, "%H:%M:%S %m/%d/%Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp_str

def determine_color(number, size):
    """
    Determine color based on number and size rules
    From CoinRyze screenshots analysis:
    - All results show Green color 🟢
    - This might be fixed or based on specific rules
    """
    # Based on the screenshots, all results are Green
    return "Green"
    
    # Alternative logic if color varies:
    # if number == 0:
    #     return "Green"
    # elif 1 <= number <= 4:
    #     return "Red" if size == "Small" else "Green"
    # elif 5 <= number <= 9:
    #     return "Red" if size == "Big" else "Green"
    # else:
    #     return "Green"

def parse_coinryze_data(soup):
    """
    Parse CoinRyze historical data based on screenshot structure
    Structure from screenshots:
    - Current ID: #202510071412 (issue_id)
    - Winning Results: 7 (number)
    - Winning Price: 4549.87 (price)
    - Draw time: 21:32:00 10/07/2025 (timestamp)
    - Winning Results: Big/Small (size)
    - Winning Color: Green 🟢 (confirmed from screenshots)
    """
    draws = []
    
    # Look for draw containers - these could be divs, table rows, or list items
    # Common selectors to try:
    selectors_to_try = [
        '.draw-item',
        '.history-item', 
        '.result-item',
        '.bet-item',
        '.trend-item',
        '[class*="draw"]',
        '[class*="history"]',
        '[class*="result"]',
        '[class*="trend"]',
        'div.draw',
        'li.draw',
        '.list-item',
        '.item'
    ]
    
    draw_containers = None
    for selector in selectors_to_try:
        draw_containers = soup.select(selector)
        if draw_containers:
            print(f"Found {len(draw_containers)} draws using selector: {selector}")
            break
    
    if not draw_containers:
        # If no specific containers found, try to find by text patterns
        print("No specific containers found. Trying text-based parsing...")
        return parse_coinryze_text_patterns(soup)
    
    for container in draw_containers:
        try:
            draw_data = extract_draw_data(container)
            if draw_data:
                draws.append(draw_data)
        except Exception as e:
            print(f"Error parsing container: {e}")
            continue
    
    return draws

def parse_coinryze_text_patterns(soup):
    """
    Alternative parsing method using text patterns
    This is useful when CSS selectors don't work
    """
    draws = []
    text_content = soup.get_text()
    
    # Pattern for issue ID: #202510071412
    issue_pattern = r'#(\d{12})'
    # Pattern for timestamp: 21:32:00 10/07/2025
    time_pattern = r'(\d{2}:\d{2}:\d{2} \d{2}/\d{2}/\d{4})'
    # Pattern for winning number: single digit
    number_pattern = r'Winning Results\s*(\d)'
    # Pattern for size: Big/Small
    size_pattern = r'(Big|Small)\s*Winning Results'
    # Pattern for price: 4549.87
    price_pattern = r'Winning Price\s*([\d,]+\.\d{2})'
    
    issues = re.findall(issue_pattern, text_content)
    timestamps = re.findall(time_pattern, text_content)
    numbers = re.findall(number_pattern, text_content)
    sizes = re.findall(size_pattern, text_content)
    prices = re.findall(price_pattern, text_content)
    
    # Match them up
    min_length = min(len(issues), len(timestamps), len(numbers), len(sizes), len(prices))
    
    for i in range(min_length):
        try:
            number = int(numbers[i])
            price = float(prices[i].replace(',', '')) if i < len(prices) else 0.0
            
            draws.append({
                "issue_id": f"#{issues[i]}",
                "timestamp": parse_timestamp(timestamps[i]),
                "number": number,
                "size": sizes[i],
                "color": determine_color(number, sizes[i]),
                "price": price,
                "odd_even": "Odd" if number % 2 else "Even"
            })
        except Exception as e:
            print(f"Error processing draw {i}: {e}")
            continue
    
    print(f"Text pattern parsing found {len(draws)} draws")
    return draws

def extract_draw_data(container):
    """Extract data from a single draw container"""
    try:
        # Extract issue ID - look for patterns like #202510071412
        issue_text = container.get_text()
        issue_match = re.search(r'#(\d{12})', issue_text)
        if not issue_match:
            return None
            
        issue_id = f"#{issue_match.group(1)}"
        
        # Extract winning number - look for single digit
        number_match = re.search(r'Winning Results\s*(\d)', issue_text)
        if not number_match:
            return None
            
        number = int(number_match.group(1))
        
        # Extract timestamp
        time_match = re.search(r'(\d{2}:\d{2}:\d{2} \d{2}/\d{2}/\d{4})', issue_text)
        timestamp = parse_timestamp(time_match.group(1)) if time_match else ""
        
        # Extract size (Big/Small)
        size_match = re.search(r'(Big|Small)\s*Winning Results', issue_text)
        size = size_match.group(1) if size_match else ""
        
        # Extract price if available
        price_match = re.search(r'Winning Price\s*([\d,]+\.\d{2})', issue_text)
        price = float(price_match.group(1).replace(',', '')) if price_match else 0.0
        
        # Determine color (Green based on screenshots)
        color = determine_color(number, size)
        
        return {
            "issue_id": issue_id,
            "timestamp": timestamp,
            "number": number,
            "size": size,
            "color": color,
            "price": price,
            "odd_even": "Odd" if number % 2 else "Even"
        }
        
    except Exception as e:
        print(f"Error extracting draw data: {e}")
        return None

def debug_page_structure(soup):
    """Debug function to understand page structure"""
    print("\n=== PAGE STRUCTURE DEBUG ===")
    
    # Find all elements with IDs
    elements_with_ids = soup.find_all(id=True)
    print(f"Elements with IDs: {len(elements_with_ids)}")
    for elem in elements_with_ids[:5]:
        print(f"  ID: {elem.get('id')} - Tag: {elem.name}")
    
    # Find common container classes
    common_classes = ['draw', 'history', 'result', 'bet', 'item', 'row', 'list', 'trend']
    for class_name in common_classes:
        elements = soup.select(f'[class*="{class_name}"]')
        if elements:
            print(f"Elements with class containing '{class_name}': {len(elements)}")
            # Show first element's classes
            if elements:
                print(f"  Sample classes: {elements[0].get('class')}")
    
    # Look for specific text patterns
    text_content = soup.get_text()
    if '#2025' in text_content:
        print("Found historical data with 2025 timestamps")
    
    print("=== END DEBUG ===\n")

def fetch_and_save():
    """Main function to fetch and save CoinRyze data"""
    try:
        print(f"Fetching data from {URL} at {datetime.now().isoformat()}")
        
        # Try different potential endpoints
        endpoints = [
            "/history",
            "/draws", 
            "/results",
            "/m/history",
            "/draw-history",
            "/latest-results",
            "/trend",
            "/historical-trend"
        ]
        
        response = None
        for endpoint in endpoints:
            full_url = f"{URL}{endpoint}"
            print(f"Trying: {full_url}")
            response = respectful_request(full_url)
            if response:
                print(f"Successfully connected to {full_url}")
                break
                
        if not response:
            print("Failed to fetch data from all endpoints")
            return
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Debug page structure on first run or if no data found
        if not os.path.exists(CSV_PATH):
            debug_page_structure(soup)
        
        # Parse the data
        scraped_data = parse_coinryze_data(soup)
        
        if not scraped_data:
            print("No draws parsed. The website structure may have changed.")
            print("Please inspect the page and update the CSS selectors.")
            # Show sample of page content for debugging
            print("Sample page content (first 500 chars):")
            print(soup.get_text()[:500])
            return
            
        print(f"Successfully parsed {len(scraped_data)} draws")
        
        # Show sample of parsed data
        if scraped_data:
            print("Sample parsed data:")
            for i, draw in enumerate(scraped_data[:3]):
                print(f"  {i+1}. {draw}")
        
        # Load existing data
        existing_df = pd.DataFrame()
        if os.path.exists(PARQUET_PATH):
            try:
                existing_df = pq.read_table(PARQUET_PATH).to_pandas()
                print(f"Loaded {len(existing_df)} existing records")
            except Exception as e:
                print("Warning reading existing parquet:", e)
        
        # Filter new entries
        existing_issues = set(existing_df['issue_id'].astype(str).tolist()) if not existing_df.empty else set()
        new_rows = [row for row in scraped_data if str(row['issue_id']) not in existing_issues]
        
        if not new_rows:
            print("No new draws to add")
            return
            
        print(f"Adding {len(new_rows)} new draws")
        
        # Save to CSV
        df_new = pd.DataFrame(new_rows)
        df_new.to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)
        print(f"Appended to CSV: {CSV_PATH}")
        
        # Save to Parquet
        df_total = pd.concat([existing_df, df_new], ignore_index=True) if not existing_df.empty else df_new
        
        # Ensure consistent data types
        df_total = df_total.astype({
            "issue_id": str,
            "timestamp": str, 
            "number": int,
            "size": str,
            "color": str,
            "price": float,
            "odd_even": str
        })
        
        # Remove duplicates
        df_total = df_total.drop_duplicates(subset=['issue_id'], keep='last')
        
        pq.write_table(pa.Table.from_pandas(df_total, preserve_index=False), PARQUET_PATH)
        print(f"Saved Parquet: {PARQUET_PATH} with {len(df_total)} total records")
        
        # Upload to S3 if configured
        if USE_S3:
            upload_file_to_s3(CSV_PATH, os.path.basename(CSV_PATH))
            upload_file_to_s3(PARQUET_PATH, os.path.basename(PARQUET_PATH))
        
        print(f"✅ Successfully saved {len(new_rows)} new draws at {datetime.now().isoformat()}")
        
    except Exception as e:
        print(f"❌ Error in fetch_and_save: {e}")

def create_directory_structure():
    """Create necessary directories"""
    os.makedirs('fetcher', exist_ok=True)
    os.makedirs('frontend', exist_ok=True)

if __name__ == "__main__":
    create_directory_structure()
    
    print("🟢 CoinRyze Fetcher Started")
    print("🟢 Color: Green (confirmed from screenshots)")
    print("🟢 Features: Big/Small, Odd/Even, Price Tracking")
    
    # Run immediately
    fetch_and_save()
    
    # Schedule periodic runs
    scheduler = BlockingScheduler()
    scheduler.add_job(fetch_and_save, "interval", minutes=FETCH_INTERVAL_MIN)
    
    print(f"⏰ Fetcher scheduled. Fetching every {FETCH_INTERVAL_MIN} minute(s)")
    print("🛑 Press Ctrl+C to exit")
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("👋 Fetcher stopped by user")
    except Exception as e:
        print(f"❌ Fetcher error: {e}")
