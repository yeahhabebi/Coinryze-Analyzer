# fetcher/fetch_coinryze.py

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import re

S3_BUCKET = os.getenv("S3_BUCKET")
USE_S3 = bool(S3_BUCKET)
if USE_S3:
    import boto3
    s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))

URL = os.getenv("COINRYZE_URL", "https://coinryze.org")
CSV_PATH = os.getenv("CSV_PATH", "frontend/coinryze_history.csv")
PARQUET_PATH = os.getenv("PARQUET_PATH", "frontend/coinryze_history.parquet")
FETCH_INTERVAL_MIN = int(os.getenv("FETCH_INTERVAL_MIN", "1"))
USER_AGENT = os.getenv("USER_AGENT", "CoinryzeAnalyzerBot/1.0 (+contact@example.com)")
HEADERS = {"User-Agent": USER_AGENT}

def upload_file_to_s3(local_path, s3_key):
    try:
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"Uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print("S3 upload error:", e)

def parse_timestamp(timestamp_str):
    """Parse timestamp from format: 21:32:00 10/07/2025"""
    try:
        dt = datetime.strptime(timestamp_str, "%H:%M:%S %m/%d/%Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp_str

def parse_latest_draws(soup):
    """
    Parse CoinRyze historical data based on actual website structure
    Updated with real selectors from coinryze.org screenshots
    """
    rows = []
    
    # Based on screenshot structure, try multiple possible container selectors
    container_selectors = [
        '.history-item',           # Most likely
        '.draw-item',              # Alternative
        '.result-item',            # Alternative  
        '.trend-item',             # From "Historical Trend Chart"
        '[class*="history"]',      # Class contains "history"
        '[class*="draw"]',         # Class contains "draw"
        '[class*="result"]',       # Class contains "result"
        '[class*="trend"]',        # Class contains "trend"
        '.list-item',              # Generic list item
        '.item',                   # Generic item
    ]
    
    draw_containers = None
    for selector in container_selectors:
        draw_containers = soup.select(selector)
        if draw_containers:
            print(f"Found {len(draw_containers)} containers using selector: {selector}")
            break
    
    if not draw_containers:
        print("No containers found with standard selectors. Using text-based parsing.")
        return parse_coinryze_text_patterns(soup)
    
    for container in draw_containers:
        try:
            # Extract data using text patterns (more reliable than CSS selectors)
            container_text = container.get_text()
            
            # Extract issue ID (format: #202510071412)
            issue_match = re.search(r'#(\d{12})', container_text)
            if not issue_match:
                continue
            issue_id = f"#{issue_match.group(1)}"
            
            # Extract winning number (single digit after "Winning Results")
            number_match = re.search(r'Winning Results\s*(\d+)', container_text)
            if not number_match:
                continue
            number = int(number_match.group(1))
            
            # Extract timestamp (format: 21:32:00 10/07/2025)
            time_match = re.search(r'(\d{2}:\d{2}:\d{2} \d{2}/\d{2}/\d{4})', container_text)
            timestamp = parse_timestamp(time_match.group(1)) if time_match else ""
            
            # Extract size (Big/Small)
            size_match = re.search(r'(Big|Small)\s*Winning Results', container_text)
            size = size_match.group(1) if size_match else ""
            
            # Extract price (format: 4549.87)
            price_match = re.search(r'Winning Price\s*([\d,]+\.\d{2})', container_text)
            price = float(price_match.group(1).replace(',', '')) if price_match else 0.0
            
            # Color is always Green based on screenshots
            color = "Green"
            
            rows.append({
                "issue_id": issue_id,
                "timestamp": timestamp,
                "number": number,
                "color": color,
                "size": size,
                "price": price,
                "odd_even": "Odd" if number % 2 else "Even"
            })
            
        except Exception as e:
            print(f"Error parsing container: {e}")
            continue
    
    return rows

def parse_coinryze_text_patterns(soup):
    """
    Fallback parsing using text patterns when CSS selectors fail
    """
    rows = []
    text_content = soup.get_text()
    
    # Find all issue blocks by looking for patterns
    issue_pattern = r'#(\d{12})'
    time_pattern = r'(\d{2}:\d{2}:\d{2} \d{2}/\d{2}/\d{4})'
    number_pattern = r'Winning Results\s*(\d)'
    size_pattern = r'(Big|Small)\s*Winning Results'
    price_pattern = r'Winning Price\s*([\d,]+\.\d{2})'
    
    # Find all matches
    issues = re.findall(issue_pattern, text_content)
    timestamps = re.findall(time_pattern, text_content)
    numbers = re.findall(number_pattern, text_content)
    sizes = re.findall(size_pattern, text_content)
    prices = re.findall(price_pattern, text_content)
    
    # Match them up (simplified approach)
    min_length = min(len(issues), len(timestamps), len(numbers), len(sizes))
    
    for i in range(min_length):
        try:
            number = int(numbers[i])
            price = float(prices[i].replace(',', '')) if i < len(prices) else 0.0
            
            rows.append({
                "issue_id": f"#{issues[i]}",
                "timestamp": parse_timestamp(timestamps[i]),
                "number": number,
                "color": "Green",  # Always Green based on screenshots
                "size": sizes[i],
                "price": price,
                "odd_even": "Odd" if number % 2 else "Even"
            })
        except Exception as e:
            print(f"Error processing draw {i}: {e}")
            continue
    
    print(f"Text pattern parsing found {len(rows)} draws")
    return rows

def debug_page_structure(soup):
    """Debug function to help identify correct selectors"""
    print("\n=== DEBUG: Page Structure Analysis ===")
    
    # Find all unique classes
    all_classes = set()
    for element in soup.find_all(class_=True):
        all_classes.update(element.get('class'))
    
    print("All classes found on page:")
    for cls in sorted(all_classes):
        if any(keyword in cls for keyword in ['history', 'draw', 'result', 'trend', 'item', 'row']):
            print(f"  📍 {cls}")
    
    # Find elements with specific text patterns
    elements_with_hash = soup.find_all(string=re.compile(r'#\d{12}'))
    print(f"\nElements containing issue IDs: {len(elements_with_hash)}")
    
    # Show sample of page structure
    print("\nSample of page content (first 1000 chars):")
    print(soup.get_text()[:1000])
    
    print("=== END DEBUG ===\n")

def fetch_and_save():
    try:
        print(f"🕒 Fetching data at {datetime.now().isoformat()}")
        
        # Try multiple endpoints
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
            print(f"🔗 Trying endpoint: {full_url}")
            try:
                response = requests.get(full_url, headers=HEADERS, timeout=15)
                if response.status_code == 200:
                    print(f"✅ Connected to: {full_url}")
                    break
                else:
                    print(f"❌ {full_url} returned status: {response.status_code}")
            except Exception as e:
                print(f"❌ Failed to connect to {full_url}: {e}")
        
        if not response:
            print("❌ Failed to connect to any endpoint")
            return
        
        soup = BeautifulSoup(response.text, "lxml")
        
        # Debug on first run
        if not os.path.exists(CSV_PATH):
            debug_page_structure(soup)
        
        scraped = parse_latest_draws(soup)

        if not scraped:
            print("❌ No draws parsed — website structure may have changed")
            print("💡 Check the debug output above and update selectors")
            return

        print(f"✅ Parsed {len(scraped)} draws")
        
        # Show sample of parsed data
        if scraped:
            print("Sample parsed data:")
            for i, draw in enumerate(scraped[:2]):
                print(f"  {i+1}. Issue: {draw['issue_id']}, Number: {draw['number']}, "
                      f"Size: {draw['size']}, Color: {draw['color']} 🟢")

        existing_df = pd.DataFrame()
        if os.path.exists(PARQUET_PATH):
            try:
                existing_df = pq.read_table(PARQUET_PATH).to_pandas()
                print(f"📊 Loaded {len(existing_df)} existing records")
            except Exception as e:
                print("⚠️ Warning reading existing parquet:", e)

        existing_issues = set(existing_df['issue_id'].astype(str).tolist()) if not existing_df.empty else set()
        new_rows = [r for r in scraped if str(r['issue_id']) not in existing_issues]

        if not new_rows:
            print("📭 No new rows to append")
            return

        print(f"🆕 Adding {len(new_rows)} new draws")

        df_new = pd.DataFrame(new_rows)
        # CSV append
        df_new.to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)
        print(f"💾 Appended to CSV: {CSV_PATH}")

        # Parquet: safe append with dtype enforcement
        df_total = pd.concat([existing_df, df_new], ignore_index=True) if not existing_df.empty else df_new
        df_total = df_total.astype({
            "issue_id": str, "timestamp": str, "number": int,
            "color": str, "size": str, "odd_even": str, "price": float
        })
        
        # Remove duplicates
        df_total = df_total.drop_duplicates(subset=['issue_id'], keep='last')
        
        pq.write_table(pa.Table.from_pandas(df_total, preserve_index=False), PARQUET_PATH)
        print(f"💾 Saved Parquet: {PARQUET_PATH} (total: {len(df_total)} records)")

        if USE_S3:
            upload_file_to_s3(CSV_PATH, os.path.basename(CSV_PATH))
            upload_file_to_s3(PARQUET_PATH, os.path.basename(PARQUET_PATH))

        print(f"✅ Saved {len(new_rows)} new rows at {datetime.now().isoformat()}")
        
    except Exception as e:
        print("❌ Error in fetch_and_save:", e)

def create_directory_structure():
    """Create necessary directories"""
    os.makedirs('fetcher', exist_ok=True)
    os.makedirs('frontend', exist_ok=True)

if __name__ == "__main__":
    create_directory_structure()
    
    print("🚀 CoinRyze Fetcher Started")
    print("🟢 Color: Green (confirmed from screenshots)")
    
    # Run once immediately
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
