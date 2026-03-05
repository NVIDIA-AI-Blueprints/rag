#!/usr/bin/env python3
"""
View ingestion history from Kafka consumer
"""
import json
import sys
from datetime import datetime
from pathlib import Path

HISTORY_FILE = '/tmp/ingestion_history.jsonl'

def format_duration(seconds):
    """Format duration nicely"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"

def view_history(history_file=HISTORY_FILE, limit=50):
    """Display ingestion history"""
    
    if not Path(history_file).exists():
        print(f"❌ History file not found: {history_file}")
        print("No ingestion history yet.")
        return
    
    records = []
    with open(history_file, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except:
                pass
    
    if not records:
        print("No records found in history file.")
        return
    
    # Show last N records
    records = records[-limit:]
    
    print(f"\n📊 INGESTION HISTORY (last {len(records)} records)\n")
    print("=" * 140)
    print(f"{'File Name':<35} {'Bucket':<18} {'Collection':<18} {'Duration':<12} {'Status':<15} {'Time':<20}")
    print("=" * 140)
    
    for record in records:
        filename = record.get('file_name', 'N/A')
        bucket = record.get('bucket', 'N/A')
        collection = record.get('collection', 'N/A')
        start_time = record.get('start_time', '')
        end_time = record.get('end_time', '')
        duration = record.get('duration_seconds', 0)
        status = record.get('status', 'UNKNOWN')
        error = record.get('error_message', '')
        
        # Format times
        try:
            start_dt = datetime.fromisoformat(start_time)
            time_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            time_str = start_time[:19] if len(start_time) > 19 else start_time
        
        # Status emoji
        status_display = {
            'SUCCESS': '✅ SUCCESS',
            'FAILED': '❌ FAILED',
            'DELETED': '🗑️  DELETED'
        }.get(status, f'⚠️  {status}')
        
        # Truncate long fields
        if len(filename) > 33:
            filename = filename[:30] + '...'
        if len(bucket) > 16:
            bucket = bucket[:13] + '...'
        if len(collection) > 16:
            collection = collection[:13] + '...'
        
        print(f"{filename:<35} {bucket:<18} {collection:<18} {format_duration(duration):<12} {status_display:<15} {time_str:<20}")
        
        if error and status == 'FAILED':
            error_short = error[:110] + '...' if len(error) > 110 else error
            print(f"  └─ Error: {error_short}")
    
    print("=" * 140)
    
    # Summary stats
    total = len(records)
    success = sum(1 for r in records if r.get('status') == 'SUCCESS')
    failed = sum(1 for r in records if r.get('status') == 'FAILED')
    deleted = sum(1 for r in records if r.get('status') == 'DELETED')
    avg_duration = sum(r.get('duration_seconds', 0) for r in records) / total if total > 0 else 0
    
    print(f"\n📈 Summary:")
    print(f"   Total: {total} | ✅ Success: {success} | ❌ Failed: {failed} | 🗑️  Deleted: {deleted}")
    print(f"   Average Duration: {format_duration(avg_duration)}")
    print()

if __name__ == '__main__':
    history_file = sys.argv[1] if len(sys.argv) > 1 else HISTORY_FILE
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    view_history(history_file, limit)

