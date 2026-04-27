#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# monitor_upload.sh — Check Kaggle upload progress
# Usage: zsh monitor_upload.sh
# ─────────────────────────────────────────────────────────────────────────────

LOG=/tmp/kaggle_upload.log
TOTAL_FILES=232

echo "═══════════════════════════════════════════════════════"
echo "  Kaggle Upload Monitor — Yale Brain Mets NIfTI"
echo "═══════════════════════════════════════════════════════"

# Check process
PID=$(ps aux | grep "kaggle datasets create" | grep -v grep | awk '{print $2}' | head -1)
if [[ -n "$PID" ]]; then
    echo "  Status  : 🔄 RUNNING (PID $PID)"
else
    echo "  Status  : ✅ COMPLETED or ❌ STOPPED"
fi

# Parse log
if [[ -f "$LOG" ]]; then
    python3 -c "
import re, sys
log = open('$LOG').read()
sizes = re.findall(r'Upload successful: .+? \((\d+)MB\)', log)
n = len(sizes)
total_mb = sum(int(s) for s in sizes)
avg_mb = total_mb / n if n > 0 else 46
total_est = avg_mb * $TOTAL_FILES
remaining = total_est - total_mb
secs = remaining / 6
pct = n / $TOTAL_FILES * 100

bar_filled = int(pct / 5)
bar = '█' * bar_filled + '░' * (20 - bar_filled)

print(f'  Files   : {n}/$TOTAL_FILES  [{bar}]  {pct:.0f}%')
print(f'  Uploaded: {total_mb:.0f} MB  /  ~{total_est:.0f} MB (~{total_est/1024:.1f} GB)')
print(f'  Remaining: ~{int(secs//60)} min at 6 MB/s avg')

# Last file being uploaded
lines = [l.strip() for l in log.split('\n') if l.strip()]
for line in reversed(lines):
    if 'Starting upload' in line or '%|' in line or 'Upload successful' in line:
        print(f'  Latest  : {line[:70]}')
        break

# Check if done
if 'Dataset URL:' in log or 'Dataset version is being processed' in log:
    import re
    url = re.search(r'https://www.kaggle.com/datasets/\S+', log)
    if url:
        print(f'')
        print(f'  ✅ UPLOAD COMPLETE!')
        print(f'  URL: {url.group()}')
"
else
    echo "  Log not found: $LOG"
fi

echo "═══════════════════════════════════════════════════════"
echo ""
echo "  To watch live: tail -f $LOG"
echo "  To kill upload: kill \$PID"
