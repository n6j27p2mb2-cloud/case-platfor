"""
Background worker — runs as subprocess, completely isolated from gunicorn.
Writes results to JSON files that the Flask app reads.
"""
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta

RESULT_DIR = Path(__file__).parent / 'results'
RESULT_DIR.mkdir(exist_ok=True)

def log(msg):
    log_file = Path(__file__).parent / 'worker.log'
    with open(log_file, 'a') as f:
        f.write(f'[{datetime.now().isoformat()}] {msg}\n')


def update_status(task_id, status, analysis_id='', error=''):
    status_file = RESULT_DIR / f'{task_id}_status.json'
    status_file.write_text(json.dumps({
        'status': status,
        'analysis_id': analysis_id,
        'error': error,
    }), ensure_ascii=False)


def main():
    if len(sys.argv) < 3:
        log('Usage: process_worker.py <task_id> <cur_path> [prev_paths...] [--hist <path>] [--owner <name>]')
        sys.exit(1)

    task_id = sys.argv[1]
    cur_path = sys.argv[2]

    # Parse optional args
    prev_paths = []
    hist_path = None
    owner_name = None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--hist' and i + 1 < len(sys.argv):
            hist_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--owner' and i + 1 < len(sys.argv):
            owner_name = sys.argv[i + 1]
            i += 2
        else:
            prev_paths.append(sys.argv[i])
            i += 1

    update_status(task_id, 'processing')
    log(f'{task_id}: Start — cur={cur_path}')

    try:
        from analyzer import parse_excel, analyze, compute_comparisons, generate_monthly_summary, generate_analysis_line

        df, csat_df, metrics, csat_quality_raw = parse_excel(cur_path)
        log(f'{task_id}: Parsed {len(df)} rows')

        current_stats = analyze(df, csat_df, metrics, csat_quality_raw)
        log(f'{task_id}: Analyzed OK')

        prev_stats_list = []
        for ppath in prev_paths:
            pdf, pcsat, pmetrics, pquality = parse_excel(ppath)
            prev_stats_list.append(analyze(pdf, pcsat, pmetrics, pquality))

        historical_data = None
        if hist_path:
            try:
                from analyzer import parse_historical
                historical_data = parse_historical(hist_path)
            except Exception:
                pass

        current_stats = compute_comparisons(current_stats, prev_stats_list)

        if owner_name:
            df2, csat_df2, metrics2, quality2 = parse_excel(cur_path)
            if 'Owner' in df2.columns:
                df2 = df2[df2['Owner'].str.contains(owner_name, na=False, case=False)]
            elif 'Individual' in df2.columns:
                df2 = df2[df2['Individual'].str.contains(owner_name, na=False, case=False)]
            current_stats = analyze(df2, csat_df2, metrics2, quality2)
            if prev_paths:
                prev_owner = []
                for ppath in prev_paths:
                    pdf2, pcsat2, pm2, pq2 = parse_excel(ppath)
                    if 'Owner' in pdf2.columns:
                        pdf2 = pdf2[pdf2['Owner'].str.contains(owner_name, na=False, case=False)]
                    elif 'Individual' in pdf2.columns:
                        pdf2 = pdf2[pdf2['Individual'].str.contains(owner_name, na=False, case=False)]
                    prev_owner.append(analyze(pdf2, pcsat2, pm2, pq2))
                current_stats = compute_comparisons(current_stats, prev_owner)

        try:
            current_stats['monthly_summary'] = generate_monthly_summary(current_stats)
        except Exception as e:
            current_stats['monthly_summary'] = f'Summary failed: {e}'

        try:
            end_str = current_stats.get('date_range', {}).get('end', '')
            end_dt = datetime.strptime(end_str, '%Y-%m-%d')
            ll = (end_dt.replace(day=1) - timedelta(days=1)).strftime("%b'%y")
        except Exception:
            ll = ''
        try:
            current_stats['analysis_summary_line'] = generate_analysis_line(current_stats, ll)
        except Exception:
            current_stats['analysis_summary_line'] = ''

        cur_filename = Path(cur_path).name
        # Get original filenames from paths
        prev_filenames = [Path(p).name for p in prev_paths]

        analysis = {
            'stats': current_stats,
            'filename': cur_filename,
            'prev_filenames': prev_filenames,
            'has_comparison': current_stats.get('has_comparison', False),
            'owner_name': owner_name or '',
            'historical': historical_data or {},
        }

        result_file = RESULT_DIR / f'{task_id}_analysis.json'
        result_file.write_text(json.dumps(analysis, ensure_ascii=False, default=str))

        update_status(task_id, 'done', analysis_id=task_id)
        log(f'{task_id}: Done')

    except Exception as e:
        tb = traceback.format_exc()
        log(f'{task_id}: {tb}')
        update_status(task_id, 'error', error=str(e)[:500])
    finally:
        # Clean up uploaded files
        for p in [cur_path] + prev_paths + ([hist_path] if hist_path else []):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == '__main__':
    main()
