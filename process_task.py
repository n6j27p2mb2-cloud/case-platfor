"""Background processing script — runs as a separate process to avoid GIL blocking."""
import sys
import json
import traceback
from pathlib import Path
from analyzer import parse_excel, analyze, compute_comparisons, generate_monthly_summary

def main():
    # Args: output_json_path cur_xlsx [prev_xlsx...] [--hist hist_xlsx] [--owner name]
    args = sys.argv[1:]
    output_path = Path(args[0])
    cur_path = Path(args[1])

    prev_paths = []
    hist_path = None
    owner_name = None
    i = 2
    while i < len(args):
        if args[i] == '--hist' and i + 1 < len(args):
            hist_path = Path(args[i + 1])
            i += 2
        elif args[i] == '--owner' and i + 1 < len(args):
            owner_name = args[i + 1]
            i += 2
        else:
            prev_paths.append(Path(args[i]))
            i += 1

    try:
        # Parse current month
        df, csat_df, metrics = parse_excel(str(cur_path))
        current_stats = analyze(df, csat_df, metrics)

        # Parse previous months
        prev_stats_list = []
        prev_filenames = [p.name for p in prev_paths]
        for pp in prev_paths:
            pdf, pcsat, pmetrics = parse_excel(str(pp))
            prev_stats_list.append(analyze(pdf, pcsat, pmetrics))

        # Parse historical
        historical_data = None
        if hist_path and hist_path.exists():
            try:
                from analyzer import parse_historical
                historical_data = parse_historical(str(hist_path))
            except Exception:
                pass

        # Comparisons
        current_stats = compute_comparisons(current_stats, prev_stats_list)

        # Owner filter
        if owner_name:
            df2, csat_df2, metrics2 = parse_excel(str(cur_path))
            if 'Owner' in df2.columns:
                df2 = df2[df2['Owner'].str.contains(owner_name, na=False, case=False)]
            elif 'Individual' in df2.columns:
                df2 = df2[df2['Individual'].str.contains(owner_name, na=False, case=False)]
            current_stats = analyze(df2, csat_df2, metrics2)
            if prev_paths:
                prev_owner = []
                for pp in prev_paths:
                    pdf2, pcsat2, pm2 = parse_excel(str(pp))
                    if 'Owner' in pdf2.columns:
                        pdf2 = pdf2[pdf2['Owner'].str.contains(owner_name, na=False, case=False)]
                    elif 'Individual' in pdf2.columns:
                        pdf2 = pdf2[pdf2['Individual'].str.contains(owner_name, na=False, case=False)]
                    prev_owner.append(analyze(pdf2, pcsat2, pm2))
                current_stats = compute_comparisons(current_stats, prev_owner)

        # Summary
        current_stats['monthly_summary'] = generate_monthly_summary(current_stats)

        # Serialize — only keep JSON-safe fields
        result = {
            'stats': current_stats,
            'filename': cur_path.name,
            'prev_filenames': prev_filenames,
            'has_comparison': current_stats.get('has_comparison', False),
            'owner_name': owner_name,
            'historical': historical_data,
        }

        output_path.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding='utf-8')
    except Exception as e:
        output_path.write_text(json.dumps({'error': f'{e}\n\n{traceback.format_exc()}'}, ensure_ascii=False), encoding='utf-8')
        sys.exit(1)

if __name__ == '__main__':
    main()
