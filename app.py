import os
import uuid
import traceback
from functools import wraps
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from dotenv import load_dotenv

from analyzer import parse_excel, analyze, compute_comparisons, generate_monthly_summary, prepare_ai_prompt, generate_analysis_line
from report_gen import generate_report

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', uuid.uuid4().hex)
SITE_PASSWORD = os.environ.get('SITE_PASSWORD', '0000')

UPLOAD_DIR = Path(__file__).parent / 'uploads'
RESULT_DIR = Path(__file__).parent / 'results'
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

analysis_store = {}

# Debug log — persisted to file so it survives worker restarts
import datetime as _dt
DEBUG_LOG_FILE = Path(__file__).parent / 'debug.log'

def _log_error(source, error):
    """Record error for debugging (file + in-memory)."""
    entry = f'[{_dt.datetime.now().isoformat()}] {source}: {error}'
    try:
        with open(DEBUG_LOG_FILE, 'a') as f:
            f.write(entry + '\n')
    except Exception:
        pass


@app.route('/debug-log')
def debug_log():
    """Show recent debug log entries from file."""
    try:
        if not DEBUG_LOG_FILE.exists():
            return '<pre>No errors logged yet.</pre>'
        lines = DEBUG_LOG_FILE.read_text().strip().split('\n')
        return '<pre>' + '\n'.join(lines[-30:]) + '</pre>'
    except Exception as e:
        return f'<pre>Error reading log: {e}</pre>'


@app.errorhandler(500)
def internal_error(e):
    tb = traceback.format_exc()
    return f'<pre style="color:red;padding:20px;">500 Internal Server Error\n\n{tb}</pre>', 500


@app.route('/debug')
def debug_route():
    try:
        from analyzer import parse_excel, analyze, parse_historical
        import sys
        info = [
            f'Python: {sys.version}',
            f'pandas: {__import__("pandas").__version__}',
            f'openpyxl: {__import__("openpyxl").__version__}',
            'Imports OK',
        ]
        return '<pre>' + '\n'.join(info) + '</pre>'
    except Exception as e:
        return f'<pre style="color:red;">Import Error: {e}\n{traceback.format_exc()}</pre>', 500


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == SITE_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        error = '密码错误'
    return render_template('login.html', error=error)


@app.route('/')
@require_auth
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
@require_auth
def upload():
    cur_file = request.files.get('cur_file')
    if not cur_file or not cur_file.filename.endswith(('.xlsx', '.xls')):
        return redirect(url_for('index', error='请上传当月报告文件'))

    prev_files = []
    prev_filenames = []
    for i in range(1, 4):
        pf = request.files.get(f'prev_file_{i}')
        if pf and pf.filename.endswith(('.xlsx', '.xls')):
            prev_files.append(pf)
            prev_filenames.append(pf.filename)

    hist_file = request.files.get('hist_file')
    owner_name = request.form.get('owner_name', '').strip() or None

    try:
        # Parse current month
        _log_error('upload', f'Starting: file={cur_file.filename}, prev={len(prev_files)}, owner={owner_name}')
        cur_path = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
        cur_file.save(str(cur_path))
        _log_error('upload', f'Saved cur file: {cur_path.stat().st_size} bytes')
        df, csat_df, metrics, csat_quality_raw = parse_excel(str(cur_path))
        _log_error('upload', f'Parsed: {len(df)} rows')
        current_stats = analyze(df, csat_df, metrics, csat_quality_raw)
        _log_error('upload', 'Analyzed current month OK')
        cur_path.unlink(missing_ok=True)

        # Parse previous months
        prev_stats_list = []
        for pf in prev_files:
            ppath = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
            pf.save(str(ppath))
            pdf, pcsat, pmetrics, pquality = parse_excel(str(ppath))
            prev_stats_list.append(analyze(pdf, pcsat, pmetrics, pquality))
            ppath.unlink(missing_ok=True)

        # Parse historical
        historical_data = None
        if hist_file and hist_file.filename.endswith(('.xlsx', '.xls')):
            try:
                from analyzer import parse_historical
                hpath = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
                hist_file.save(str(hpath))
                historical_data = parse_historical(str(hpath))
                hpath.unlink(missing_ok=True)
            except Exception:
                pass

        # Comparisons
        current_stats = compute_comparisons(current_stats, prev_stats_list)

        # Owner filter
        if owner_name:
            cur_path2 = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
            cur_file.seek(0)
            cur_file.save(str(cur_path2))
            df2, csat_df2, metrics2, quality2 = parse_excel(str(cur_path2))
            if 'Owner' in df2.columns:
                df2 = df2[df2['Owner'].str.contains(owner_name, na=False, case=False)]
            elif 'Individual' in df2.columns:
                df2 = df2[df2['Individual'].str.contains(owner_name, na=False, case=False)]
            current_stats = analyze(df2, csat_df2, metrics2, quality2)
            if prev_files:
                prev_owner = []
                for pf in prev_files:
                    pf.seek(0)
                    ppath2 = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
                    pf.save(str(ppath2))
                    pdf2, pcsat2, pm2, pq2 = parse_excel(str(ppath2))
                    if 'Owner' in pdf2.columns:
                        pdf2 = pdf2[pdf2['Owner'].str.contains(owner_name, na=False, case=False)]
                    elif 'Individual' in pdf2.columns:
                        pdf2 = pdf2[pdf2['Individual'].str.contains(owner_name, na=False, case=False)]
                    prev_owner.append(analyze(pdf2, pcsat2, pm2, pq2))
                    ppath2.unlink(missing_ok=True)
                current_stats = compute_comparisons(current_stats, prev_owner)
            cur_path2.unlink(missing_ok=True)

        # Summary
        try:
            current_stats['monthly_summary'] = generate_monthly_summary(current_stats)
        except Exception as e:
            current_stats['monthly_summary'] = f'Summary generation failed: {e}'
            _log_error('generate_monthly_summary', e)

        # Analysis line
        try:
            from datetime import datetime, timedelta
            end_str = current_stats.get('date_range', {}).get('end', '')
            end_dt = datetime.strptime(end_str, '%Y-%m-%d')
            last_month_label = (end_dt.replace(day=1) - timedelta(days=1)).strftime("%b'%y")
        except Exception:
            last_month_label = ''
        try:
            current_stats['analysis_summary_line'] = generate_analysis_line(current_stats, last_month_label)
        except Exception as e:
            current_stats['analysis_summary_line'] = ''
            _log_error('generate_analysis_line', e)

        # Store
        analysis_id = uuid.uuid4().hex
        analysis_store[analysis_id] = {
            'stats': current_stats,
            'filename': cur_file.filename,
            'prev_filenames': prev_filenames,
            'has_comparison': current_stats.get('has_comparison', False),
            'owner_name': owner_name,
            'historical': historical_data,
        }

        return redirect(url_for('dashboard', analysis_id=analysis_id))

    except Exception as e:
        tb = traceback.format_exc()
        _log_error('upload', tb)
        return f'<pre style="color:red;padding:20px;white-space:pre-wrap;">上传失败: {e}\n\n{tb}</pre>', 500


@app.route('/dashboard/<analysis_id>')
@require_auth
def dashboard(analysis_id):
    data = analysis_store.get(analysis_id)
    if not data:
        return redirect(url_for('index', error='分析结果已过期，请重新上传'))
    return render_template('dashboard.html',
                           analysis_id=analysis_id,
                           stats=data['stats'],
                           filename=data['filename'],
                           prev_filenames=data.get('prev_filenames', []),
                           has_comparison=data.get('has_comparison', False),
                           owner_name=data.get('owner_name', ''),
                           historical=data.get('historical'))


@app.route('/api/ai-analysis/<analysis_id>', methods=['POST'])
@require_auth
def ai_analysis(analysis_id):
    data = analysis_store.get(analysis_id)
    if not data:
        return jsonify({'error': '分析结果不存在'}), 404

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return jsonify({'error': '未配置 ANTHROPIC_API_KEY'}), 400

    prompt = prepare_ai_prompt(data['stats'])

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=2048,
            system='你是一位资深的HR运营分析师。请基于提供的Case统计数据，给出：1) 核心发现 (3-5条) 2) 改善建议 (按优先级排序) 3) 值得关注的趋势。用中文回复，简洁直接。',
            messages=[{'role': 'user', 'content': prompt}],
        )
        result = message.content[0].text
        data['ai_result'] = result
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'AI 分析调用失败: {str(e)}'}), 500


@app.route('/api/stats/<analysis_id>')
@require_auth
def get_stats(analysis_id):
    data = analysis_store.get(analysis_id)
    if not data:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(data['stats'])


@app.route('/api/report/<analysis_id>')
@require_auth
def download_report(analysis_id):
    data = analysis_store.get(analysis_id)
    if not data:
        return redirect(url_for('index', error='分析结果已过期，请重新上传'))

    buf = generate_report(data['stats'], data['filename'],
                          has_comparison=data.get('has_comparison', False))

    report_name = f"HR_Case_Report_{data['stats']['date_range']['end']}.docx"
    return send_file(
        buf,
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        as_attachment=True,
        download_name=report_name,
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('RENDER') is None
    print(f'Case Intelligence Platform — http://0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=debug)
