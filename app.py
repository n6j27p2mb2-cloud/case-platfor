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

# In-memory storage (lives as long as the worker process)
analysis_store = {}


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

    # Save files
    cur_path = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
    cur_file.save(str(cur_path))

    prev_paths = []
    prev_filenames = []
    for i in range(1, 4):
        pf = request.files.get(f'prev_file_{i}')
        if pf and pf.filename.endswith(('.xlsx', '.xls')):
            ppath = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
            pf.save(str(ppath))
            prev_paths.append(str(ppath))
            prev_filenames.append(pf.filename)

    hist_path = None
    hist_file = request.files.get('hist_file')
    if hist_file and hist_file.filename.endswith(('.xlsx', '.xls')):
        hpath = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
        hist_file.save(str(hpath))
        hist_path = str(hpath)

    owner_name = request.form.get('owner_name', '').strip() or None

    # Process synchronously
    cur_filename = cur_file.filename
    try:
        df, csat_df, metrics, csat_quality_raw = parse_excel(str(cur_path))
        current_stats = analyze(df, csat_df, metrics, csat_quality_raw)

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
            df2, csat_df2, metrics2, quality2 = parse_excel(str(cur_path))
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
            from datetime import datetime, timedelta
            end_str = current_stats.get('date_range', {}).get('end', '')
            end_dt = datetime.strptime(end_str, '%Y-%m-%d')
            ll = (end_dt.replace(day=1) - timedelta(days=1)).strftime("%b'%y")
        except Exception:
            ll = ''
        try:
            current_stats['analysis_summary_line'] = generate_analysis_line(current_stats, ll)
        except Exception:
            current_stats['analysis_summary_line'] = ''

        analysis_id = uuid.uuid4().hex
        analysis_store[analysis_id] = {
            'stats': current_stats,
            'filename': cur_filename,
            'prev_filenames': prev_filenames,
            'has_comparison': current_stats.get('has_comparison', False),
            'owner_name': owner_name or '',
            'historical': historical_data,
        }

        # Clean up uploaded files
        for p in [str(cur_path)] + prev_paths + ([hist_path] if hist_path else []):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

        return redirect(url_for('dashboard', analysis_id=analysis_id))

    except Exception as e:
        tb = traceback.format_exc()
        # Log to file for debugging
        try:
            log_file = Path(__file__).parent / 'errors.log'
            with open(log_file, 'a') as f:
                f.write(f'[{__import__("datetime").datetime.now().isoformat()}] {tb}\n')
        except Exception:
            pass
        # Clean up on error
        for p in [str(cur_path)] + prev_paths + ([hist_path] if hist_path else []):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        return f'<pre style="color:red;padding:20px;">分析失败:\n\n{tb[-2000:]}</pre>', 500


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


@app.route('/errors-log')
def errors_log():
    """Show processing errors."""
    try:
        log_file = Path(__file__).parent / 'errors.log'
        if not log_file.exists():
            return '<pre>No errors yet.</pre>'
        return '<pre>' + log_file.read_text()[-5000:] + '</pre>'
    except Exception as e:
        return f'<pre>Error: {e}</pre>'


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


@app.route('/debug')
def debug_route():
    try:
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('RENDER') is None
    print(f'Case Intelligence Platform — http://0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=debug)
