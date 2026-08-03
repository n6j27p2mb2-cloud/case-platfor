import os
import sys
import json
import uuid
import subprocess
import traceback
from functools import wraps
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from dotenv import load_dotenv

from analyzer import prepare_ai_prompt
from report_gen import generate_report

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', uuid.uuid4().hex)
SITE_PASSWORD = os.environ.get('SITE_PASSWORD', '0000')

UPLOAD_DIR = Path(__file__).parent / 'uploads'
RESULT_DIR = Path(__file__).parent / 'results'
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

WORKER_LOG = Path(__file__).parent / 'worker.log'


def _get_task(task_id):
    """Read task status from result file."""
    f = RESULT_DIR / f'{task_id}_status.json'
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _get_analysis(analysis_id):
    """Read analysis data from result file."""
    f = RESULT_DIR / f'{analysis_id}_analysis.json'
    if not f.exists():
        return None
    return json.loads(f.read_text())


@app.route('/worker-log')
def worker_log():
    """Show worker subprocess log (processing progress)."""
    try:
        if not WORKER_LOG.exists():
            return '<pre>No worker log yet.</pre>'
        lines = WORKER_LOG.read_text().strip().split('\n')
        return '<pre>' + '\n'.join(lines[-50:]) + '</pre>'
    except Exception as e:
        return f'<pre>Error: {e}</pre>'


@app.errorhandler(500)
def internal_error(e):
    tb = traceback.format_exc()
    return f'<pre style="color:red;padding:20px;">500 Internal Server Error\n\n{tb}</pre>', 500


@app.route('/debug')
def debug_route():
    try:
        import sys
        info = [
            f'Python: {sys.version}',
            f'pandas: {__import__("pandas").__version__}',
            f'openpyxl: {__import__("openpyxl").__version__}',
            'Imports OK',
            f'Results dir: {RESULT_DIR}',
            f'Files: {list(RESULT_DIR.glob("*"))}',
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

    # Save current file
    cur_path = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
    cur_file.save(str(cur_path))

    # Save previous files
    prev_paths = []
    prev_filenames = []
    for i in range(1, 4):
        pf = request.files.get(f'prev_file_{i}')
        if pf and pf.filename.endswith(('.xlsx', '.xls')):
            ppath = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
            pf.save(str(ppath))
            prev_paths.append(str(ppath))
            prev_filenames.append(pf.filename)

    # Save historical
    hist_path = None
    hist_file = request.files.get('hist_file')
    if hist_file and hist_file.filename.endswith(('.xlsx', '.xls')):
        hpath = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
        hist_file.save(str(hpath))
        hist_path = str(hpath)

    owner_name = request.form.get('owner_name', '').strip() or None

    # Create task — status stored in result file
    task_id = uuid.uuid4().hex
    status_file = RESULT_DIR / f'{task_id}_status.json'
    status_file.write_text(json.dumps({'status': 'pending', 'analysis_id': '', 'error': ''}))

    # Build command for subprocess worker
    cmd = [
        sys.executable, str(Path(__file__).parent / 'process_worker.py'),
        task_id, str(cur_path),
    ]
    for pp in prev_paths:
        cmd.append(pp)
    if hist_path:
        cmd.extend(['--hist', hist_path])
    if owner_name:
        cmd.extend(['--owner', owner_name])

    # Run processing in subprocess (completely independent of gunicorn)
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return redirect(url_for('processing', task_id=task_id), code=303)


@app.route('/processing/<task_id>')
@require_auth
def processing(task_id):
    if _get_task(task_id) is None:
        return redirect(url_for('index', error='任务已过期，请重新上传'))
    return render_template('processing.html', task_id=task_id)


@app.route('/api/status/<task_id>')
@require_auth
def task_status(task_id):
    task = _get_task(task_id)
    if not task:
        return jsonify({'status': 'error', 'error': '任务不存在'}), 404
    return jsonify(task)


@app.route('/dashboard/<analysis_id>')
@require_auth
def dashboard(analysis_id):
    data = _get_analysis(analysis_id)
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
    data = _get_analysis(analysis_id)
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
    data = _get_analysis(analysis_id)
    if not data:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(data['stats'])


@app.route('/api/report/<analysis_id>')
@require_auth
def download_report(analysis_id):
    data = _get_analysis(analysis_id)
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
