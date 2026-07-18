import os
import uuid
from functools import wraps
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from dotenv import load_dotenv

from analyzer import parse_excel, analyze, compute_comparison, prepare_ai_prompt
from report_gen import generate_report

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', uuid.uuid4().hex)
SITE_PASSWORD = os.environ.get('SITE_PASSWORD', '0617')

UPLOAD_DIR = Path(__file__).parent / 'uploads'
RESULT_DIR = Path(__file__).parent / 'results'
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# {analysis_id: {stats, filename, session_id, ...}}
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
    file = request.files.get('file')
    if not file or not file.filename.endswith(('.xlsx', '.xls')):
        return redirect(url_for('index', error='请上传本月 .xlsx 或 .xls 文件'))

    prev_file = request.files.get('prev_file')
    has_prev = prev_file and prev_file.filename.endswith(('.xlsx', '.xls'))

    filepath = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
    file.save(filepath)

    prev_filepath = None
    if has_prev:
        prev_filepath = UPLOAD_DIR / f'{uuid.uuid4().hex}.xlsx'
        prev_file.save(prev_filepath)

    try:
        df = parse_excel(str(filepath))
        stats = analyze(df)

        prev_stats = None
        if has_prev:
            prev_df = parse_excel(str(prev_filepath))
            prev_stats = analyze(prev_df)

        stats = compute_comparison(stats, prev_stats)

        analysis_id = uuid.uuid4().hex
        analysis_store[analysis_id] = {
            'stats': stats,
            'filename': file.filename,
            'prev_filename': prev_file.filename if has_prev else None,
            'has_comparison': has_prev,
            'session_id': session.get('user_id', 'default'),
        }

        filepath.unlink(missing_ok=True)
        if prev_filepath:
            prev_filepath.unlink(missing_ok=True)

        return redirect(url_for('dashboard', analysis_id=analysis_id))
    except Exception as e:
        filepath.unlink(missing_ok=True)
        if prev_filepath:
            prev_filepath.unlink(missing_ok=True)
        return redirect(url_for('index', error=f'解析失败: {str(e)}'))


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
                           prev_filename=data.get('prev_filename'),
                           has_comparison=data.get('has_comparison', False))


@app.route('/api/ai-analysis/<analysis_id>', methods=['POST'])
@require_auth
def ai_analysis(analysis_id):
    data = analysis_store.get(analysis_id)
    if not data:
        return jsonify({'error': '分析结果不存在'}), 404

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return jsonify({'error': '未配置 ANTHROPIC_API_KEY，请在 .env 文件中设置'}), 400

    prompt = prepare_ai_prompt(data['stats'])

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=2048,
            system='你是一位资深的HR运营分析师。请基于提供的Case统计数据，给出：1) 核心发现 (3-5条) 2) 改善建议 (按优先级排序) 3) 值得关注的趋势。用中文回复，简洁直接，每条建议都要具体可执行。',
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
    import os as _os
    port = int(_os.environ.get('PORT', 5050))
    debug = _os.environ.get('RENDER') is None
    print(f'Case Intelligence Platform — http://127.0.0.1:{port}')
    app.run(host='127.0.0.1', port=port, debug=debug)
