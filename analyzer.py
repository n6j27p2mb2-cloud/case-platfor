import re
import warnings
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

warnings.filterwarnings('ignore', category=UserWarning)

# ── PII columns ────────────────────────────────────────────────────────────────
PII_DROP = [
    'EmployeeId', 'First Name', 'Last Name', 'EE Name',
    'PrimaryEmailAddress', 'Country', 'Department', 'OrgName',
    'EE FirstName', 'EE LastName', 'Employee Full Name',
    'OWNER FullName', 'Comment - Feedback',
]

CHINESE_STOP_WORDS = {
    # 泛词 / 无意义动词
    '咨询', '员工', '个人', '查询', '帮助', '入职', '离职', '相关', '进行',
    '问题', '处理', '提供', '一个', '联系', '了解', '可以', '需要', '什么',
    '没有', '这个', '他们', '还是', '怎么', '如何', '是否', '已经', '并且',
    '那个', '或者', '申请', '公司', '工作', '请问', '问一下', '想问', '帮查',
    '一下', '想问一下', '麻烦', '帮忙', '回复', '谢谢', '请问一下',
    '重复', '未咨询', '咨询未', '重复咨询', '请问下',
    '请您', '操作', '转发', '提示', '以及', '协助', '反馈', '跟进',
    '是否可以', '有没有', '能不能', '可不可以', '你好', '您好',
    '请问您', '想问您', '感谢', '收到', '希望', '告知', '确认',
    '询问', '核实', '查看', '沟通', '说明', '咨询一下',
    # 城市 / 地名
    '上海', '武汉', '北京', '深圳', '广州', '杭州', '成都', '南京',
    '苏州', '重庆', '天津', '西安', '长沙', '郑州', '东莞', '合肥',
    '大连', '青岛', '厦门', '宁波', '福州', '无锡', '佛山', '济南',
    '沈阳', '昆明', '贵阳', '南宁', '海口', '石家庄', '哈尔滨', '长春',
    '中国', '境内', '境外', '海外', '大陆',
}

# ── Helpers ─────────────────────────────────────────────────────────────────────

def _simplify_case_type(ct):
    """Split CaseType into major (before first colon) and minor (after)."""
    if pd.isna(ct):
        return 'Unknown', 'Unknown'
    s = str(ct).strip()
    if ':' in s:
        idx = s.index(':')
        return s[:idx].strip(), s[idx+1:].strip()
    return s, s


def _classify_origin(origin):
    """Normalize origin to: Chat / Email / Manual / Other."""
    if pd.isna(origin):
        return 'Other'
    s = str(origin).strip().lower()
    if 'chat' in s:
        return 'Chat'
    if 'email' in s:
        return 'Email'
    if 'manual' in s:
        return 'Chat'
    return 'Other'


def _extract_title_keywords(titles, top_n=15, min_count=40):
    """Extract meaningful Chinese keywords from titles, excluding stop words."""
    kw_counter = Counter()
    for t in titles:
        if pd.isna(t):
            continue
        t_clean = re.sub(r'[（(][^)）]*[)）]', '', str(t))
        t_clean = re.sub(r'[a-zA-Z0-9\s\.\-\/]+', '', t_clean)
        chunks = re.findall(r'[一-鿿]{2,6}', t_clean)
        for chunk in chunks:
            if chunk not in CHINESE_STOP_WORDS and len(chunk) >= 2:
                kw_counter[chunk] += 1
    results = [{'keyword': k, 'count': c} for k, c in kw_counter.most_common(top_n) if c >= min_count]
    return results


def _extract_hot_keywords_from_major(df, major, top_n=10):
    """Extract hot keywords from Titles within a specific major CaseType."""
    major_df = df[df['_case_major'] == major]
    titles = major_df['Title'].dropna().tolist() if 'Title' in major_df.columns else []
    return _extract_title_keywords(titles, top_n, min_count=20)


def _fmt_duration(sec):
    if sec is None:
        return '—'
    if sec < 60:
        return f'{sec:.0f}秒'
    if sec < 3600:
        return f'{sec/60:.1f}分钟'
    return f'{sec/3600:.1f}小时'


# ── Parse ───────────────────────────────────────────────────────────────────────

def parse_excel(filepath):
    """Parse a monthly HR Ops report. Returns (df, csat_df, metrics_dict) for China only."""
    # Read HROps sheet with only needed columns to save memory on Render
    needed_cols = ['GEO', 'Created', 'FirstCloseDate', 'CaseType', 'Origin',
                   'FirstCloseResolution', 'Owner', 'Individual',
                   'Title', 'SLA Met Flag', 'FTF', 'Reopen']
    # First peek to get available columns
    df_peek = pd.read_excel(filepath, sheet_name='HROps', nrows=0, engine='openpyxl')
    avail_cols = list(df_peek.columns)
    read_cols = [c for c in needed_cols if c in avail_cols]
    df = pd.read_excel(filepath, sheet_name='HROps', usecols=read_cols, engine='openpyxl')

    # Filter China only
    if 'GEO' in df.columns:
        df = df[df['GEO'] == 'China'].copy()

    # Normalize dates
    for col in ['Created', 'FirstCloseDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)

    # Case Type hierarchy
    if 'CaseType' in df.columns:
        parsed = df['CaseType'].apply(_simplify_case_type)
        df['_case_major'] = parsed.apply(lambda x: x[0])
        df['_case_minor'] = parsed.apply(lambda x: x[1])

    # Origin classification
    if 'Origin' in df.columns:
        df['_origin_class'] = df['Origin'].apply(_classify_origin)

    # Resolution classification
    if 'FirstCloseResolution' in df.columns:
        df['_resolution'] = df['FirstCloseResolution'].apply(_classify_resolution)

    # Is Non-HR
    df['_is_non_hr'] = df['CaseType'].apply(
        lambda x: str(x).startswith('Non-HR') if pd.notna(x) else False
    )

    # ── CSAT sheet (China only) ──
    csat_df = None
    csat_quality_raw = None
    try:
        csat_peek = pd.read_excel(filepath, sheet_name='CSAT', nrows=0, engine='openpyxl')
    except ValueError:
        csat_peek = None

    if csat_peek is not None:
        csat_avail = list(csat_peek.columns)
        csat_read = [c for c in needed_cols if c in csat_avail]
        # Also read quality-related columns
        quality_col_names = [c for c in csat_avail
                            if any(kw in str(c).lower() for kw in ['interact', 'pers', 'comm', 'qual'])]
        all_csat_cols = list(set(csat_read + quality_col_names))
        csat_df = pd.read_excel(filepath, sheet_name='CSAT', usecols=all_csat_cols, engine='openpyxl')
        if 'GEO' in csat_df.columns:
            csat_df = csat_df[csat_df['GEO'] == 'China'].copy()
        quality_cols = {}
        for col in csat_df.columns:
            col_lower = str(col).strip().lower()
            if 'interaction' in col_lower or 'interact' in col_lower:
                quality_cols['interaction'] = col
            elif 'pers' in col_lower and ('serv' in col_lower or 'personal' in col_lower):
                quality_cols['pers_serv'] = col
            elif 'comm' in col_lower:
                quality_cols['comm'] = col
        if quality_cols:
            csat_quality_raw = {'cols': quality_cols, 'df': csat_df}

    # ── Metrics sheet ──
    metrics = {}
    try:
        metrics_df = pd.read_excel(filepath, sheet_name='Metrics', header=None, engine='openpyxl')
    except ValueError:
        metrics_df = None

    if metrics_df is not None:
        metrics = _parse_metrics_sheet(metrics_df, df, csat_df)

    return df, csat_df, metrics, csat_quality_raw


def _classify_resolution(res_str):
    if pd.isna(res_str):
        return 'No CSAT'
    s = str(res_str).strip()
    if 'Duplicate' in s:
        return 'Duplicate'
    if 'third party' in s.lower():
        return '3rd Party'
    if 'No CSAT' in s:
        return 'No CSAT'
    if 'CSAT' in s:
        return 'CSAT'
    return 'No CSAT'


def _parse_metrics_sheet(metrics_df, hrops_df, csat_df):
    """Extract key metrics for China from the Metrics sheet + raw data fallback."""
    result = {}

    # ── Try to extract from Metrics sheet first ──
    # Scan for section headers and find China values
    section_map = {
        'SLA (target': ('sla_compliance', False),
        'CSAT (target': ('csat_score', False),
        'CSAT RR (target': ('csat_rr', True),
        'FTF (target': ('ftf_rate', False),
        'REOPEN RATE': ('reopen_rate', False),
    }

    metrics_extracted = {}
    for row_idx in range(len(metrics_df)):
        for col_idx in range(len(metrics_df.columns)):
            val = str(metrics_df.iloc[row_idx, col_idx])
            for section_key, (result_key, is_pct) in section_map.items():
                if section_key in val:
                    # Found a section header, scan nearby rows for "China"
                    for r in range(row_idx, min(len(metrics_df), row_idx + 20)):
                        for c in range(max(0, col_idx - 1), min(len(metrics_df.columns), col_idx + 3)):
                            cell_val = str(metrics_df.iloc[r, c])
                            if 'China' in cell_val:
                                # Get the value from the next column or same area
                                for vc in range(c + 1, min(len(metrics_df.columns), c + 4)):
                                    v = metrics_df.iloc[r, vc]
                                    if pd.notna(v) and isinstance(v, (int, float)):
                                        if is_pct and v < 1:
                                            v = round(v * 100, 2)
                                        elif is_pct and v > 1:
                                            v = round(v, 2)
                                        else:
                                            v = round(v, 2) if isinstance(v, float) else v
                                        metrics_extracted[result_key] = v
                                        break
                                break
    result.update(metrics_extracted)

    # ── Fallback: compute from raw data ──

    # CSAT score
    if 'csat_score' not in result and csat_df is not None and 'AVARAGE' in csat_df.columns:
        scores = csat_df['AVARAGE'].dropna()
        if len(scores) > 0:
            result['csat_score'] = round(scores.mean(), 2)
            result['csat_responses'] = len(scores)

    # CSAT RR from CSAT sheet (use CHINA CSAT RR if available)
    if 'csat_rr' not in result and csat_df is not None:
        total_csat = len(csat_df)
        responded = len(csat_df[csat_df['AVARAGE'].notna()]) if 'AVARAGE' in csat_df.columns else 0
        if total_csat > 0:
            result['csat_rr'] = round(responded / total_csat * 100, 2)

    # SLA from HROps
    if 'sla_compliance' not in result and 'SLA Met Flag' in hrops_df.columns:
        sla_col = hrops_df['SLA Met Flag'].dropna()
        if len(sla_col) > 0:
            result['sla_compliance'] = round(sla_col.mean() * 100, 2)

    # FTF
    if 'ftf_rate' not in result and 'FTF' in hrops_df.columns:
        ftf_col = hrops_df['FTF'].dropna()
        if len(ftf_col) > 0:
            result['ftf_rate'] = round(ftf_col.mean() * 100, 2)

    # Reopen
    if 'reopen_rate' not in result and 'Reopen' in hrops_df.columns:
        reopen_col = hrops_df['Reopen']
        reopen_n = int(reopen_col.sum()) if reopen_col.notna().sum() > 0 else 0
        result['reopen_count'] = reopen_n
        result['reopen_rate'] = round(reopen_n / len(hrops_df) * 100, 2) if len(hrops_df) > 0 else 0

    return result


# ── Analyze ─────────────────────────────────────────────────────────────────────

def analyze(df, csat_df=None, metrics=None, csat_quality_raw=None):
    """Analyze China-only data and return stats dict."""
    total = len(df)

    # ── Basic counts ──
    non_hr_count = int(df['_is_non_hr'].sum()) if '_is_non_hr' in df.columns else 0
    hr_count = total - non_hr_count

    dup_count = int((df['_resolution'] == 'Duplicate').sum()) if '_resolution' in df.columns else 0
    third_party_count = int((df['_resolution'] == '3rd Party').sum()) if '_resolution' in df.columns else 0

    # ── Case Type hierarchy ──
    major_counter = Counter()
    major_minors = defaultdict(Counter)
    for _, row in df.iterrows():
        major = row.get('_case_major', 'Unknown')
        minor = row.get('_case_minor', 'Unknown')
        major_counter[major] += 1
        major_minors[major][minor] += 1

    type_hierarchy = []
    for major, count in major_counter.most_common():
        pct = round(count / total * 100, 1) if total > 0 else 0
        entry = {
            'major': major,
            'count': count,
            'percentage': pct,
            'is_non_hr': str(major).startswith('Non-HR'),
            'minors': [],
            # Comparison defaults (filled by compute_comparisons)
            'avg_3m': None,
            'vs_3m_diff': None,
            'vs_3m_pct': None,
            'prev_count': None,
            'delta': None,
            'delta_pct': None,
        }
        for minor, mcount in major_minors[major].most_common():
            mpct = round(mcount / total * 100, 1) if total > 0 else 0
            entry['minors'].append({
                'name': minor,
                'count': mcount,
                'percentage': mpct,
                'major_pct': round(mcount / count * 100, 1) if count > 0 else 0,
            })
        type_hierarchy.append(entry)

    # ── Subtype ranking (across all majors, excl Non-HR, threshold >= 30) ──
    subtype_ranking = _subtype_ranking(type_hierarchy, total, threshold=30)

    # ── Origin ──
    origin_counts = df['_origin_class'].value_counts().to_dict() if '_origin_class' in df.columns else {}

    # ── Resolution ──
    resolution_counts = df['_resolution'].value_counts().to_dict() if '_resolution' in df.columns else {}

    # ── Daily trend ──
    daily_trend = _daily_trend(df)

    # ── Date range ──
    date_range = _date_range(df)

    # ── Hot keywords from titles ──
    all_titles = df['Title'].dropna().tolist() if 'Title' in df.columns else []
    hot_keywords = _extract_title_keywords(all_titles, top_n=30, min_count=40)

    # ── Hot keywords per major (for drill-down) ──
    major_hot_keywords = {}
    for h in type_hierarchy[:8]:
        if not h['is_non_hr']:
            major_hot_keywords[h['major']] = _extract_hot_keywords_from_major(df, h['major'])

    # ── Case breakdown ──
    actual_hr = hr_count - third_party_count - dup_count
    case_breakdown = {
        'HR 实际处理': max(0, actual_hr),
        'Non-HR': non_hr_count,
        '转第三方': third_party_count,
        '系统重复': dup_count,
    }

    # ── CSAT Quality Metrics (from CSAT sheet columns) ──
    csat_quality = {}
    if csat_quality_raw is not None:
        qdf = csat_quality_raw['df']
        cols = csat_quality_raw['cols']
        quality_labels = {
            'interaction': 'Solution Quality',
            'pers_serv': 'Service Professionalism',
            'comm': 'Clear Communication',
        }
        for key, col_name in cols.items():
            if col_name in qdf.columns:
                vals = pd.to_numeric(qdf[col_name], errors='coerce').dropna()
                if len(vals) > 0:
                    avg_val = vals.mean()
                    pct = round(avg_val / 5 * 100, 1)
                    csat_quality[quality_labels[key]] = {
                        'avg': round(avg_val, 2),
                        'pct': pct,
                        'count': len(vals),
                    }

    # ── KPI metrics ──
    m = metrics or {}
    csat_score = m.get('csat_score')
    csat_rr = m.get('csat_rr')
    sla_compliance = m.get('sla_compliance')
    if sla_compliance is None and 'SLA Met Flag' in df.columns:
        sla_col = df['SLA Met Flag'].dropna()
        sla_compliance = round(sla_col.mean() * 100, 2) if len(sla_col) > 0 else None

    # CSAT from raw data if not in metrics
    if csat_score is None and csat_df is not None and 'AVARAGE' in csat_df.columns:
        scores = csat_df['AVARAGE'].dropna()
        csat_score = round(scores.mean(), 2) if len(scores) > 0 else None

    if csat_rr is None and csat_df is not None:
        total_csat = len(csat_df)
        responded = len(csat_df[csat_df['AVARAGE'].notna()]) if 'AVARAGE' in csat_df.columns else 0
        csat_rr = round(responded / total_csat * 100, 2) if total_csat > 0 else None

    stats = {
        'total_cases': total,
        'hr_count': hr_count,
        'non_hr_count': non_hr_count,
        'duplicate_count': dup_count,
        'third_party_count': third_party_count,
        'type_hierarchy': type_hierarchy,
        'subtype_ranking': subtype_ranking,
        'origin_distribution': _fmt_dist(origin_counts, total),
        'resolution_counts': resolution_counts,
        'daily_trend': daily_trend,
        'hot_keywords': hot_keywords,
        'major_hot_keywords': major_hot_keywords,
        'case_breakdown': case_breakdown,
        'date_range': date_range,
        # Four-card metrics
        'csat_score': csat_score,
        'csat_rr': csat_rr,
        'sla_compliance': sla_compliance,
        'ftf_rate': m.get('ftf_rate'),
        'reopen_rate': m.get('reopen_rate'),
        'reopen_count': m.get('reopen_count'),
        'csat_quality': csat_quality,
        # Comparison defaults (filled by compute_comparisons)
        'has_comparison': False,
        'card_comparisons': {},
        'hotspots': [],
        'subtype_growth_ranking': [],
    }

    return stats


# ── Multi-month comparison ──────────────────────────────────────────────────────

def compute_comparisons(current, prev_months):
    """
    Compute vs上月 and vs近三月均值 for the current month.
    prev_months: list of stats dicts, ordered most recent first (e.g. [May, April, March])
    Returns current with added comparison fields.
    """
    if not prev_months:
        current['has_comparison'] = False
        current['hotspots'] = []
        current['subtype_growth_ranking'] = []
        return current

    current['has_comparison'] = True
    last_month = prev_months[0]

    # ── Four card comparisons ──
    current['card_comparisons'] = {}

    for card_key, stat_key in [
        ('total', 'total_cases'),
        ('sla', 'sla_compliance'),
        ('csat_score', 'csat_score'),
        ('rr', 'csat_rr'),
    ]:
        cur_val = current.get(stat_key)
        lm_val = last_month.get(stat_key)
        avg_vals = [m.get(stat_key) for m in prev_months if m.get(stat_key) is not None]

        comp = {'current': cur_val}
        if lm_val is not None and cur_val is not None:
            comp['vs_last'] = round(cur_val - lm_val, 2)
            comp['vs_last_pct'] = round((cur_val - lm_val) / lm_val * 100, 1) if lm_val != 0 else None
        if avg_vals:
            avg_3m = sum(avg_vals) / len(avg_vals)
            comp['vs_3m_avg'] = round(avg_3m, 2)
            comp['vs_3m_diff'] = round(cur_val - avg_3m, 2)
            comp['vs_3m_pct'] = round((cur_val - avg_3m) / avg_3m * 100, 1) if avg_3m != 0 else None
        current['card_comparisons'][card_key] = comp

    # ── CaseType comparison with 3-month avg ──
    prev_major_counts = defaultdict(list)
    for pm in prev_months:
        for h in pm.get('type_hierarchy', []):
            prev_major_counts[h['major']].append(h['count'])

    for h in current['type_hierarchy']:
        prev_counts = prev_major_counts.get(h['major'], [])
        if prev_counts:
            h['avg_3m'] = round(sum(prev_counts) / len(prev_counts), 1)
            h['vs_3m_diff'] = h['count'] - h['avg_3m']
            h['vs_3m_pct'] = round((h['count'] - h['avg_3m']) / h['avg_3m'] * 100, 1)
        # vs last month
        lm_majors = {hh['major']: hh['count'] for hh in last_month.get('type_hierarchy', [])}
        if h['major'] in lm_majors:
            h['prev_count'] = lm_majors[h['major']]
            h['delta'] = h['count'] - h['prev_count']
            h['delta_pct'] = round(h['delta'] / h['prev_count'] * 100, 1) if h['prev_count'] > 0 else None

    # ── Subtype comparison with 3-month avg ──
    prev_subtype_counts = defaultdict(list)
    for pm in prev_months:
        for s in pm.get('subtype_ranking', []):
            key = (s['major'], s['subtype'])
            prev_subtype_counts[key].append(s['count'])

    for s in current['subtype_ranking']:
        key = (s['major'], s['subtype'])
        prev_counts = prev_subtype_counts.get(key, [])
        if prev_counts:
            s['avg_3m'] = round(sum(prev_counts) / len(prev_counts), 1)
            s['vs_3m_diff'] = s['count'] - s['avg_3m']
            s['vs_3m_pct'] = round((s['count'] - s['avg_3m']) / s['avg_3m'] * 100, 1)
        # vs last month
        lm_subs = {(ss['major'], ss['subtype']): ss['count'] for ss in last_month.get('subtype_ranking', [])}
        if key in lm_subs:
            s['prev_count'] = lm_subs[key]
            s['delta'] = s['count'] - s['prev_count']
            s['delta_pct'] = round(s['delta'] / s['prev_count'] * 100, 1) if s['prev_count'] > 0 else None

    # ── Minor-level comparison within each major ──
    all_minors_with_growth = []
    # Build 3-month avg lookup for minors
    prev_minor_counts = defaultdict(list)
    for pm in prev_months:
        for ph in pm.get('type_hierarchy', []):
            for pmn in ph.get('minors', []):
                prev_minor_counts[(ph['major'], pmn['name'])].append(pmn['count'])

    for h in current['type_hierarchy']:
        major_name = h['major']
        lm_minors = {}
        for lh in last_month.get('type_hierarchy', []):
            if lh['major'] == major_name:
                lm_minors = {m['name']: m['count'] for m in lh.get('minors', [])}
                break
        for m in h['minors']:
            # vs last month
            if m['name'] in lm_minors:
                m['prev_count'] = lm_minors[m['name']]
                m['delta'] = m['count'] - m['prev_count']
                m['delta_pct'] = round(m['delta'] / m['prev_count'] * 100, 1) if m['prev_count'] > 0 else None
            # vs 3-month avg
            prev_counts = prev_minor_counts.get((major_name, m['name']), [])
            if prev_counts:
                m['avg_3m'] = round(sum(prev_counts) / len(prev_counts), 1)
                m['vs_3m_diff'] = round(m['count'] - m['avg_3m'], 1)
                m['vs_3m_pct'] = round((m['count'] - m['avg_3m']) / m['avg_3m'] * 100, 1) if m['avg_3m'] > 0 else None
            if m.get('vs_3m_pct') is not None:
                all_minors_with_growth.append(m)
    # Rank minors by |vs_3m_pct|, only those with |vs_3m_diff| > 30 qualify
    all_minors_with_growth.sort(key=lambda x: -abs(x['vs_3m_pct']))
    rank_counter = 0
    for m in all_minors_with_growth:
        m['growth_rank'] = None
        m['growth_tag'] = None
        if m.get('vs_3m_diff') is not None and abs(m['vs_3m_diff']) > 30:
            rank_counter += 1
            if rank_counter <= 20:
                m['growth_rank'] = rank_counter
                m['growth_tag'] = 'red' if rank_counter <= 10 else 'orange'
        elif m.get('vs_3m_pct') is not None and abs(m['vs_3m_pct']) >= 50:
            m['growth_tag'] = 'anomaly'

    # ── Hotspot detection (两层逻辑) ──
    current['hotspots'] = _detect_hotspots(current, min_base=30, min_growth_pct=20)

    # ── Subtype growth ranking (filter: |vs_3m_pct| >= 10 AND avg_3m >= 30) ──
    current['subtype_growth_ranking'] = sorted(
        [s for s in current['subtype_ranking']
         if s.get('vs_3m_pct') is not None and abs(s['vs_3m_pct']) >= 10 and s.get('avg_3m', 0) >= 30],
        key=lambda s: -abs(s['vs_3m_pct'])
    )[:25]

    return current


def _detect_hotspots(stats, min_base=30, min_growth_pct=20):
    """
    两层逻辑筛选热点：
    第一层：近三月均值 >= min_base（过滤量太小的）
    第二层：当月 vs 近三月均值涨幅 >= min_growth_pct
    """
    hotspots = []
    for h in stats.get('type_hierarchy', []):
        if h.get('is_non_hr'):
            continue
        avg = h.get('avg_3m', 0)
        if avg < min_base:
            continue
        pct = h.get('vs_3m_pct')
        if pct is not None and abs(pct) >= min_growth_pct:
            hotspots.append({
                'name': h['major'],
                'count': h['count'],
                'avg_3m': avg,
                'vs_3m_pct': pct,
                'is_increase': pct > 0,
                'direction': '↑' if pct > 0 else '↓',
            })
    hotspots.sort(key=lambda x: -abs(x['vs_3m_pct']))
    return hotspots


def generate_monthly_summary(stats):
    """Generate text summary for the monthly report."""
    total = stats['total_cases']
    cards = stats.get('card_comparisons', {})
    lines = []

    # Overview
    lines.append(f'## 一、整体概览\n')
    lines.append(f'本月 China 共处理 **{total}** 个 Case。')

    sla_c = cards.get('sla', {})
    if sla_c.get('current') is not None:
        lines.append(f"SLA 达标率 **{sla_c['current']}%**。")

    csat_c = cards.get('csat_score', {})
    if csat_c.get('current') is not None:
        lines.append(f"CSAT 满意度均分 **{csat_c['current']}/5**。")

    rr_c = cards.get('rr', {})
    if rr_c.get('current') is not None:
        vs_3m = rr_c.get('vs_3m_diff', 0)
        note = '高于近三月均值' if vs_3m > 0 else '低于近三月均值，需关注'
        lines.append(f"CSAT 回复率 **{rr_c['current']}%**（{note}）。")

    # Daily trend summary
    daily = stats.get('daily_trend', [])
    if daily:
        counts = [d['count'] for d in daily]
        avg_daily = round(sum(counts) / len(counts), 1)
        peak = max(daily, key=lambda d: d['count'])
        valley = min(daily, key=lambda d: d['count'])
        lines.append(f'\n## 二、每日趋势\n')
        lines.append(f'日均 {avg_daily} 个，峰值 {peak["date"]}（{peak["count"]} 个），'
                     f'最低 {valley["date"]}（{valley["count"]} 个）。')

    # Type distribution
    lines.append(f'\n## 三、类型分布\n')
    top3 = stats['type_hierarchy'][:3]
    top3_str = '、'.join(f"{h['major']}（{h['count']}）" for h in top3)
    lines.append(f'TOP3 大类：{top3_str}。')
    lines.append(f'Non-HR 转出 **{stats["non_hr_count"]}** 个，占比 {round(stats["non_hr_count"]/total*100, 1)}%。')

    # Hotspots
    hotspots = stats.get('hotspots', [])
    if hotspots:
        lines.append(f'\n## 四、热点洞察\n')
        for h in hotspots[:5]:
            lines.append(f'- {h["direction"]} **{h["name"]}**：当月 {h["count"]}，'
                         f'近三月均值 {h["avg_3m"]}，变化 {h["vs_3m_pct"]:+.1f}%')

    # Subtype growth
    growth = stats.get('subtype_growth_ranking', [])[:8]
    if growth:
        lines.append(f'\n## 五、子类变化\n')
        lines.append('| 子类 | 当月 | 三月均值 | 变化 |')
        lines.append('|------|------|---------|------|')
        for s in growth:
            lines.append(f'| {s["subtype"]} ({s["major"]}) | {s["count"]} | {s.get("avg_3m", "—")} | {s.get("vs_3m_pct", 0):+.1f}% |')

    lines.append(f'\n---\n*由 Case Intelligence Platform 自动生成*')
    return '\n'.join(lines)


def generate_analysis_line(stats, last_month_label=''):
    """Generate the one-line analysis summary displayed below the trend chart."""
    total = stats['total_cases']
    cards = stats.get('card_comparisons', {})
    tc = cards.get('total', {})

    parts = [f'Total case amount: {total}']

    vs_last = tc.get('vs_last')
    vs_last_pct = tc.get('vs_last_pct')
    if vs_last is not None and vs_last_pct is not None:
        direction = 'increase' if vs_last > 0 else 'decrease'
        lm = f' ({last_month_label})' if last_month_label else ''
        parts.append(f' with {direction} of {abs(vs_last)} cases compared to last month{lm} ({vs_last_pct:+.1f}%)')

    sla = stats.get('sla_compliance', 0) or 0
    ftf = stats.get('ftf_rate', 0) or 0
    csat = stats.get('csat_score', 0) or 0
    rr = stats.get('csat_rr', 0) or 0
    reopen = stats.get('reopen_rate', 0) or 0

    parts.append(f'. SLA performance is {sla}%, FCF result is {ftf}%, survey result is {csat}%, CSAT RR is {rr}% and reopen rate is {reopen}%.')

    return ''.join(parts)


def prepare_ai_prompt(stats):
    """Prepare prompt for AI analysis."""
    lines = ['## China HR Case 数据\n']
    lines.append(f"总 Case: {stats['total_cases']}")
    lines.append(f"HR 相关: {stats['hr_count']} / Non-HR: {stats['non_hr_count']}")
    lines.append(f"SLA 达标率: {stats.get('sla_compliance', 'N/A')}%")
    lines.append(f"CSAT: {stats.get('csat_score', 'N/A')}/5, RR: {stats.get('csat_rr', 'N/A')}%")

    lines.append('\n### Case Type 大类')
    for h in stats['type_hierarchy'][:10]:
        vs = ''
        if h.get('vs_3m_pct') is not None:
            vs = f' (vs近三月: {h["vs_3m_pct"]:+.1f}%)'
        lines.append(f"- {h['major']}: {h['count']} ({h['percentage']}%){vs}")

    lines.append('\n### 热点')
    for h in stats.get('hotspots', [])[:5]:
        lines.append(f"- {h['direction']} {h['name']}: {h['vs_3m_pct']:+.1f}%")

    lines.append('\n### 高频子类')
    for s in stats.get('subtype_ranking', [])[:10]:
        lines.append(f"- {s['subtype']} ({s['major']}): {s['count']}条 ({s['percentage']}%)")

    return '\n'.join(lines)


# ── Internal ────────────────────────────────────────────────────────────────────

def _subtype_ranking(type_hierarchy, total, threshold=30):
    """Flat subtype ranking across all majors, excluding Non-HR, threshold by avg."""
    all_subtypes = []
    for h in type_hierarchy:
        if h['is_non_hr']:
            continue
        for m in h['minors']:
            all_subtypes.append({
                'subtype': m['name'],
                'major': h['major'],
                'count': m['count'],
                'percentage': m['percentage'],
                'avg_3m': None,
                'vs_3m_pct': None,
                'vs_3m_diff': None,
                'prev_count': None,
                'delta': None,
                'delta_pct': None,
            })
    all_subtypes.sort(key=lambda x: -x['count'])
    return all_subtypes


def parse_historical(filepath):
    """Parse the 12-months historical metrics file. Returns {metric: [{month, china_value}, ...]}."""
    df = pd.read_excel(filepath, sheet_name='12months', header=None)

    # Section header rows: Case Count=1, SLA=11, FTF=21, CSAT=31, CSAT RR=41
    # Each section: header, months row, then 6 data rows (AG, AP, EMEA, China, Avg, Target)
    sections = {
        'case_count': 1,
        'sla': 11,
        'ftf': 21,
        'csat': 31,
        'csat_rr': 41,
    }

    # Months are in row 2, cols 1-13
    months = []
    for c in range(1, 14):
        month_val = df.iloc[2, c]
        if pd.notna(month_val):
            months.append(str(month_val))

    result = {}
    for section_name, header_row in sections.items():
        series = []
        # Data rows: header_row+2 through header_row+7 (6 rows)
        for r in range(header_row + 2, header_row + 8):
            if r >= len(df):
                break
            geo = str(df.iloc[r, 0]).strip()
            if 'CHINA' in geo.upper():
                for c in range(1, min(14, len(df.columns))):
                    if c - 1 < len(months):
                        val = df.iloc[r, c]
                        if pd.notna(val) and isinstance(val, (int, float)):
                            series.append({
                                'month': months[c - 1],
                                'value': round(float(val), 2),
                            })
                break
        if series:
            result[section_name] = series

    return result


def _daily_trend(df):
    if 'Created' not in df.columns:
        return []
    valid = df.dropna(subset=['Created'])
    if len(valid) == 0:
        return []
    daily = valid.set_index('Created').resample('D').size()
    return [{'date': d.strftime('%m-%d'), 'count': int(c)} for d, c in daily.items() if c > 0]


def _date_range(df):
    if 'Created' not in df.columns:
        return {'start': 'N/A', 'end': 'N/A', 'days': 0}
    valid = df['Created'].dropna()
    if len(valid) == 0:
        return {'start': 'N/A', 'end': 'N/A', 'days': 0}
    return {
        'start': valid.min().strftime('%Y-%m-%d'),
        'end': valid.max().strftime('%Y-%m-%d'),
        'days': max(1, (valid.max() - valid.min()).days),
    }


def _fmt_dist(counter, total):
    return [
        {'name': str(k), 'count': int(v), 'percentage': round(v / total * 100, 1)}
        for k, v in sorted(counter.items(), key=lambda x: -x[1])
    ]
