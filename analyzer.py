import re
import warnings
from pathlib import Path

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
    """Normalize origin to: Chat / Email / HROA/Workday单据."""
    if pd.isna(origin):
        return 'HROA/Workday单据'
    s = str(origin).strip().lower()
    if 'chat' in s or 'manual' in s:
        return 'Chat'
    if 'email' in s:
        return 'Email'
    return 'HROA/Workday单据'


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

# ── Column-name alias keys (lowercase substrings) ───────────────────────────────
# Used to auto-detect columns regardless of exact export naming.

_CASE_DETAIL_KEYS = {
    'CaseID': ['caseid', 'case id', 'case number', 'casenumber', 'case no'],
    'CaseType': ['casetype', 'case type', 'case_type', 'case category', 'category'],
    'Created': ['created', 'create date', 'date created', 'created date', 'open date', 'opened'],
    'Origin': ['origin', 'source', 'channel'],
    'GEO': ['geo', 'region', 'geography'],
    'Title': ['title', 'subject', 'summary'],
    'FirstCloseResolution': ['first close resolution', 'last close resolution', 'close resolution', 'resolution'],
    'Owner': ['owner login name', 'owner login', 'owner'],
    'Individual': ['individual', 'assigned to', 'assignee'],
    'SLA Met Flag': ['sla met flag', 'sla met', 'sla status'],
    'FTF': ['ftf', 'fcf', 'first time fix', 'first contact fix', 'first time resolution'],
    'Reopen': ['reopen'],
    'Role': ['role', 'queue name', 'queue'],
}

_CSAT_SUB_KEYS = ['interaction', 'interact', 'solution quality', 'pers.serv', 'pers serv',
                  'professionalism', 'serv', 'comm', 'clear']
_CSAT_OVERALL_KEYS = ['avarage', 'average', 'satisfaction', 'survey result', 'csat score', 'csat']
# Distinctive columns that reliably mark a CSAT sheet (avoid false positives like
# "No CSAT Count" / "Average Time" on the main case sheet).
_CSAT_SHEET_KEYS = ['interaction', 'interact', 'pers.serv', 'pers serv', 'avarage']
_MONTH_KEYS = ['month', 'period', 'yearmonth', 'year-month']
_KPI_KEYS = ['sla', 'ftf', 'fcf', 'response rate', 'case volume', 'csat rr', 'case count']


def _norm(s):
    return str(s).strip().lower()


def _match_col(cols, keys):
    """Return the first column whose lowercased name contains any key."""
    for col in cols:
        c = _norm(col)
        if any(k in c for k in keys):
            return col
    return None


def _classify_sheet(cols, sheet_name):
    """Classify a sheet by its header content (not its name)."""
    c = [_norm(x) for x in cols]
    name = _norm(sheet_name)

    has_month = any(any(k in x for k in _MONTH_KEYS) for x in c)
    has_kpi = any(any(k in x for k in _KPI_KEYS) for x in c)
    if has_month and has_kpi:
        return 'matrix'

    has_csat = any(any(k in x for k in _CSAT_SHEET_KEYS) for x in c)
    if has_csat:
        return 'csat'

    if 'metric' in name or 'kpi' in name or any('target' in x for x in c):
        return 'metrics_pivot'

    has_casetype = any('casetype' in x or 'case type' in x for x in c)
    has_caseid = any('caseid' in x or 'case id' in x or 'case number' in x for x in c)
    has_created = any('created' in x for x in c)
    if has_casetype or (has_caseid and has_created):
        return 'case_detail'

    return 'unknown'


def _map_columns(cols, col_keys):
    mapping = {}
    for canonical, keys in col_keys.items():
        col = _match_col(cols, keys)
        if col is not None and col not in mapping:
            mapping[col] = canonical
    return mapping


def _read_case_detail(xls, sheet, cols):
    col_map = _map_columns(cols, _CASE_DETAIL_KEYS)
    if not col_map:
        return None
    df = pd.read_excel(xls, sheet_name=sheet, usecols=list(col_map.keys()))
    df = df.rename(columns=col_map)
    if 'GEO' in df.columns:
        df = df[df['GEO'].astype(str).str.strip() == 'China'].copy()
    return df


def _read_csat(xls, sheet, cols):
    col_map = {}
    overall_col = _match_col(cols, _CSAT_OVERALL_KEYS)
    if overall_col is not None:
        col_map[overall_col] = 'AVARAGE'
    geo_col = _match_col(cols, _CASE_DETAIL_KEYS['GEO'])
    if geo_col is not None:
        col_map[geo_col] = 'GEO'

    quality_cols = {}
    for col in cols:
        cn = _norm(col)
        if 'comment' in cn or 'feedback' in cn:
            continue
        if any(k in cn for k in ['interaction', 'interact', 'solution quality']):
            quality_cols.setdefault('interaction', col)
        elif any(k in cn for k in ['pers', 'profession', 'serv']):
            quality_cols.setdefault('pers_serv', col)
        elif any(k in cn for k in ['comm', 'clear']):
            quality_cols.setdefault('comm', col)
    for col in quality_cols.values():
        if col not in col_map:
            col_map[col] = col

    if not col_map:
        return None, None

    df = pd.read_excel(xls, sheet_name=sheet, usecols=list(col_map.keys()))
    df = df.rename(columns=col_map)
    if 'GEO' in df.columns:
        df = df[df['GEO'].astype(str).str.strip() == 'China'].copy()
    csat_quality_raw = {'cols': quality_cols, 'df': df} if quality_cols else None
    return df, csat_quality_raw


def parse_excel(filepath):
    """Parse a case-detail file (old HR Ops / New Case Query / new Ops+CSAT).
    Auto-detects sheet roles and column names.
    Returns (df, csat_df, metrics, csat_quality_raw)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xls = pd.ExcelFile(filepath, engine='openpyxl')

        df = None
        csat_df = None
        csat_quality_raw = None
        metrics = {}

        for sheet in xls.sheet_names:
            try:
                peek = pd.read_excel(xls, sheet_name=sheet, nrows=0)
            except Exception:
                continue
            cols = list(peek.columns)
            role = _classify_sheet(cols, sheet)

            if role == 'case_detail' and df is None:
                df = _read_case_detail(xls, sheet, cols)
            elif role == 'csat' and csat_df is None:
                csat_df, csat_quality_raw = _read_csat(xls, sheet, cols)
            elif role == 'metrics_pivot' and not metrics:
                try:
                    mdf = pd.read_excel(xls, sheet_name=sheet, header=None)
                except Exception:
                    mdf = None
                if mdf is not None:
                    metrics = _parse_metrics_sheet(
                        mdf, df if df is not None else pd.DataFrame(), csat_df)

    if df is None or len(df) == 0:
        return pd.DataFrame(), None, {}, None

    # ── Common processing (dates, hierarchy, origin, resolution, non-hr) ──

    for col in ['Created', 'FirstCloseDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)

    if 'CaseType' in df.columns:
        parsed = df['CaseType'].apply(_simplify_case_type)
        df['_case_major'] = parsed.apply(lambda x: x[0])
        df['_case_minor'] = parsed.apply(lambda x: x[1])
    else:
        df['_case_major'] = 'Unknown'
        df['_case_minor'] = 'Unknown'

    if 'Origin' in df.columns:
        df['_origin_class'] = df['Origin'].apply(_classify_origin)

    if 'FirstCloseResolution' in df.columns:
        df['_resolution'] = df['FirstCloseResolution'].apply(_classify_resolution)

    if 'CaseType' in df.columns:
        df['_is_non_hr'] = df['CaseType'].apply(
            lambda x: str(x).startswith('Non-HR') if pd.notna(x) else False
        )
    else:
        df['_is_non_hr'] = False

    return df, csat_df, metrics, csat_quality_raw


def _to_pct(v):
    """Normalize a KPI rate to a percentage (0-100). Assumes 0-1 means a rate."""
    if v is None:
        return None
    v = float(v)
    if 0 <= v <= 1:
        return round(v * 100, 2)
    return round(v, 2)


def _extract_china_value(sdf):
    for r in range(len(sdf)):
        for c in range(min(3, len(sdf.columns))):
            if 'china' in _norm(sdf.iloc[r, c]):
                for cc in range(c + 1, min(len(sdf.columns), c + 6)):
                    v = sdf.iloc[r, cc]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return float(v)
                for cc in range(c - 1, -1, -1):
                    v = sdf.iloc[r, cc]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return float(v)
    return None


def _parse_matrix_table(xls, sheet, cols):
    df = pd.read_excel(xls, sheet_name=sheet)
    month_col = _match_col(cols, _MONTH_KEYS)
    case_col = _match_col(cols, ['case volume', 'case count', 'total case', 'volume', 'cases logged', 'case'])
    sla_col = _match_col(cols, ['sla'])
    ftf_col = _match_col(cols, ['ftf', 'fcf', 'first time fix', 'first contact fix', 'first time resolution'])
    csat_col = _match_col(cols, ['survey result', 'csat', 'satisfaction', 'csat score'])
    rr_col = _match_col(cols, ['csat rr', 'rr', 'response rate'])

    def num(v):
        if pd.isna(v):
            return None
        try:
            return float(str(v).replace('%', '').replace(',', '').strip())
        except (ValueError, TypeError):
            return None

    rows = []
    for _, row in df.iterrows():
        if month_col is None or pd.isna(row.get(month_col)):
            continue
        rows.append({
            'month': _fmt_month(row[month_col]),
            'case_count': num(row[case_col]) if case_col else None,
            'sla': num(row[sla_col]) if sla_col else None,
            'ftf': num(row[ftf_col]) if ftf_col else None,
            'csat': num(row[csat_col]) if csat_col else None,
            'csat_rr': num(row[rr_col]) if rr_col else None,
        })

    if not rows:
        return {}, None

    historical = {}
    for metric in ['case_count', 'sla', 'ftf', 'csat', 'csat_rr']:
        series = []
        for r in rows:
            v = r[metric]
            if v is None:
                continue
            if metric != 'case_count':
                v = _to_pct(v)
            series.append({'month': r['month'], 'value': v})
        if series:
            historical[metric] = series

    last = rows[-1]
    metrics = {}
    if last['case_count'] is not None:
        metrics['total_cases'] = int(round(last['case_count']))
    if last['sla'] is not None:
        metrics['sla_compliance'] = _to_pct(last['sla'])
    if last['ftf'] is not None:
        metrics['ftf_rate'] = _to_pct(last['ftf'])
    if last['csat'] is not None:
        metrics['csat_score'] = _to_pct(last['csat'])
    if last['csat_rr'] is not None:
        metrics['csat_rr'] = _to_pct(last['csat_rr'])

    return metrics, historical


def _parse_raw_metrics_report(xls, sheet_names):
    metrics = {}
    for sheet in sheet_names:
        n = _norm(sheet)
        if 'rr' in n or 'response' in n:
            key = 'csat_rr'
        elif 'sla' in n:
            key = 'sla_compliance'
        elif 'ftf' in n or 'fcf' in n or 'first' in n:
            key = 'ftf_rate'
        elif 'csat' in n or 'survey' in n or 'satisfaction' in n:
            key = 'csat_score'
        elif 'case' in n or 'volume' in n or 'logged' in n:
            key = 'total_cases'
        else:
            continue
        if key in metrics:
            continue
        try:
            sdf = pd.read_excel(xls, sheet_name=sheet, header=None)
        except Exception:
            continue
        val = _extract_china_value(sdf)
        if val is None:
            continue
        if key == 'total_cases':
            metrics[key] = int(round(val))
        else:
            metrics[key] = _to_pct(val)
    return metrics


def parse_matrix(filepath):
    """Parse a KPI/trend file into (metrics, historical).

    Supports:
      - new matrix (Month + Case Volume/SLA/FTF|FCF/Survey Result/CSAT RR columns)
      - old 12months pivot (section-based, trend only)
      - raw per-GEO metrics report (best-effort current-month KPI)
    metrics: current-month KPI (rates already x100)
    historical: {case_count|sla|ftf|csat|csat_rr: [{month, value}, ...]}
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xls = pd.ExcelFile(filepath, engine='openpyxl')
        sheet_names = xls.sheet_names

        for sheet in sheet_names:
            try:
                peek = pd.read_excel(xls, sheet_name=sheet, nrows=0)
            except Exception:
                continue
            cols = list(peek.columns)
            if _classify_sheet(cols, sheet) == 'matrix':
                return _parse_matrix_table(xls, sheet, cols)

        if any('12month' in _norm(s) or 'historical' in _norm(s) for s in sheet_names):
            try:
                return {}, parse_historical(filepath)
            except Exception:
                pass

        metrics = _parse_raw_metrics_report(xls, sheet_names)
        return metrics, None


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
        sla_col = pd.to_numeric(hrops_df['SLA Met Flag'], errors='coerce').dropna()
        if len(sla_col) == 0:
            sla_mapped = hrops_df['SLA Met Flag'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0})
            sla_col = pd.to_numeric(sla_mapped, errors='coerce').dropna()
        if len(sla_col) > 0:
            result['sla_compliance'] = round(sla_col.mean() * 100, 2)

    # FTF
    if 'ftf_rate' not in result and 'FTF' in hrops_df.columns:
        ftf_col = pd.to_numeric(hrops_df['FTF'], errors='coerce').dropna()
        if len(ftf_col) == 0:
            ftf_mapped = hrops_df['FTF'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0})
            ftf_col = pd.to_numeric(ftf_mapped, errors='coerce').dropna()
        if len(ftf_col) > 0:
            result['ftf_rate'] = round(ftf_col.mean() * 100, 2)

    # Reopen
    if 'reopen_rate' not in result and 'Reopen' in hrops_df.columns:
        reopen_col = pd.to_numeric(hrops_df['Reopen'], errors='coerce')
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
        sla_col = pd.to_numeric(df['SLA Met Flag'], errors='coerce').dropna()
        if len(sla_col) == 0:
            # Try mapping Yes/No → 1/0
            sla_mapped = df['SLA Met Flag'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0})
            sla_col = pd.to_numeric(sla_mapped, errors='coerce').dropna()
        sla_compliance = round(sla_col.mean() * 100, 2) if len(sla_col) > 0 else None

    # CSAT from raw data if not in metrics
    if csat_score is None and csat_df is not None and 'AVARAGE' in csat_df.columns:
        scores = csat_df['AVARAGE'].dropna()
        csat_score = round(scores.mean(), 2) if len(scores) > 0 else None

    if csat_rr is None and csat_df is not None:
        total_csat = len(csat_df)
        responded = len(csat_df[csat_df['AVARAGE'].notna()]) if 'AVARAGE' in csat_df.columns else 0
        csat_rr = round(responded / total_csat * 100, 2) if total_csat > 0 else None

    # Normalize CSAT to a percentage: old format yields a 1-5 mean (<=5) → x20;
    # the new matrix already yields a percentage (>5).
    if csat_score is not None and csat_score <= 5:
        csat_score = round(csat_score * 20, 2)

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

def compute_comparisons(current, prev_months, kpi_history=None):
    """
    Compute vs上月 and vs近三月均值 for the current month.
    prev_months: list of stats dicts, ordered most recent first (e.g. [May, April, March]).
    kpi_history: optional {metric: [{month, value}]} from a 12-month matrix, preferred for KPI cards.
    Returns current with added comparison fields.
    """
    if not prev_months and not kpi_history:
        current['has_comparison'] = False
        current['hotspots'] = []
        current['subtype_growth_ranking'] = []
        return current

    current['has_comparison'] = True
    last_month = prev_months[0] if prev_months else {}

    # ── Four card comparisons (matrix KPI history preferred) ──
    current['card_comparisons'] = {}
    kpi_section_map = {'total': 'case_count', 'sla': 'sla', 'csat_score': 'csat', 'rr': 'csat_rr'}

    for card_key, stat_key in [
        ('total', 'total_cases'),
        ('sla', 'sla_compliance'),
        ('csat_score', 'csat_score'),
        ('rr', 'csat_rr'),
    ]:
        cur_val = current.get(stat_key)
        lm_val = None
        avg_vals = []
        hist = (kpi_history or {}).get(kpi_section_map[card_key])
        if hist and len(hist) >= 2:
            hist_vals = [h['value'] for h in hist]
            lm_val = hist_vals[-2]
            prev_vals = hist_vals[:-1]
            avg_vals = prev_vals[-3:]
        else:
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

    # ── Origin distribution comparison ──
    current['origin_comparisons'] = []
    lm_origins = {o['name']: o for o in last_month.get('origin_distribution', [])}
    for o in current.get('origin_distribution', []):
        oc = {'name': o['name'], 'current': o['count'], 'current_pct': o['percentage']}
        if o['name'] in lm_origins:
            lm = lm_origins[o['name']]
            oc['last_count'] = lm['count']
            oc['delta'] = o['count'] - lm['count']
            oc['delta_pct'] = round((o['count'] - lm['count']) / lm['count'] * 100, 1) if lm['count'] > 0 else None
        prev_counts = []
        for pm in prev_months:
            for po in pm.get('origin_distribution', []):
                if po['name'] == o['name']:
                    prev_counts.append(po['count'])
        if prev_counts:
            oc['avg_3m'] = round(sum(prev_counts) / len(prev_counts), 1)
            oc['vs_3m_diff'] = round(o['count'] - oc['avg_3m'], 1)
            oc['vs_3m_pct'] = round((o['count'] - oc['avg_3m']) / oc['avg_3m'] * 100, 1) if oc['avg_3m'] > 0 else None
        current['origin_comparisons'].append(oc)

    # ── CSAT Quality comparison ──
    current['quality_comparisons'] = {}
    lm_quality = last_month.get('csat_quality', {})
    for key, val in current.get('csat_quality', {}).items():
        qc = {'current_avg': val['avg'], 'current_pct': val['pct'], 'current_count': val['count']}
        if key in lm_quality:
            lmv = lm_quality[key]
            qc['last_pct'] = lmv['pct']
            qc['last_avg'] = lmv['avg']
            qc['delta_pct'] = round(val['pct'] - lmv['pct'], 1)
            qc['delta_avg'] = round(val['avg'] - lmv['avg'], 2)
        current['quality_comparisons'][key] = qc

    # ── Prev months quality (for grouped bar chart) ──
    current['prev_months_quality'] = []
    for pm in prev_months:
        pq = pm.get('csat_quality', {})
        if pq:
            current['prev_months_quality'].append(pq)

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
        avg = h.get('avg_3m') or 0
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
        lines.append(f"CSAT 满意度 **{csat_c['current']}%**。")

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
    lines.append(f"CSAT: {stats.get('csat_score', 'N/A')}%, RR: {stats.get('csat_rr', 'N/A')}%")

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


def _fmt_month(val):
    """Normalize any month representation to 'Jun2025' format."""
    import datetime as _dt
    if isinstance(val, (_dt.datetime, pd.Timestamp)):
        return val.strftime('%b%Y')
    s = str(val).strip()
    if s.endswith('.0'):
        s = s[:-2]
    # Try common string patterns
    import re
    # "2025-06-01 ..." (pandas Timestamp str)
    m = re.match(r'(\d{4})-(\d{2})-\d{2}', s)
    if m:
        from calendar import month_abbr
        mi = int(m.group(2))
        return month_abbr[mi] + m.group(1)
    # Already "Jun2025"
    if re.match(r'^[A-Za-z]{3}\d{4}$', s):
        return s
    # "Jun-25" or "Jun 25"
    m = re.match(r'([A-Za-z]{3})[- ](\d{2})$', s)
    if m:
        return m.group(1).capitalize() + '20' + m.group(2)
    # "Jun25"
    m = re.match(r'([A-Za-z]{3})(\d{2})$', s)
    if m:
        return m.group(1).capitalize() + '20' + m.group(2)
    # "202506"
    m = re.match(r'(\d{4})(\d{2})$', s)
    if m:
        from calendar import month_abbr
        mi = int(m.group(2))
        return month_abbr[mi] + m.group(1)
    return s


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

    # Months are in row 2, cols 1-13 — normalize to "Jun2025" format
    months = []
    for c in range(1, 14):
        month_val = df.iloc[2, c]
        if pd.notna(month_val):
            months.append(_fmt_month(month_val))

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
