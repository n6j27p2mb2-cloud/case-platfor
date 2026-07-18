import re
import warnings
import pandas as pd
from collections import Counter, defaultdict

warnings.filterwarnings('ignore', category=UserWarning)

# ── PII columns: dropped immediately ────────────────────────────────────────
PII_DROP = [
    'Employee Tags', 'Employee Id', 'Employee Last Name',
    'Employee Site Id', 'Employee Country', 'Employee Organization Id',
    'Employee Organization Name',
    'Concerning Employee Tags', 'Concerning Employee Country',
    'Concerning Employee Site Id',
    'Concerning', 'Concerning Employee Id',
    'Concerning Employee First Name', 'Concerning Employee Last Name',
    'Concerning Employee Job Title', 'Concerning Employee Site Name',
    'Originator First Name', 'Originator Last Name',
    'Owner Last Name', 'Owner Login Name', 'Owner Site Name',
    'Owner Department', 'Owner Workgroup',
    'Involved Party - Employee - First Name',
    'Involved Party - Employee - Last Name',
    'Involved Party - Employee - Id',
    'Involved Party - External Person - Name',
    'Involved Party - Role',
]

CHINESE_STOP_WORDS = {
    '咨询', '员工', '个人', '查询', '帮助', '入职', '离职', '相关', '进行',
    '问题', '处理', '提供', '一个', '联系', '了解', '可以', '需要', '什么',
    '没有', '这个', '他们', '还是', '怎么', '如何', '是否', '已经', '并且',
    '那个', '或者', '申请', '公司', '工作', '请问', '问一下', '想问', '帮查',
    '一下', '想问一下', '麻烦', '帮忙', '回复', '谢谢', '请问一下',
    '重复', '未咨询', '咨询未', '重复咨询',
}

CITY_PROVINCE = {
    'Beijing': '北京', 'Shanghai': '上海', 'Tianjin': '天津',
    'Shenyang': '辽宁', 'Dalian': '辽宁',
    'Shenzhen': '广东', 'Guangzhou': '广东', 'Dongguan': '广东',
    'Hangzhou': '浙江', 'Ningbo': '浙江',
    'Nanjing': '江苏', 'Wuxi': '江苏',
    'Chengdu': '四川', 'Chongqing': '重庆',
    'Wuhan': '湖北', 'Xian': '陕西', 'Xiamen': '福建', 'Fuzhou': '福建',
    'Qingdao': '山东', 'Jinan': '山东',
    'Hefei': '安徽', 'Changsha': '湖南', 'Zhengzhou': '河南',
    'Kunming': '云南', 'Nanning': '广西', 'Guiyang': '贵州',
    'Harbin': '黑龙江', 'Changchun': '吉林',
    'Shijiazhuang': '河北', 'Taiyuan': '山西',
    'Suzhou': '江苏', 'Nanchang': '江西',
    'Haikou': '海南', 'Lanzhou': '甘肃', 'Urumqi': '新疆',
    'Hohhot': '内蒙古', 'Yinchuan': '宁夏', 'Xining': '青海',
    'Lhasa': '西藏',
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _extract_province(site_name):
    if pd.isna(site_name):
        return '未知'
    parts = str(site_name).split(' - ')
    if len(parts) >= 2:
        city = parts[-1].strip()
        return CITY_PROVINCE.get(city, city)
    return str(site_name).strip()


def _classify_employee(row):
    """Classify employee into: 系统内员工 / 外部 / 其他.
    Excludes Open cases (returns None)."""
    status = str(row.get('Status', '')).strip()
    if status == 'Open':
        return None

    first_name = str(row.get('Employee First Name', '')).strip()
    priority = row.get('Priority')
    severity = row.get('Severity')
    resolution = str(row.get('Last Close Resolution', '')).strip()

    # Check 3rd party and Duplicate first
    if 'third party' in resolution.lower() or 'Duplicate' in resolution:
        return '其他'

    # external email
    if first_name.lower() == 'unknown email in':
        return '外部'

    # P1 + S2 + No CSAT → external
    if priority == 1 and severity == 2 and 'No CSAT' in resolution:
        return '外部'

    # No CSAT (must check before CSAT since "No CSAT" contains "CSAT")
    if 'No CSAT' in resolution:
        return '其他'

    # CSAT (chat or email) → internal
    if 'CSAT' in resolution:
        return '系统内员工'

    return '其他'


def _simplify_case_type(ct):
    """Split Case Type into major (before first colon) and minor (everything after)."""
    if pd.isna(ct):
        return 'Unknown', 'Unknown'
    s = str(ct).strip()
    if ':' in s:
        idx = s.index(':')
        return s[:idx].strip(), s[idx+1:].strip()
    return s, s


def _classify_resolution(res_str):
    """Merge resolution into: CSAT / Duplicate / 3rd Party / No CSAT."""
    if pd.isna(res_str):
        return 'No CSAT'
    s = str(res_str).strip()
    if 'Duplicate' in s:
        return 'Duplicate'
    if 'third party' in s.lower():
        return '3rd Party'
    # Check "No CSAT" BEFORE "CSAT" — "No CSAT" contains the substring "CSAT"
    if 'No CSAT' in s:
        return 'No CSAT'
    if 'CSAT' in s:
        return 'CSAT'
    return 'No CSAT'


def _extract_title_keywords(titles, top_n=6):
    kw_counter = Counter()
    for t in titles:
        t_clean = re.sub(r'[（(][^)）]*[)）]', '', str(t))
        chunks = re.findall(r'[一-鿿]{2,8}', t_clean)
        for chunk in chunks:
            if chunk not in CHINESE_STOP_WORDS:
                kw_counter[chunk] += 1
    return [{'keyword': k, 'count': c} for k, c in kw_counter.most_common(top_n)]


def _detect_hotspots(type_stats, threshold_pct=5):
    return [item for item in type_stats if item['percentage'] >= threshold_pct]


def _count_non_hr(df):
    if 'Case Type' not in df.columns:
        return 0
    return int(df['Case Type'].apply(
        lambda x: str(x).startswith('Non-HR') if pd.notna(x) else False
    ).sum())


# ── Main ────────────────────────────────────────────────────────────────────

def parse_excel(filepath):
    df = pd.read_excel(filepath)

    # Filter: remove "Please Select Case Type"
    if 'Case Type' in df.columns:
        df = df[df['Case Type'] != 'Please Select Case Type']

    # Drop full PII columns
    to_drop = [c for c in PII_DROP if c in df.columns]
    df = df.drop(columns=to_drop, errors='ignore')

    # Normalize dates
    for col in ['Created', 'Last Modified', 'Due Date', 'Last Close Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)

    # Province
    if 'Employee Site Name' in df.columns:
        df['_province'] = df['Employee Site Name'].apply(_extract_province)

    # Employee classification (None for Open cases)
    df['_employee_type'] = df.apply(_classify_employee, axis=1)

    # Case Type hierarchy
    if 'Case Type' in df.columns:
        parsed = df['Case Type'].apply(_simplify_case_type)
        df['_case_major'] = parsed.apply(lambda x: x[0])
        df['_case_minor'] = parsed.apply(lambda x: x[1])

    # Resolution simplified
    if 'Last Close Resolution' in df.columns:
        df['_resolution'] = df['Last Close Resolution'].apply(_classify_resolution)

    return df


def analyze(df):
    total = len(df)

    # ── Counts ──
    non_hr_count = _count_non_hr(df)
    hr_count = total - non_hr_count

    dup_count = 0
    if '_resolution' in df.columns:
        dup_count = int((df['_resolution'] == 'Duplicate').sum())

    third_party_count = 0
    if '_resolution' in df.columns:
        third_party_count = int((df['_resolution'] == '3rd Party').sum())

    # ── Case Type hierarchy (with minor-level title keywords) ──
    major_counter = Counter()
    major_minors = defaultdict(Counter)
    minor_titles = defaultdict(list)
    for _, row in df.iterrows():
        major = row.get('_case_major', 'Unknown')
        minor = row.get('_case_minor', 'Unknown')
        major_counter[major] += 1
        major_minors[major][minor] += 1
        if 'Title' in df.columns and pd.notna(row.get('Title')):
            minor_titles[(major, minor)].append(row['Title'])

    type_hierarchy = []
    for major, count in major_counter.most_common():
        pct = round(count / total * 100, 1) if total > 0 else 0
        entry = {
            'major': major,
            'count': count,
            'percentage': pct,
            'is_non_hr': major.startswith('Non-HR'),
            'minors': [],
        }
        for minor, mcount in major_minors[major].most_common():
            mpct = round(mcount / total * 100, 1) if total > 0 else 0
            keywords = _extract_title_keywords(
                [t for t in minor_titles[(major, minor)] if pd.notna(t)],
                top_n=6
            )
            entry['minors'].append({
                'name': minor,
                'count': mcount,
                'percentage': mpct,
                'keywords': keywords,
            })
        type_hierarchy.append(entry)

    # ── Hot spots ──
    hr_major_stats = [
        {'name': h['major'], 'count': h['count'], 'percentage': h['percentage']}
        for h in type_hierarchy if not h['is_non_hr']
    ]
    hotspots = _detect_hotspots(hr_major_stats, threshold_pct=5)

    # ── Province × FULL Case Type ──
    province_case = defaultdict(Counter)
    province_total = Counter()
    if '_province' in df.columns:
        for _, row in df.iterrows():
            prov = row.get('_province', '未知')
            ctype = row.get('Case Type', 'Unknown')
            if pd.notna(ctype):
                province_case[prov][str(ctype)] += 1
                province_total[prov] += 1
    province_summary = [
        {
            'province': p,
            'total': province_total[p],
            'top_types': [{'type': t, 'count': c} for t, c in province_case[p].most_common(5)],
        }
        for p, _ in province_total.most_common(15)
    ]

    # ── Employee type (closed cases only) ──
    emp_type_counter = Counter()
    emp_type_case = defaultdict(Counter)
    if '_employee_type' in df.columns:
        closed = df[df['_employee_type'].notna()]
        for _, row in closed.iterrows():
            etype = row.get('_employee_type', '其他')
            minor = row.get('_case_minor', 'Unknown')
            emp_type_counter[etype] += 1
            emp_type_case[etype][minor] += 1
    emp_total = sum(emp_type_counter.values())
    employee_type_stats = [
        {
            'type': et,
            'count': c,
            'percentage': round(c / emp_total * 100, 1) if emp_total > 0 else 0,
            'top_case_types': [
                {'type': t, 'count': tc}
                for t, tc in emp_type_case[et].most_common(5)
            ],
        }
        for et, c in emp_type_counter.most_common()
    ]

    # ── Owner workload ──
    owner_counter = Counter()
    if 'Owner First Name' in df.columns:
        for name in df['Owner First Name'].dropna():
            owner_counter[str(name).strip()] += 1
    owner_stats = [
        {'name': n, 'count': c, 'percentage': round(c / total * 100, 1) if total > 0 else 0}
        for n, c in owner_counter.most_common()
    ]

    # ── Weekly trend ──
    weekly_trend = _weekly_trend(df)

    # ── Origin ──
    origin_counts = df['Origin'].value_counts().to_dict() if 'Origin' in df.columns else {}

    # ── Resolution (simplified) ──
    resolution_counts = {}
    if '_resolution' in df.columns:
        resolution_counts = df['_resolution'].value_counts().to_dict()

    # ── Date range ──
    date_range = _date_range(df)

    # ── Hot keywords ──
    all_titles = df['Title'].dropna().tolist() if 'Title' in df.columns else []
    hot_keywords = _extract_title_keywords(all_titles, top_n=12)

    # ── Case breakdown for summary chart ──
    actual_hr = hr_count - third_party_count - dup_count
    case_breakdown = {
        'HR 实际处理': actual_hr,
        'Non-HR': non_hr_count,
        '转第三方': third_party_count,
        '系统重复': dup_count,
    }

    return {
        'total_cases': total,
        'hr_count': hr_count,
        'non_hr_count': non_hr_count,
        'duplicate_count': dup_count,
        'third_party_count': third_party_count,
        'type_hierarchy': type_hierarchy,
        'hotspots': hotspots,
        'province_summary': province_summary,
        'employee_type_stats': employee_type_stats,
        'owner_stats': owner_stats,
        'weekly_trend': weekly_trend,
        'origin_distribution': _fmt_dist(origin_counts, total),
        'resolution_counts': resolution_counts,
        'date_range': date_range,
        'hot_keywords': hot_keywords,
        'case_breakdown': case_breakdown,
    }


def compute_comparison(current, previous=None):
    if previous is None:
        prev_total = 0
        prev_type = {}
        prev_owner = {}
        prev_emp_type = {}
    else:
        prev_total = previous['total_cases']
        prev_type = {h['major']: h['count'] for h in previous['type_hierarchy']}
        prev_owner = {o['name']: o['count'] for o in previous['owner_stats']}
        prev_emp_type = {e['type']: e['count'] for e in previous['employee_type_stats']}

    cur_total = current['total_cases']
    delta = cur_total - prev_total
    current['prev_total'] = prev_total
    current['total_delta'] = delta
    current['total_delta_pct'] = round(delta / prev_total * 100, 1) if prev_total > 0 else None

    for h in current['type_hierarchy']:
        pcount = prev_type.get(h['major'], 0)
        h['prev_count'] = pcount
        h['delta'] = h['count'] - pcount
        h['delta_pct'] = round((h['count'] - pcount) / pcount * 100, 1) if pcount > 0 else None

    for hs in current['hotspots']:
        pcount = prev_type.get(hs['name'], 0)
        hs['prev_count'] = pcount
        hs['delta'] = hs['count'] - pcount
        hs['delta_pct'] = round((hs['count'] - pcount) / pcount * 100, 1) if pcount > 0 else None

    for o in current['owner_stats']:
        pcount = prev_owner.get(o['name'], 0)
        o['prev_count'] = pcount
        o['delta'] = o['count'] - pcount

    for e in current['employee_type_stats']:
        pcount = prev_emp_type.get(e['type'], 0)
        e['prev_count'] = pcount
        e['delta'] = e['count'] - pcount

    return current


# ── Internal ─────────────────────────────────────────────────────────────────

def _weekly_trend(df):
    if 'Created' not in df.columns:
        return []
    valid = df.dropna(subset=['Created'])
    if len(valid) == 0:
        return []
    weekly = valid.set_index('Created').resample('W').size()
    return [{'week': d.strftime('%m/%d'), 'count': int(c)} for d, c in weekly.items()]


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


def prepare_ai_prompt(stats):
    lines = ['## 本月 HR Case 数据\n']
    lines.append(f"总 Case: {stats['total_cases']}")
    lines.append(f"HR 相关: {stats['hr_count']} / Non-HR: {stats['non_hr_count']}")

    lines.append('\n### Case Type 大类排行')
    for h in stats['type_hierarchy'][:10]:
        delta_str = ''
        if h.get('delta') and h['delta'] != 0:
            a = '↑' if h['delta'] > 0 else '↓'
            delta_str = f' ({a}{abs(h["delta"])})'
        lines.append(f"- {h['major']}: {h['count']} ({h['percentage']}%){delta_str}")

    lines.append('\n### 员工来源')
    for e in stats['employee_type_stats']:
        lines.append(f"- {e['type']}: {e['count']} ({e['percentage']}%)")

    lines.append('\n### 省份 × Case 子类')
    for p in stats['province_summary'][:8]:
        lines.append(f"- {p['province']} ({p['total']}条): " +
                     ', '.join(f"{t['type']}({t['count']})" for t in p['top_types'][:3]))

    return '\n'.join(lines)
