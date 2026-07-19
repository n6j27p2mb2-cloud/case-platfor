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

INSIGHT_THRESHOLDS = {
    'nonhr_pct_alert': 15.0,        # Non-HR 占比 > 15% → 分流建议
    'csat_low_pct': 60.0,           # CSAT 回收率 < 60% → 满意度建议
    'lt1min_high_pct': 70.0,        # 1分钟内闭环率 > 70% → 自助化潜力
    'subtype_concentration': 8.0,   # 子类占比 > 8% → 值得关注
    'eom_peak_ratio': 1.30,         # (不再使用，保留占位)
    'sla_alert_pct': 95.0,          # SLA 达成率 < 95% → 预警
}

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


def _subtype_ranking(type_hierarchy, total, top_n=20):
    """Flat subtype ranking across all majors, excluding Non-HR."""
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
            })
    all_subtypes.sort(key=lambda x: -x['count'])
    return all_subtypes[:top_n]


def _chat_heatmap(df):
    """Build chat-only hourly heatmap: per-day matrix + hourly average.
    Only Chat origin, hours 9-18. Returns (heatmap_data, hourly_avg, max_val)."""
    if 'Origin' not in df.columns or 'Created' not in df.columns:
        return [], [], 0

    chat = df[(df['Origin'] == 'Chat') & df['Created'].notna()].copy()
    if len(chat) == 0:
        return [], [], 0

    chat['_date'] = chat['Created'].dt.strftime('%m-%d')
    chat['_hour'] = chat['Created'].dt.hour

    # Filter 9-18
    chat = chat[(chat['_hour'] >= 9) & (chat['_hour'] <= 18)]

    dates = sorted(chat['_date'].unique())
    hours = list(range(9, 19))  # 9..18

    # Per-day matrix
    day_hour_counts = chat.groupby(['_date', '_hour']).size()
    max_val = 0
    heatmap_data = []
    for d in dates:
        row = []
        for h in hours:
            c = int(day_hour_counts.get((d, h), 0))
            row.append(c)
            if c > max_val:
                max_val = c
        heatmap_data.append({'date': d, 'hours': row})

    # Hourly average (across all days that have chat data)
    workdays = len(dates) or 1
    hourly_counts = chat.groupby('_hour').size()
    hourly_avg = [
        {'hour': f'{h}:00', 'avg': round(int(hourly_counts.get(h, 0)) / workdays, 1)}
        for h in hours
    ]

    return heatmap_data, hourly_avg, max_val


def _processing_efficiency(df):
    """Compute processing duration stats for Closed cases only.
    Returns {median_sec, p90_sec, avg_sec, lt1min_n, lt1min_pct, total_closed}."""
    if 'Created' not in df.columns or 'Last Close Date' not in df.columns:
        return {'median_sec': None, 'p90_sec': None, 'avg_sec': None,
                'lt1min_n': 0, 'lt1min_pct': 0, 'total_closed': 0}

    closed = df[
        (df['Condition'] == 'Closed') &
        df['Created'].notna() &
        df['Last Close Date'].notna()
    ]
    if len(closed) == 0:
        return {'median_sec': None, 'p90_sec': None, 'avg_sec': None,
                'lt1min_n': 0, 'lt1min_pct': 0, 'total_closed': 0}

    durations = []
    for _, row in closed.iterrows():
        dur = (row['Last Close Date'] - row['Created']).total_seconds()
        if dur >= 0:
            durations.append(dur)

    if not durations:
        return {'median_sec': None, 'p90_sec': None, 'avg_sec': None,
                'lt1min_n': 0, 'lt1min_pct': 0, 'total_closed': len(closed)}

    durations.sort()
    n = len(durations)
    lt1min_n = sum(1 for d in durations if d < 60)

    return {
        'median_sec': durations[n // 2],
        'p90_sec': durations[int(n * 0.9)],
        'avg_sec': sum(durations) / n,
        'lt1min_n': lt1min_n,
        'lt1min_pct': round(lt1min_n / n * 100, 1),
        'total_closed': len(closed),
    }


def _sla_stats(df):
    """SLA violation stats. Returns {violated, total_checked, compliance_rate}."""
    if 'SLA Is Violated' not in df.columns:
        return {'violated': 0, 'total_checked': 0, 'compliance_rate': None}

    col = df['SLA Is Violated']
    violated = int(sum(
        col.apply(lambda x: str(x).strip().lower() in ('true', '1', 'yes'))
    ))
    return {
        'violated': violated,
        'total_checked': len(df),
        'compliance_rate': round((len(df) - violated) / len(df) * 100, 1) if len(df) > 0 else None,
    }


def _fmt_duration(sec):
    """Format seconds to human-readable string."""
    if sec is None:
        return '—'
    if sec < 60:
        return f'{sec:.0f}秒'
    if sec < 3600:
        return f'{sec/60:.1f}分钟'
    return f'{sec/3600:.1f}小时'


def generate_insight_text(stats):
    """Generate a Chinese markdown summary from stats using rule engine."""
    lines = []
    lines.append('## 本月摘要')
    lines.append('')
    lines.append(f"**统计区间**：{stats['date_range']['start']} → {stats['date_range']['end']}  "
                 f"（{stats['date_range']['days']} 天）")
    lines.append('')

    # 一、总量
    lines.append('### 一、总量概览')
    total = stats['total_cases']
    origin = stats['origin_distribution']
    chat_n = next((d['count'] for d in origin if d['name'] == 'Chat'), 0)
    email_n = next((d['count'] for d in origin if d['name'] == 'Email'), 0)
    lines.append(f'本月共处理 **{total}** 个 Case。')
    if chat_n or email_n:
        parts = []
        if chat_n:
            parts.append(f'Chat {chat_n}（{round(chat_n/total*100,1)}%）')
        if email_n:
            parts.append(f'Email {email_n}（{round(email_n/total*100,1)}%）')
        lines.append(f'渠道：{"、".join(parts)}。')
    sla = stats.get('sla_stats', {})
    if sla.get('compliance_rate') is not None:
        viol_text = '全部达标' if sla['violated'] == 0 else f"{sla['violated']}个违规"
        lines.append(f"SLA 达成率 **{sla['compliance_rate']}%**（{viol_text}）。")
    lines.append('')

    # 二、处理效率
    eff = stats.get('processing_efficiency', {})
    if eff.get('median_sec') is not None:
        lines.append('### 二、处理效率')
        lines.append(f"中位处理时长 **{_fmt_duration(eff['median_sec'])}**，"
                     f"P90 {_fmt_duration(eff['p90_sec'])}。")
        if eff['lt1min_pct'] > INSIGHT_THRESHOLDS['lt1min_high_pct']:
            lines.append(f"{eff['lt1min_n']} 个 Case（{eff['lt1min_pct']}%）在 1 分钟内闭环，"
                         f"秒答型为主，自助化潜力大。")
        else:
            lines.append(f"{eff['lt1min_n']} 个 Case（{eff['lt1min_pct']}%）在 1 分钟内闭环。")
        lines.append('')

    # 三、结构分布
    lines.append('### 三、结构分布')
    top3 = stats['type_hierarchy'][:3]
    top3_str = '、'.join(f"{h['major']}({h['count']})" for h in top3)
    lines.append(f'Case 量 TOP3 大类：{top3_str}。')
    lines.append(f"Non-HR 转出 {stats['non_hr_count']} 个，占比 {round(stats['non_hr_count']/total*100,1)}%。")

    hr_top = [h for h in stats['type_hierarchy'] if not h['is_non_hr']]
    if hr_top:
        hr_total = sum(h['count'] for h in hr_top)
        lines.append(f"HR 内部最大类 **{hr_top[0]['major']}**"
                     f"（{hr_top[0]['count']} 个，占 HR 总量 {round(hr_top[0]['count']/hr_total*100,1)}%）。")
    lines.append('')

    # 子类排名 Top 5
    ranking = stats.get('subtype_ranking', [])
    if ranking:
        lines.append('### 四、高频子类 Top 5')
        for i, r in enumerate(ranking[:5], 1):
            lines.append(f"{i}. **{r['subtype']}**（{r['major']}）：{r['count']} 条（{r['percentage']}%）")
        lines.append('')

    # 五、质量信号
    res = stats.get('resolution_counts', {})
    csat_total = res.get('CSAT', 0)
    closed_total = eff.get('total_closed', total)
    csat_pct = round(csat_total / closed_total * 100, 1) if closed_total > 0 else 0
    no_csat = res.get('No CSAT', 0)
    third_party = res.get('3rd Party', 0)
    dup = res.get('Duplicate', 0)
    lines.append('### 五、质量信号')
    lines.append(f"CSAT 回收率 {csat_pct}%（{csat_total} / {closed_total}），"
                 f"No CSAT {no_csat} 个，转三方 {third_party} 个，重复工单 {dup} 个。")
    lines.append('')

    # 六、建议
    suggestions = []
    nonhr_pct = round(stats['non_hr_count'] / total * 100, 1) if total > 0 else 0
    if nonhr_pct > INSIGHT_THRESHOLDS['nonhr_pct_alert']:
        suggestions.append(f"Non-HR 占比 {nonhr_pct}% 偏高，建议优化员工自助导航分流")
    if csat_pct < INSIGHT_THRESHOLDS['csat_low_pct'] and closed_total > 0:
        suggestions.append(f"CSAT 回收率 {csat_pct}% 偏低（<{INSIGHT_THRESHOLDS['csat_low_pct']}%），建议优化结束话术")
    if sla.get('compliance_rate') is not None and sla['compliance_rate'] < INSIGHT_THRESHOLDS['sla_alert_pct']:
        suggestions.append(f"SLA 达成率 {sla['compliance_rate']}% 低于 {INSIGHT_THRESHOLDS['sla_alert_pct']}%，需关注")
    if eff.get('lt1min_pct', 0) > INSIGHT_THRESHOLDS['lt1min_high_pct']:
        suggestions.append(f"1分钟内闭环率达 {eff['lt1min_pct']}%，可考虑 FAQ/知识库自助化")

    if suggestions:
        lines.append('### 六、建议行动')
        for i, s in enumerate(suggestions, 1):
            lines.append(f'{i}. {s}')
        lines.append('')

    lines.append('---')
    lines.append('*由 Case Intelligence Platform 规则引擎自动生成*')

    return '\n'.join(lines)


def _build_scatter(chat_heatmap):
    """Convert heatmap data to Chart.js scatter format [{x: hour, y: count}]."""
    points = []
    for day in chat_heatmap:
        for i, count in enumerate(day['hours']):
            if count > 0:
                points.append({'x': 9 + i, 'y': count})
    return points


def _section_insights(stats):
    """Generate per-section analysis snippets (rule engine, no AI)."""
    total = stats['total_cases']
    insights = {}

    # Section 1: 大类 + 构成 + 来源
    top_hr = [h for h in stats['type_hierarchy'] if not h['is_non_hr']]
    if top_hr:
        insights['section1'] = (
            f"本月 HR 相关 Case 以 {top_hr[0]['major']} 为主导"
            f"（{top_hr[0]['count']} 条，{top_hr[0]['percentage']}%），"
            f"Non-HR 转出 {stats['non_hr_count']} 条（{round(stats['non_hr_count']/total*100,1)}%）。"
        )
        chat_n = sum(d['count'] for d in stats['origin_distribution'] if d['name'] == 'Chat')
        if chat_n / total > 0.85:
            insights['section1'] += f" Chat 渠道占比 {round(chat_n/total*100,1)}%，自助化改造空间大。"

    # Section 2: 子类排名
    ranking = stats.get('subtype_ranking', [])
    if ranking:
        top = ranking[0]
        insights['section2'] = (
            f"刨除 Non-HR 后，最高频子类为 {top['subtype']}"
            f"（{top['major']}，{top['count']} 条，{top['percentage']}%），"
        )
        if len(ranking) >= 3:
            top3_total = sum(r['count'] for r in ranking[:3])
            insights['section2'] += (
                f"Top 3 子类合计 {top3_total} 条"
                f"（{round(top3_total/total*100,1)}%）。"
            )

    # Section 3: Chat 进线
    hourly = stats.get('chat_hourly_avg', [])
    if hourly:
        peak = max(hourly, key=lambda h: h['avg'])
        insights['section3'] = (
            f"Chat 进线高峰在 {peak['hour']}（日均 {peak['avg']} 条），"
        )
        if peak['avg'] >= 3:
            insights['section3'] += "建议此时段保持充足人力。"
        else:
            insights['section3'] += "整体进线量不高，当前人力可覆盖。"

    # Section 4: 处理效率
    eff = stats.get('processing_efficiency', {})
    if eff.get('median_sec') is not None:
        if eff['lt1min_pct'] > 70:
            insights['section4'] = (
                f"{eff['lt1min_pct']}% 的 Case 在 1 分钟内闭环，"
                f"中位处理时长仅 {_fmt_duration(eff['median_sec'])}，"
                f"说明大部分为秒答型问题。P90 {_fmt_duration(eff['p90_sec'])}，"
                f"复杂 Case 仍有优化空间。"
            )
        else:
            insights['section4'] = (
                f"中位处理时长 {_fmt_duration(eff['median_sec'])}，"
                f"P90 {_fmt_duration(eff['p90_sec'])}。"
            )

    # Section 5: 地区 + 员工
    prov = stats.get('province_summary', [])
    emp = stats.get('employee_type_stats', [])
    if prov:
        insights['section5'] = f"{prov[0]['province']} Case 量最高（{prov[0]['total']} 条）。"
        if emp:
            internal = [e for e in emp if '系统内' in e['type']]
            if internal:
                insights['section5'] += f" 员工来源以{internal[0]['type']}为主（{internal[0]['count']} 条）。"

    # Section 6: 工作量 + 解决
    owners = stats.get('owner_stats', [])
    res = stats.get('resolution_counts', {})
    if owners:
        insights['section6'] = f"团队 {len(owners)} 人，工作量最高为 {owners[0]['name']}（{owners[0]['count']} 条）。"
        csat = res.get('CSAT', 0)
        no_csat = res.get('No CSAT', 0)
        closed_total = eff.get('total_closed', total)
        csat_pct = round(csat / closed_total * 100, 1) if closed_total > 0 else 0
        if csat_pct < 60:
            insights['section6'] += f" CSAT 回收率 {csat_pct}% 偏低，建议推动满意度收集。"

    return insights


def _growth_alerts(type_hierarchy, threshold_pct=30):
    """Detect items with MoM growth > threshold. Returns list of {name, delta_pct}."""
    alerts = []
    for h in type_hierarchy:
        if h.get('delta_pct') and abs(h['delta_pct']) >= threshold_pct:
            alerts.append({
                'name': h['major'],
                'delta': h['delta'],
                'delta_pct': h['delta_pct'],
                'is_increase': h['delta'] > 0,
            })
    # Sort by abs delta_pct descending
    alerts.sort(key=lambda a: -abs(a['delta_pct']))
    return alerts


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

    # ── Daily trend ──
    daily_trend = _daily_trend(df)

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

    # ── NEW: Subtype cross-category ranking ──
    subtype_ranking = _subtype_ranking(type_hierarchy, total)

    # ── NEW: Chat heatmap ──
    chat_heatmap, chat_hourly_avg, heatmap_max = _chat_heatmap(df)
    chat_scatter = _build_scatter(chat_heatmap)

    # ── NEW: Processing efficiency ──
    processing_efficiency = _processing_efficiency(df)

    # ── NEW: SLA stats ──
    sla_stats = _sla_stats(df)

    stats = {
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
        'daily_trend': daily_trend,
        'origin_distribution': _fmt_dist(origin_counts, total),
        'resolution_counts': resolution_counts,
        'date_range': date_range,
        'hot_keywords': hot_keywords,
        'case_breakdown': case_breakdown,
        # New fields
        'subtype_ranking': subtype_ranking,
        'chat_heatmap': chat_heatmap,
        'chat_hourly_avg': chat_hourly_avg,
        'chat_scatter': chat_scatter,
        'heatmap_max': heatmap_max,
        'processing_efficiency': processing_efficiency,
        'sla_stats': sla_stats,
        'section_insights': _section_insights({
            'total_cases': total, 'non_hr_count': non_hr_count,
            'type_hierarchy': type_hierarchy, 'origin_distribution': _fmt_dist(origin_counts, total),
            'subtype_ranking': subtype_ranking, 'chat_hourly_avg': chat_hourly_avg,
            'processing_efficiency': processing_efficiency, 'province_summary': province_summary,
            'employee_type_stats': employee_type_stats, 'owner_stats': owner_stats,
            'resolution_counts': resolution_counts,
        }),
    }

    # Keep insight_text for Word report only (not displayed on dashboard)
    stats['insight_text'] = generate_insight_text(stats)

    return stats


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

    # Growth alerts: items with MoM change > 30%
    current['growth_alerts'] = _growth_alerts(current['type_hierarchy'], threshold_pct=30)

    # Subtype growth: compare subtype_ranking between months
    prev_subtype_map = {}
    if previous:
        for s in previous.get('subtype_ranking', []):
            prev_subtype_map[(s['major'], s['subtype'])] = s['count']
    for s in current.get('subtype_ranking', []):
        key = (s['major'], s['subtype'])
        pcount = prev_subtype_map.get(key, 0)
        s['prev_count'] = pcount
        if pcount > 0:
            s['delta'] = s['count'] - pcount
            s['delta_pct'] = round(s['delta'] / pcount * 100, 1)
        else:
            s['delta'] = s['count'] if pcount == 0 else 0
            s['delta_pct'] = None  # New subtype, no baseline

    # Subtype growth alerts
    current['subtype_alerts'] = [
        {'subtype': s['subtype'], 'major': s['major'],
         'delta': s['delta'], 'delta_pct': s['delta_pct'],
         'is_increase': s['delta'] > 0}
        for s in current.get('subtype_ranking', [])
        if s.get('delta_pct') and abs(s['delta_pct']) >= 30
    ]
    current['subtype_alerts'].sort(key=lambda a: -abs(a['delta_pct']))

    return current


# ── Internal ─────────────────────────────────────────────────────────────────

def _daily_trend(df):
    if 'Created' not in df.columns:
        return []
    valid = df.dropna(subset=['Created'])
    if len(valid) == 0:
        return []
    daily = valid.set_index('Created').resample('D').size()
    return [{'date': d.strftime('%m-%d'), 'count': int(c)} for d, c in daily.items() if c > 0]


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
