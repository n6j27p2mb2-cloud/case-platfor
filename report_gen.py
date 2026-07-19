"""Generate .docx analysis report from dashboard stats with embedded charts."""
import io
import os
from datetime import datetime

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Chart style ──
PRIMARY = '#4f46e5'
NON_HR = '#f59e0b'
ACCENT = '#6366f1'
GRAY = '#9ca3af'
LIGHT_BG = '#f9fafb'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})


def _chart_to_image(fig):
    """Convert matplotlib figure to BytesIO PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def _draw_category_bar(stats):
    """Horizontal bar: case type hierarchy (top 10)."""
    types = stats['type_hierarchy'][:10]
    labels = [('Non-HR ' if h['is_non_hr'] else '') + h['major'] for h in types]
    values = [h['count'] for h in types]

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [NON_HR if h['is_non_hr'] else PRIMARY for h in types]
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel('Case Count')
    ax.set_title('Case Distribution by Category')

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=8, color='#374151')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    return _chart_to_image(fig)


def _draw_subtype_bar(stats):
    """Horizontal bar: top 15 subtypes across all categories."""
    subtypes = stats.get('subtype_ranking', [])[:15]
    if not subtypes:
        return None

    labels = [s['subtype'] for s in subtypes]
    values = [s['count'] for s in subtypes]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(labels, values, color=PRIMARY, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel('Case Count')
    ax.set_title('Top Subtypes (Cross-Category)')

    for bar, s in zip(bars, subtypes):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{s['count']} ({s['percentage']}%)", va='center', fontsize=7, color='#6b7280')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    return _chart_to_image(fig)


def _draw_heatmap_chart(stats):
    """Heatmap image: Chat hourly distribution (day x hour)."""
    heatmap_data = stats.get('chat_heatmap', [])
    if not heatmap_data:
        return None

    hours = ['9h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h']
    dates = [d['date'] for d in heatmap_data]
    matrix = [d['hours'] for d in heatmap_data]

    fig, ax = plt.subplots(figsize=(7, max(3, len(dates) * 0.25)))
    im = ax.imshow(matrix, cmap='Purples', aspect='auto')

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours, fontsize=7)
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates, fontsize=7)
    ax.set_title('Chat Incoming Heatmap (Daily x Hourly)')

    # Annotate cells with values > 0
    for i in range(len(dates)):
        for j in range(len(hours)):
            val = matrix[i][j]
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=6, color='white' if val > max(1, max(matrix[i]) / 2) else '#374151')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Cases')
    return _chart_to_image(fig)


def _draw_hourly_avg_chart(stats):
    """Bar chart: average hourly chat distribution."""
    hourly = stats.get('chat_hourly_avg', [])
    if not hourly:
        return None

    labels = [h['hour'] for h in hourly]
    values = [h['avg'] for h in hourly]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(labels, values, color=PRIMARY, width=0.6, alpha=0.85)
    ax.set_ylabel('Avg Cases')
    ax.set_title('Average Hourly Chat Distribution')

    for i, (label, val) in enumerate(zip(labels, values)):
        if val > 0:
            ax.text(i, val + max(values) * 0.02, str(val),
                    ha='center', fontsize=7, color='#374151')

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    return _chart_to_image(fig)


def _draw_resolution_donut(stats):
    """Donut: resolution breakdown."""
    res = stats.get('resolution_counts', {})
    if not res:
        return None

    labels = list(res.keys())
    values = list(res.values())
    colors = ['#4f46e5', '#9ca3af', '#f87171', '#fb923c'][:len(labels)]

    fig, ax = plt.subplots(figsize=(4, 3))
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct='%1.1f%%', startangle=90,
        colors=colors, pctdistance=0.75
    )
    ax.legend(wedges, [f'{l} ({v})' for l, v in zip(labels, values)],
              loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_title('Case Resolution')
    return _chart_to_image(fig)


# ── Main report generator ─────────────────────────────────────────────────

def generate_report(stats, filename, has_comparison=False):
    doc = Document()

    # Page setup
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)

    # Styles
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(10)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), 'PingFang SC')

    # Title
    title = doc.add_heading('HR Case Analysis Report', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        f"Period: {stats['date_range']['start']} -> {stats['date_range']['end']} "
        f"({stats['date_range']['days']} days)  |  Source: {filename}"
    ).font.size = Pt(9)

    meta2 = doc.add_paragraph()
    meta2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta2.add_run(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}").font.size = Pt(9)
    doc.add_paragraph()

    # ── 1. KPI ──
    doc.add_heading('1. Key Metrics', level=1)
    kpi_table = doc.add_table(rows=2, cols=7, style='Light Grid Accent 1')
    kpi_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    kpi_headers = ['Total', 'HR Cases', 'Non-HR', 'Duplicate', '3rd Party', 'SLA Rate', 'Provinces']
    sla_str = f"{stats.get('sla_stats', {}).get('compliance_rate', '—')}%" if stats.get('sla_stats') else '—'
    kpi_values = [
        str(stats['total_cases']), str(stats['hr_count']), str(stats['non_hr_count']),
        str(stats['duplicate_count']), str(stats['third_party_count']),
        sla_str, str(len(stats['province_summary'])),
    ]
    for i, h in enumerate(kpi_headers):
        kpi_table.rows[0].cells[i].text = h
        kpi_table.rows[1].cells[i].text = kpi_values[i]

    if has_comparison and stats.get('total_delta') is not None:
        p = doc.add_paragraph()
        arrow = 'up' if stats['total_delta'] > 0 else 'down' if stats['total_delta'] < 0 else 'flat'
        p.add_run(
            f"MoM: {arrow} {abs(stats['total_delta'])} cases "
            f"({stats.get('total_delta_pct', '—')}%) vs prev month {stats.get('prev_total', 0)} cases"
        ).font.size = Pt(9)

    doc.add_paragraph()

    # ── 2. Category Distribution + Chart ──
    doc.add_heading('2. Case Category Distribution', level=1)
    try:
        chart_img = _draw_category_bar(stats)
        doc.add_picture(chart_img, width=Inches(5.5))
    except Exception:
        doc.add_paragraph('(Chart unavailable)').font.size = Pt(9)

    doc.add_paragraph()
    type_table = doc.add_table(rows=1, cols=5, style='Light Grid Accent 1')
    type_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(['Category', 'Count', 'Pct', 'MoM', 'Top Subtypes']):
        type_table.rows[0].cells[i].text = h

    for item in stats['type_hierarchy']:
        row = type_table.add_row()
        row.cells[0].text = ('[Non-HR] ' if item['is_non_hr'] else '') + item['major']
        row.cells[1].text = str(item['count'])
        row.cells[2].text = f"{item['percentage']}%"
        if has_comparison and item.get('delta') is not None and item['delta'] != 0:
            a = '+' if item['delta'] > 0 else ''
            row.cells[3].text = f"{a}{item['delta']} ({item.get('delta_pct', '—')}%)"
        else:
            row.cells[3].text = '—'
        minors = [f"{m['name']}({m['count']})" for m in item['minors'][:3]]
        row.cells[4].text = ', '.join(minors)

    doc.add_paragraph()

    # ── 3. Subtype Ranking + Chart ──
    doc.add_heading('3. Top Subtype Issues (Cross-Category)', level=1)
    try:
        subtype_img = _draw_subtype_bar(stats)
        if subtype_img:
            doc.add_picture(subtype_img, width=Inches(5.5))
    except Exception:
        pass

    doc.add_paragraph()
    subtypes = stats.get('subtype_ranking', [])[:15]
    if subtypes:
        st_table = doc.add_table(rows=1, cols=4, style='Light Grid Accent 1')
        st_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(['Subtype', 'Category', 'Count', 'Pct']):
            st_table.rows[0].cells[i].text = h
        for s in subtypes:
            row = st_table.add_row()
            row.cells[0].text = s['subtype']
            row.cells[1].text = s['major']
            row.cells[2].text = str(s['count'])
            row.cells[3].text = f"{s['percentage']}%"

    doc.add_paragraph()

    # ── 4. Chat Heatmap ──
    doc.add_heading('4. Chat Incoming Patterns', level=1)
    try:
        hm_img = _draw_heatmap_chart(stats)
        if hm_img:
            doc.add_picture(hm_img, width=Inches(5.5))
            doc.add_paragraph()
    except Exception:
        pass

    try:
        ha_img = _draw_hourly_avg_chart(stats)
        if ha_img:
            doc.add_picture(ha_img, width=Inches(5))
    except Exception:
        pass

    doc.add_paragraph()

    # ── 5. Processing Efficiency ──
    eff = stats.get('processing_efficiency', {})
    if eff.get('median_sec') is not None:
        doc.add_heading('5. Processing Efficiency', level=1)
        eff_table = doc.add_table(rows=2, cols=5, style='Light Grid Accent 1')
        eff_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        eff_headers = ['Median', 'P90', 'Average', '<1min Rate', '<1min Count']
        eff_values = [
            f"{eff['median_sec']:.0f}s" if eff.get('median_sec') else '—',
            f"{eff['p90_sec']/3600:.1f}h" if eff.get('p90_sec') else '—',
            f"{eff['avg_sec']/60:.1f}min" if eff.get('avg_sec') else '—',
            f"{eff.get('lt1min_pct', 0)}%",
            str(eff.get('lt1min_n', 0)),
        ]
        for i, h in enumerate(eff_headers):
            eff_table.rows[0].cells[i].text = h
            eff_table.rows[1].cells[i].text = eff_values[i]
        doc.add_paragraph()

    # ── 6. Province Analysis ──
    doc.add_heading('6. Regional Analysis', level=1)
    prov_table = doc.add_table(rows=1, cols=4, style='Light Grid Accent 1')
    prov_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(['Province', 'Total', 'Top 1', 'Top 2']):
        prov_table.rows[0].cells[i].text = h
    for p in stats['province_summary'][:15]:
        row = prov_table.add_row()
        row.cells[0].text = p['province']
        row.cells[1].text = str(p['total'])
        if len(p['top_types']) >= 1:
            row.cells[2].text = f"{p['top_types'][0]['type']} ({p['top_types'][0]['count']})"
        if len(p['top_types']) >= 2:
            row.cells[3].text = f"{p['top_types'][1]['type']} ({p['top_types'][1]['count']})"

    doc.add_paragraph()

    # ── 7. Employee Source ──
    doc.add_heading('7. Employee Source', level=1)
    emp_table = doc.add_table(rows=1, cols=4, style='Light Grid Accent 1')
    emp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(['Type', 'Count', 'Pct', 'Top Case Types']):
        emp_table.rows[0].cells[i].text = h
    for e in stats['employee_type_stats']:
        row = emp_table.add_row()
        row.cells[0].text = e['type']
        row.cells[1].text = str(e['count'])
        row.cells[2].text = f"{e['percentage']}%"
        row.cells[3].text = ', '.join(f"{ct['type']}({ct['count']})" for ct in e['top_case_types'][:3])

    doc.add_paragraph()

    # ── 8. Team Workload ──
    doc.add_heading('8. Team Workload', level=1)
    owner_table = doc.add_table(rows=1, cols=3, style='Light Grid Accent 1')
    owner_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(['Owner', 'Cases', 'Pct']):
        owner_table.rows[0].cells[i].text = h
    for o in stats['owner_stats']:
        row = owner_table.add_row()
        row.cells[0].text = o['name']
        row.cells[1].text = str(o['count'])
        row.cells[2].text = f"{o['percentage']}%"

    doc.add_paragraph()

    # ── 9. Summary / Insight ──
    if stats.get('insight_text'):
        doc.add_heading('9. Monthly Insight Summary', level=1)
        for line in stats['insight_text'].split('\n'):
            clean = line.strip().lstrip('#').strip()
            if clean.startswith('---') or not clean:
                continue
            if clean.startswith('*由'):
                p = doc.add_paragraph()
                p.add_run(clean.strip('*')).font.size = Pt(8)
                continue
            if clean.startswith('**') or clean.startswith('###'):
                doc.add_paragraph(clean.replace('*', '').strip(), style='List Bullet')
            else:
                p = doc.add_paragraph(clean)
                p.style.font.size = Pt(9)

    # ── Footer ──
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('— Case Intelligence Platform Auto-Generated —').font.size = Pt(8)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
