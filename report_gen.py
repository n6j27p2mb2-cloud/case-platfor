"""Generate .docx analysis report from dashboard stats."""
import io
from datetime import datetime

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn


def generate_report(stats, filename, has_comparison=False):
    doc = Document()

    # ── Page setup ──
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)

    # ── Styles ──
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(10)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

    # ── Title ──
    title = doc.add_heading('HR Case 分析报告', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        f"数据范围: {stats['date_range']['start']} → {stats['date_range']['end']} "
        f"({stats['date_range']['days']} 天)  |  数据来源: {filename}"
    ).font.size = Pt(9)

    meta2 = doc.add_paragraph()
    meta2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta2.add_run(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}").font.size = Pt(9)
    doc.add_paragraph()

    # ── Section 1: KPI ──
    doc.add_heading('一、核心指标', level=1)
    kpi_table = doc.add_table(rows=2, cols=6, style='Light Grid Accent 1')
    kpi_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    kpi_headers = ['总 Case', 'HR 相关', 'Non-HR', '系统重复', '热点类型', '涉及省份']
    kpi_values = [
        str(stats['total_cases']),
        str(stats['hr_count']),
        str(stats['non_hr_count']),
        str(stats['duplicate_count']),
        str(len(stats['hotspots'])),
        str(len(stats['province_summary'])),
    ]
    for i, h in enumerate(kpi_headers):
        kpi_table.rows[0].cells[i].text = h
        kpi_table.rows[1].cells[i].text = kpi_values[i]

    if has_comparison and stats.get('total_delta') is not None:
        p = doc.add_paragraph()
        arrow = '↑' if stats['total_delta'] > 0 else '↓' if stats['total_delta'] < 0 else '→'
        p.add_run(
            f"环比: {arrow} {abs(stats['total_delta'])} 条 "
            f"({stats.get('total_delta_pct', '—')}%) vs 上月 {stats.get('prev_total', 0)} 条"
        ).font.size = Pt(9)

    doc.add_paragraph()

    # ── Section 2: Case Type Hierarchy ──
    doc.add_heading('二、Case 大类分布', level=1)
    type_table = doc.add_table(rows=1, cols=5, style='Light Grid Accent 1')
    type_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    type_headers = ['大类', '数量', '占比', '环比变化', '主要子类']
    for i, h in enumerate(type_headers):
        type_table.rows[0].cells[i].text = h

    for item in stats['type_hierarchy']:
        row = type_table.add_row()
        row.cells[0].text = ('[Non-HR] ' if item['is_non_hr'] else '') + item['major']
        row.cells[1].text = str(item['count'])
        row.cells[2].text = f"{item['percentage']}%"
        if has_comparison and item.get('delta') is not None and item['delta'] != 0:
            a = '↑' if item['delta'] > 0 else '↓'
            row.cells[3].text = f"{a}{abs(item['delta'])} ({item.get('delta_pct', '—')}%)"
        else:
            row.cells[3].text = '—'
        minors = [f"{m['name']}({m['count']})" for m in item['minors'][:3]]
        row.cells[4].text = ', '.join(minors)

    doc.add_paragraph()

    # ── Section 3: Hotspots ──
    doc.add_heading('三、热点区域', level=1)
    if stats['hotspots']:
        for hs in stats['hotspots']:
            p = doc.add_paragraph(style='List Bullet')
            p.add_run(f"{hs['name']}: {hs['count']} 条 ({hs['percentage']}%)").bold = True
            if has_comparison and hs.get('delta') is not None and hs['delta'] != 0:
                a = '↑' if hs['delta'] > 0 else '↓'
                p.add_run(f"  环比 {a}{abs(hs['delta'])}")
    else:
        doc.add_paragraph('本月无超过 5% 占比的热点类型。')

    doc.add_paragraph()

    # ── Section 4: Province × Case Type ──
    doc.add_heading('四、地区分析', level=1)
    prov_table = doc.add_table(rows=1, cols=4, style='Light Grid Accent 1')
    prov_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(['省份', '总量', 'Top 1 类型', 'Top 2 类型']):
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

    # ── Section 5: Employee Source ──
    doc.add_heading('五、员工来源分析', level=1)
    emp_table = doc.add_table(rows=1, cols=4, style='Light Grid Accent 1')
    emp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(['来源类型', '数量', '占比', '主要咨询类型']):
        emp_table.rows[0].cells[i].text = h

    for e in stats['employee_type_stats']:
        row = emp_table.add_row()
        row.cells[0].text = e['type']
        row.cells[1].text = str(e['count'])
        row.cells[2].text = f"{e['percentage']}%"
        row.cells[3].text = ', '.join(f"{ct['type']}({ct['count']})" for ct in e['top_case_types'][:3])

    doc.add_paragraph()

    # ── Section 6: Owner ──
    doc.add_heading('六、团队工作量分布', level=1)
    owner_table = doc.add_table(rows=1, cols=3, style='Light Grid Accent 1')
    owner_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(['Owner', 'Case 数量', '占比']):
        owner_table.rows[0].cells[i].text = h

    for o in stats['owner_stats']:
        row = owner_table.add_row()
        row.cells[0].text = o['name']
        row.cells[1].text = str(o['count'])
        row.cells[2].text = f"{o['percentage']}%"

    doc.add_paragraph()

    # ── Section 7: Origin & Resolution ──
    doc.add_heading('七、Case 来源与解决方式', level=1)
    doc.add_paragraph(
        '来源分布: ' +
        ', '.join(f"{d['name']}: {d['count']} ({d['percentage']}%)"
                  for d in stats['origin_distribution'])
    )
    if stats.get('resolution_counts'):
        doc.add_paragraph(
            '解决方式: ' +
            ', '.join(f"{k}: {v}" for k, v in stats['resolution_counts'].items())
        )

    # ── Footer ──
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('— Case Intelligence Platform 自动生成 —').font.size = Pt(8)

    # ── Save to bytes ──
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
