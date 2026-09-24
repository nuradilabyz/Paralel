"""Build the formal Word report from measured CSV data and generated figures."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
OUTPUT = Path(__file__).resolve().parent / "OpenMP_Parallel_Programming_Lab_Report.docx"

NAVY = "17324D"
BLUE = "2E75B6"
PALE_BLUE = "EAF2F8"
LIGHT_GRAY = "D9D9D9"
MID_GRAY = "6B7280"
WHITE = "FFFFFF"
BLACK = "000000"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_mean(rows: list[dict[str, str]], keys: tuple[str, ...], value: str) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get(value):
            grouped[tuple(row[key] for key in keys)].append(float(row[value]))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 90, start: int = 110, bottom: int = 90, end: int = 110) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), LIGHT_GRAY)


def repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
    font_size: float = 9,
    vertical_margin: int = 90,
):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    repeat_table_header(table.rows[0])
    for index, (cell, label) in enumerate(zip(table.rows[0].cells, headers)):
        set_cell_shading(cell, NAVY)
        set_cell_margins(cell, top=vertical_margin, bottom=vertical_margin)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(label)
        run.bold = True
        run.font.color.rgb = RGBColor.from_string(WHITE)
        run.font.size = Pt(font_size)
        if widths:
            cell.width = Inches(widths[index])
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for index, (cell, value) in enumerate(zip(cells, values)):
            set_cell_margins(cell, top=vertical_margin, bottom=vertical_margin)
            if row_index % 2:
                set_cell_shading(cell, PALE_BLUE)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(str(value))
            run.font.size = Pt(font_size)
            if widths:
                cell.width = Inches(widths[index])
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(10)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(MID_GRAY)


def add_figure(doc: Document, filename: str, caption: str, width: float = 6.45) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    inline_shape = run.add_picture(str(PLOTS / filename), width=Inches(width))
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("descr", caption)
    add_caption(doc, caption)


def add_question(doc: Document, label: str, title: str, answer: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(7)
    paragraph.paragraph_format.space_after = Pt(2)
    lead = paragraph.add_run(f"{label} {title}. ")
    lead.bold = True
    paragraph.add_run(answer)


def set_repeat_header_text(section, text: str) -> None:
    header = section.header
    paragraph = header.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    paragraph.style = "Caption"
    for run in paragraph.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(MID_GRAY)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Page ")
    run.font.size = Pt(8)
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_begin)
    run._r.append(instr_text)
    run._r.append(fld_char_end)


def configure_styles(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.70)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    title = doc.styles["Title"]
    title.font.name = "Aptos Display"
    title._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
    title._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
    title.font.size = Pt(30)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(BLACK)
    title.paragraph_format.space_after = Pt(10)
    title_ppr = title._element.find(qn("w:pPr"))
    if title_ppr is not None:
        title_border = title_ppr.find(qn("w:pBdr"))
        if title_border is not None:
            title_ppr.remove(title_border)

    for style_name, size, before, after in (("Heading 1", 19, 12, 6), ("Heading 2", 14, 9, 4), ("Heading 3", 11.5, 7, 3)):
        style = doc.styles[style_name]
        style.font.name = "Aptos Display"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLACK)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    caption = doc.styles["Caption"]
    caption.font.name = "Aptos"
    caption.font.size = Pt(9)
    caption.font.italic = True
    caption.font.color.rgb = RGBColor.from_string(MID_GRAY)

    set_repeat_header_text(section, "OpenMP Paradigms in Python")
    add_page_number(section.footer.paragraphs[0])


def add_cover(doc: Document) -> None:
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(62)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("OpenMP Paradigms in Python")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(36)
    run = subtitle.add_run("Parallel Programming Laboratory Report for Labs 1 through 5")
    run.font.name = "Aptos Display"
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor.from_string(NAVY)

    meta = add_table(
        doc,
        ["Field", "Details"],
        [
            ["Student", "________________________________"],
            ["Student ID", "________________________________"],
            ["Course", "High Performance and Parallel Computing"],
            ["Instructor", "Sufyan bin Uzayr"],
            ["Language", "Python 3"],
            ["Scope", "Labs 1 to 5; bonus Lab 6 omitted"],
            ["Date", "24 September 2026"],
        ],
        widths=[1.45, 4.85],
    )
    for row in meta.rows[1:]:
        row.cells[0].paragraphs[0].runs[0].bold = True

    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(30)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Measured on Apple M2 hardware using the reproducible quick benchmark profile")
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor.from_string(MID_GRAY)
    doc.add_page_break()


def build_report() -> None:
    system = json.loads((DATA / "system_info.json").read_text(encoding="utf-8"))
    lab1 = read_csv("lab1_oversubscription.csv")
    lab1_cpu = read_csv("lab1_cpu_saturation.csv")
    lab2_race = read_csv("lab2_race.csv")
    lab2_critical = read_csv("lab2_critical.csv")
    lab2_reduction = read_csv("lab2_reduction.csv")
    lab3 = read_csv("lab3_scheduling.csv")
    lab4 = read_csv("lab4_false_sharing.csv")
    lab5 = read_csv("lab5_cutoff_sweep.csv")
    lab5_span = read_csv("lab5_work_span.csv")[0]

    doc = Document()
    configure_styles(doc)
    add_cover(doc)

    doc.add_heading("Report overview", level=1)
    doc.add_paragraph(
        "This report documents five Python experiments that reproduce the main OpenMP paradigms without native pragma directives. "
        "The implementations use native threads for fork-join team identification, worker processes for CPU-bound parallelism and shared-memory races, "
        "Numba for the Mandelbrot compute kernel, and NumPy kernels for task-based sorting. The measured results show that correctness and granularity matter more than simply adding workers: "
        "the naive shared accumulator loses most updates, fine-grained locks serialize the integration, small dynamic chunks amplify scheduling overhead, cache-line isolation reduces contention, "
        "and tiny merge-sort cutoffs create excessive tasks."
    )
    doc.add_paragraph(
        "All tables and figures are generated from raw CSV files in the submission. The benchmark run uses a laptop-scale quick profile so the complete matrix can be reproduced in minutes. "
        "The source code also provides a full profile with the manual's large input sizes. Each table states the actual measured problem size, and no result is presented as a full-scale measurement when it is not."
    )

    doc.add_heading("Contents", level=1)
    for item in (
        "1 System and hardware specifications",
        "2 Experimental methodology",
        "3 Lab 1 fork join teams",
        "4 Lab 2 numerical integration and reductions",
        "5 Lab 3 work sharing and scheduling",
        "6 Lab 4 cache coherency and false sharing",
        "7 Lab 5 recursive task parallelism",
        "8 Conclusions",
        "9 Reproduction guide and references",
    ):
        paragraph = doc.add_paragraph(style="List Number")
        paragraph.add_run(item.split(" ", 1)[1])
    doc.add_page_break()

    doc.add_heading("1 System and hardware specifications", level=1)
    cache_l3 = system["l3_bytes"]
    cache_l3_text = "not exposed by sysctl" if cache_l3 in ("", "unavailable", None) else f"{int(cache_l3) / (1024 * 1024):.1f} MiB"
    add_table(
        doc,
        ["Property", "Observed value"],
        [
            ["CPU model", str(system["cpu_model"])],
            ["Architecture", str(system["machine"])],
            ["Physical cores", str(system["physical_cores"])],
            ["Logical hardware threads", str(system["logical_threads"])],
            ["L1 data cache", f"{int(system['l1_data_bytes']) / 1024:.0f} KiB"],
            ["L2 cache", f"{int(system['l2_bytes']) / (1024 * 1024):.0f} MiB"],
            ["L3 cache", cache_l3_text],
            ["Reported cache line", f"{system['cache_line_bytes']} bytes"],
            ["Operating system", str(system["operating_system"])],
            ["Python runtime", str(system["python"])],
        ],
        widths=[2.15, 4.15],
    )
    doc.add_paragraph(
        "The Apple M2 reports eight physical and eight logical cores, so it does not expose simultaneous multithreading. It reports a 128-byte cache line through macOS sysctl. "
        "For that reason, Lab 4 uses a stride of 16 signed 64-bit counters rather than the generic stride of 8 used for a 64-byte line."
    )

    doc.add_heading("2 Experimental methodology", level=1)
    doc.add_paragraph(
        "I used time.perf_counter() as the wall-clock timer. Every configuration records independent trials in CSV, and plots compute their values from those files. "
        "Multiprocessing uses the spawn start method so behavior is portable between macOS, Windows, and Linux. The random merge-sort input uses the fixed seed 20260924. "
        "The Mandelbrot kernel is warmed once before measurements so the parent process does not include its first Numba compilation in the table."
    )
    add_table(
        doc,
        ["Lab", "Quick profile measured", "Full profile setting"],
        [
            ["1", "3 team trials; 250,000 square roots per worker", "7 trials; 10,000,000 square roots per worker"],
            ["2", "250,000 race steps; 100,000 locked steps; 2,000,000 reduction steps", "100,000,000 race and reduction steps; 1,000,000 locked steps"],
            ["3", "400 x 240 pixels; maximum 300 iterations; 3 trials", "1920 x 1080 pixels; maximum 1,000 iterations; 3 trials"],
            ["4", "200,000 increments per worker; 3 trials", "100,000,000 increments per worker; 5 trials"],
            ["5", "20,000 integers; 3 trials per cutoff", "5,000,000 integers; explicit safety cap for tiny cutoffs"],
        ],
        widths=[0.55, 2.9, 2.9],
        font_size=8,
        vertical_margin=45,
    )
    doc.add_paragraph(
        "The quick profile is intentionally smaller because process creation and millions of Python-level memory updates would otherwise dominate a laptop run. "
        "That choice affects the absolute times and can move the optimum cutoff, but it does not change the correctness mechanisms being demonstrated."
    )
    doc.add_heading("3 Lab 1 fork join teams", level=1)
    doc.add_heading("3.1 Implementation and results", level=2)
    doc.add_paragraph(
        "The team-identification program creates a ThreadPoolExecutor, assigns a logical rank to each task, waits at a barrier, and prints results in completion order. "
        "I executed it ten times and saved every output line in data/lab1_nondeterminism.txt. The order changed across runs even though the source and team size did not."
    )
    lab1_means = group_mean(lab1, ("threads",), "seconds")
    add_table(
        doc,
        ["Threads P", "Mean fork join time ms"],
        [[str(p), f"{lab1_means[(str(p),)] * 1000:.3f}"] for p in (1, 2, 4, 8, 16, 32, 64)],
        widths=[2.0, 3.1],
    )
    add_figure(doc, "lab1_oversubscription.png", "Figure 1. Team creation and join overhead rises with oversubscription.")
    doc.add_page_break()
    add_table(
        doc,
        ["Workers", "Work per worker", "Elapsed s", "Sampled CPU busy percent"],
        [[r["workers"], f"{int(r['work_items_per_worker']):,}", r["seconds"], r["sampled_cpu_busy_percent"]] for r in lab1_cpu],
        widths=[1.0, 1.8, 1.35, 2.15],
    )
    doc.add_paragraph(
        "The empty team cost grew from approximately "
        f"{lab1_means[('1',)] * 1000:.3f} ms at P = 1 to {lab1_means[('64',)] * 1000:.3f} ms at P = 64. "
        "The CPU workload used processes rather than Python threads so the global interpreter lock did not prevent multi-core execution. The sampled CPU figure is a short system-wide observation and is therefore descriptive rather than a controlled utilization trace."
    )
    doc.add_heading("3.2 Analytical responses", level=2)
    add_question(doc, "Question 1.1", "Scheduling", "Print order is determined by the operating-system scheduler, not by logical rank. Runnable kernel threads compete on per-core run queues; interrupts, current core load, wake-up timing, cache residency, and frequency changes alter which thread reaches the print operation first. Out-of-order CPU execution affects instruction timing inside a core, while the OS scheduler decides which software thread runs on which core and for how long.")
    add_question(doc, "Question 1.2", "Oversubscription", "When P exceeds the available hardware contexts, extra threads wait and the scheduler context-switches among them. Each switch preserves architectural state such as registers, the program counter, stack pointer, and scheduling metadata. The new thread can displace useful cache and translation lookaside buffer entries, so the old thread often resumes with a cold working set. On an SMT processor, two hardware threads also share execution units and caches; on this M2 there are no extra SMT contexts, so P greater than 8 is direct oversubscription.")
    add_question(doc, "Question 1.3", "Barriers", "The implicit barrier guarantees that every worker has finished the parallel region before serial execution continues. It also provides the synchronization point needed for completed writes to become visible according to the memory model. Without it, later code could read partially initialized arrays, reuse buffers still being modified, free data that a worker still references, or begin a dependent phase before its inputs exist.")
    add_question(doc, "Question 1.4", "Architectural mapping", "A hardware execution thread is architectural state presented by a physical core, and SMT lets multiple such states share one core pipeline. A kernel thread is an OS-schedulable object with registers, stack, priority, and accounting state that can be mapped onto any hardware thread. A green or virtual thread is scheduled by a language runtime onto a smaller set of carrier kernel threads; it is cheaper to create and block, but it is not itself an independent hardware context.")
    doc.add_page_break()
    doc.add_heading("4 Lab 2 numerical integration and reductions", level=1)
    doc.add_heading("4.1 Three synchronization variants", level=2)
    doc.add_paragraph(
        "The midpoint-rule implementation evaluates 4 / (1 + x squared) on [0, 1]. Variant A performs unsynchronized read-modify-write updates to a shared double. "
        "Variant B protects every update with a process-shared lock. Variant C calculates one private subtotal per worker and combines those values with math.fsum after the workers finish."
    )
    race_group: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in lab2_race:
        race_group[int(row["threads"])].append(row)
    race_table = []
    for p in (1, 2, 4, 8):
        records = race_group[p]
        race_table.append(
            [
                str(p),
                f"{statistics.fmean(float(r['pi']) for r in records):.9f}",
                f"{statistics.fmean(float(r['absolute_error']) for r in records):.3e}",
                f"{statistics.fmean(float(r['seconds']) for r in records):.4f}",
            ]
        )
    add_table(doc, ["Workers", "Mean calculated Pi", "Mean absolute error", "Mean time s"], race_table, widths=[0.9, 1.8, 1.8, 1.45])
    doc.add_paragraph(
        "With one worker, the only error is floating-point rounding. With multiple workers, the error jumps by many orders of magnitude and the final value varies between trials. "
        "This is direct evidence of lost updates, because each worker reads and overwrites the same memory location without coordination."
    )

    add_table(
        doc,
        ["Workers", "Locked time s", "Serial time s", "Lock overhead percent"],
        [[r["threads"], r["seconds"], r["serial_seconds"], r["lock_overhead_percent"]] for r in lab2_critical],
        widths=[0.9, 1.55, 1.55, 2.0],
    )
    doc.add_paragraph(
        "The critical-section variant is numerically correct, but locking each term is extremely expensive. Even at P = 1, synchronization and process startup produce more than 3,000 percent overhead relative to the serial loop; at larger P, workers contend on the same lock and the overhead exceeds 9,000 percent."
    )

    reduction_summary = {}
    for row in lab2_reduction:
        reduction_summary[int(row["threads"])] = row
    add_table(
        doc,
        ["Workers", "Mean time s", "Speedup", "Efficiency", "Absolute error"],
        [
            [str(p), reduction_summary[p]["mean_seconds"], f"{float(reduction_summary[p]['speedup']):.3f}", f"{float(reduction_summary[p]['efficiency']):.3f}", f"{float(reduction_summary[p]['absolute_error']):.2e}"]
            for p in (1, 2, 4, 8, 16)
        ],
        widths=[0.8, 1.25, 1.15, 1.15, 1.65],
    )
    add_figure(doc, "lab2_speedup.png", "Figure 2. Measured reduction speedup compared with ideal linear scaling.")
    doc.add_paragraph(
        f"The best mean result occurred at P = 4 with a speedup of {float(reduction_summary[4]['speedup']):.2f}x. "
        "At P = 8 and P = 16, process startup and interprocess coordination exceeded the useful work saved by parallelism. The reduction remained accurate because each process used a private accumulator and only five subtotal values were combined."
    )
    doc.add_heading("4.2 Analytical responses", level=2)
    add_question(doc, "Question 2.1", "Memory incoherency", "A source-level update expands conceptually into a load from memory or cache into a register, an arithmetic add in that register, and a store back to memory. Two workers can both load the same old value, calculate different new values, and then store in either order. The later store overwrites the earlier one, so one contribution disappears. Cache coherence can keep the bytes coherent without making the three-instruction sequence atomic.")
    add_question(doc, "Question 2.2", "Reduction trees", "A centralized critical section forces P workers through one serialization point and repeatedly transfers ownership of the lock and accumulator cache line. A reduction lets each worker accumulate locally, then combines partial values in pairwise levels. The combination depth grows as log2(P), exposes parallel work at every level, and avoids a write to shared state for every integration step.")
    add_question(doc, "Question 2.3", "Amdahl's Law", "If the serial fraction is 0.05, the infinite-processor limit is 1 / 0.05 = 20x. Real speedup normally flattens earlier because of worker startup, synchronization, task scheduling, load imbalance, memory bandwidth, cache capacity, coherence traffic, and the finite amount of work per processor. The quick profile demonstrates this overhead-dominated regime.")
    add_question(doc, "Question 2.4", "Hardware primitives", "Modern x86 processors usually implement a locked atomic update by obtaining exclusive ownership of the target cache line through the coherence protocol; they do not lock the external memory bus for an ordinary cacheable line. Competing cores observe invalidations and must wait for ownership. ARM-style load-exclusive and store-exclusive instructions use an exclusive monitor: the store succeeds only if no conflicting write has invalidated the reservation, otherwise software retries. Both mechanisms turn the read-modify-write into an indivisible memory-ordering event.")
    doc.add_heading("5 Lab 3 work sharing and scheduling", level=1)
    doc.add_heading("5.1 Scheduler construction and matrix", level=2)
    doc.add_paragraph(
        "The static scheduler assigns row chunks round-robin before workers start. The dynamic scheduler places row chunks in a shared queue and lets each worker claim another chunk after finishing its current one. "
        "Each worker reports the number of Mandelbrot iterations it executed, not merely its row count. The imbalance metric is (maximum work - minimum work) / average work."
    )
    add_figure(doc, "lab3_scheduler_heatmap.png", "Figure 3. Mean Mandelbrot time for the complete static and dynamic 4 by 4 configuration matrices.", width=6.55)
    lab3_means = group_mean(lab3, ("policy", "threads", "chunk_size"), "seconds")
    lab3_imb = group_mean(lab3, ("policy", "threads", "chunk_size"), "work_imbalance")
    best_rows = []
    for policy in ("static", "dynamic"):
        for p in (2, 4, 8, 16):
            best_chunk = min((1, 16, 64, 256), key=lambda c: lab3_means[(policy, str(p), str(c))])
            best_rows.append(
                [
                    policy.title(),
                    str(p),
                    str(best_chunk),
                    f"{lab3_means[(policy, str(p), str(best_chunk))]:.3f}",
                    f"{lab3_imb[(policy, str(p), str(best_chunk))]:.3f}",
                ]
            )
    add_table(doc, ["Policy", "Workers", "Best chunk", "Mean time s", "Work imbalance"], best_rows, widths=[1.3, 0.85, 1.1, 1.25, 1.45])
    doc.add_paragraph(
        "On this small image, process startup dominates at high worker counts, so P = 2 is consistently faster than P = 8 or P = 16. Chunk size 256 creates only one task for a 240-row image; its short time is therefore not evidence of parallel speedup, and its imbalance equals the number of workers because all remaining workers are idle. "
        "The invariant checksum 63,511,591,926 was obtained in every cell, which verifies that scheduling changes did not change the computed image."
    )
    doc.add_heading("5.2 Analytical responses", level=2)
    add_question(doc, "Question 3.1", "Fine-grained contention", "With C = 1, each completed row requires another shared-queue operation. The contested resource is the queue head or atomic work index and the cache line containing it. Ownership of that line bounces among cores, while locks, kernel wakeups, pipe traffic, and task descriptors add overhead. Perfect balance cannot compensate when scheduling a row costs a significant fraction of computing it.")
    add_question(doc, "Question 3.2", "Spatial imbalance", "For contiguous static row blocks, the ranks assigned rows near the vertical center y = 0 usually become stragglers. Those rows cross the main cardioid and bulbs of the Mandelbrot set, where many points remain bounded until MAX_ITER. Top and bottom rows contain more points that escape after a few iterations, so their owners reach the barrier earlier.")
    add_question(doc, "Question 3.3", "Guided policy", "Guided scheduling begins with a chunk roughly proportional to remaining_iterations / P, subject to a minimum chunk size. After workers claim chunks, the remaining count shrinks and the next chunk becomes smaller, approximately exponentially. Large early chunks reduce queue operations, while small late chunks distribute the irregular tail and reduce stragglers.")
    add_question(doc, "Question 3.4", "Engineering decision framework", "Use static scheduling when per-iteration cost is uniform, locality matters, and the iteration count is known. Use dynamic scheduling when costs vary unpredictably and each chunk contains enough work to amortize queue contention. Use guided scheduling when the loop is large and irregular: large early allocations keep overhead low, and shrinking chunks improve balance near completion.")
    doc.add_heading("6 Lab 4 cache coherency and false sharing", level=1)
    doc.add_heading("6.1 Scaling profiles", level=2)
    doc.add_paragraph(
        "The unpadded variant places adjacent signed 64-bit counters in one shared array. The padded variant separates counters by the detected cache-line stride. The local variant increments a process-local scalar and writes the final value once. "
        "Every trial verified that each counter reached the requested iteration count."
    )
    add_figure(doc, "lab4_false_sharing.png", "Figure 4. Unpadded, cache-line-padded, and local-accumulator scaling.")
    lab4_means = group_mean(lab4, ("variant", "threads"), "seconds")
    add_table(
        doc,
        ["Workers", "Unpadded s", "Padded s", "Local accumulator s"],
        [
            [str(p), f"{lab4_means[('unpadded', str(p))]:.4f}", f"{lab4_means[('padded', str(p))]:.4f}", f"{lab4_means[('local', str(p))]:.4f}"]
            for p in (1, 2, 4, 8, 16)
        ],
        widths=[1.0, 1.55, 1.55, 2.0],
    )
    unpadded16 = lab4_means[("unpadded", "16")]
    padded16 = lab4_means[("padded", "16")]
    local16 = lab4_means[("local", "16")]
    doc.add_paragraph(
        f"At P = 16, padding reduced mean time from {unpadded16:.4f} s to {padded16:.4f} s, a {(unpadded16 / padded16):.2f}x improvement. "
        f"The local accumulator was fastest at {local16:.4f} s because it performed only one shared write per worker. macOS does not expose Linux perf stat, so the requested L1 miss counters are recorded as not available rather than estimated."
    )
    doc.add_heading("6.2 MESI transition sketch", level=2)
    add_table(
        doc,
        ["Step", "Operation", "Core 0 line", "Core 1 line", "Coherence effect"],
        [
            ["0", "Initial", "Invalid", "Invalid", "Line is not cached"],
            ["1", "Core 0 reads", "Exclusive", "Invalid", "Core 0 receives the only clean copy"],
            ["2", "Core 0 writes slot A", "Modified", "Invalid", "Core 0 owns dirty line"],
            ["3", "Core 1 writes slot B", "Invalid", "Modified", "Read-for-ownership invalidates Core 0"],
            ["4", "Core 0 writes slot A", "Modified", "Invalid", "Ownership returns to Core 0"],
        ],
        widths=[0.45, 1.45, 1.0, 1.0, 2.4],
    )
    doc.add_paragraph(
        "Slots A and B can be different 64-bit integers within the same physical cache line. MESI therefore tracks them as one unit. Alternating writes transfer ownership of the entire line even though the program variables do not overlap."
    )
    doc.add_heading("6.3 Analytical responses", level=2)
    add_question(doc, "Question 4.1", "MESI state transitions", "A first read can place a line in Exclusive state when no other cache holds it. A write changes it to Modified. When another core requests write ownership, coherence messages invalidate the first copy; a modified owner must supply or write back the current data. The second core then owns the line in Modified state. Alternating writes repeat this transfer for the whole line, including untouched neighboring counters.")
    add_question(doc, "Question 4.2", "Bus bouncing", "Adding writers increases the rate of exclusive-ownership requests. The physical bottleneck is the coherence interconnect and the latency of moving or invalidating a cache line between private caches. Useful arithmetic is only an increment, so ownership traffic and serialization can cost far more than the computation itself.")
    add_question(doc, "Question 4.3", "True and false sharing", "True sharing means multiple threads access the same logical variable and coordination is required for correctness. False sharing means threads access independent variables that occupy the same coherence block; the values are logically independent, but hardware still transfers ownership at cache-line granularity. Padding can fix false sharing, but it cannot make a true shared update race-free.")
    add_question(doc, "Question 4.4", "Adjacent-line prefetching", "The JVM's Contended mechanism commonly separates annotated fields or groups with more than one 64-byte line. A 128-byte gap protects against object alignment variation and adjacent-line prefetchers that may fetch two neighboring lines as a pair. Leading and trailing padding also prevents unrelated fields on either side of the object from landing in the protected line. The trade-off is greater memory footprint, so the internal annotation is restricted unless the relevant VM option is enabled.")
    doc.add_heading("7 Lab 5 recursive task parallelism", level=1)
    doc.add_heading("7.1 Verification and cutoff sweep", level=2)
    doc.add_paragraph(
        "The program recursively partitions the array until each leaf contains at most K values. Leaf tasks sort independent NumPy slices in a bounded ThreadPoolExecutor, after which parent levels merge pairs and wait before proceeding. "
        "NumPy releases the GIL in its compiled sorting kernel, so leaf sorts can overlap. Every measured result passed an ascending-order assertion."
    )
    lab5_means = group_mean(lab5, ("cutoff",), "seconds")
    lab5_tasks = {int(row["cutoff"]): int(row["tasks"]) for row in lab5}
    add_table(
        doc,
        ["Cutoff K", "Leaf tasks", "Mean time s", "Verified"],
        [
            [f"{k:,}", f"{lab5_tasks[k]:,}", f"{lab5_means[(str(k),)]:.4f}", "Yes"]
            for k in (1, 10, 100, 1_000, 10_000, 50_000, 100_000)
        ],
        widths=[1.35, 1.35, 1.45, 1.25],
    )
    add_figure(doc, "lab5_cutoff.png", "Figure 5. Merge-sort task cutoff sweep on a logarithmic K axis.", width=5.7)
    doc.add_paragraph(
        "K = 1 created 20,000 leaf tasks and was over 200 times slower than a single-leaf configuration. The best time for this 20,000-element input occurs when K is at least 50,000, which effectively selects one optimized NumPy sort. "
        "That result should not be generalized to five million elements; the full profile is provided to locate the larger-input trough, with a safety cap that prevents accidental creation of millions of task descriptors."
    )
    doc.add_heading("7.2 Work and span", level=2)
    add_table(
        doc,
        ["Quantity", "Model or measured value"],
        [
            ["Measured one-core optimized sort T1", f"{float(lab5_span['measured_t1_seconds']):.6f} s"],
            ["Work model", f"N log2 N = {float(lab5_span['work_model']):,.0f} units"],
            ["Span model", f"2N - 1 = {float(lab5_span['span_model']):,.0f} units"],
            ["Theoretical parallelism", f"T1 / Tinf = {float(lab5_span['model_parallelism']):.3f}"],
        ],
        widths=[2.85, 3.15],
    )
    doc.add_paragraph(
        "Standard merge sort performs total work T1(N) = 2T1(N/2) + Theta(N) = Theta(N log N). If the two recursive sorts run in parallel but each merge remains sequential, the critical path satisfies Tinf(N) = Tinf(N/2) + Theta(N) = Theta(N). "
        "Using unit coefficients, the geometric merge series is approximately 2N - 1, so the numeric model gives about 7.14-way average parallelism at N = 20,000. Asymptotically, available parallelism is Theta(log N), not unlimited."
    )
    doc.add_heading("7.3 Analytical responses", level=2)
    add_question(doc, "Question 5.1", "Granularity breakdown", "At K = 1, every element becomes a leaf task. Five million elements would require millions of task records, futures, queue entries, references, and temporary slices, plus recursive bookkeeping. Allocation, scheduling, queue contention, and garbage collection can exhaust memory or make progress orders of magnitude slower than sorting. The safety cap makes this failure explicit instead of allowing an accidental resource collapse.")
    add_question(doc, "Question 5.2", "Work stealing", "A worker normally pushes and pops tasks at the bottom of its own double-ended queue using a locality-friendly LIFO order. An idle worker steals from the opposite end of another worker's deque, usually taking an older and therefore larger task. Opposite-end access reduces contention between the owner and thieves, while stealing large subtrees gives the thief enough work to amortize the steal.")
    add_question(doc, "Question 5.3", "Span bottlenecks", "A sequential root merge costs Theta(N) and lies on every completion path, so it dominates the critical path even when both halves sort concurrently. A parallel merge can binary-search a pivot from one sorted half into the other, place the pivot in its final position, and recursively merge the two independent partitions. Prefix sums or merge-path partitioning can also divide the output into disjoint ranges, reducing merge span at the cost of more coordination.")
    add_question(doc, "Question 5.4", "Loop and task paradigms", "Loop work sharing is best for a known canonical iteration range whose iterations are independent and similar enough to partition directly. Tasks represent dynamically discovered units with parent-child dependencies, so they fit recursive divide-and-conquer, graph traversal, branch-and-bound, sparse algorithms, and irregular trees. Tasking is superior when the work graph is not known before execution or when nested parallelism is natural.")
    doc.add_heading("8 Conclusions", level=1)
    doc.add_paragraph(
        "The experiments support five practical conclusions. First, concurrency does not imply deterministic order; barriers establish completion, not ranking. Second, a correct synchronization strategy must match operation granularity: locking every arithmetic update is correct but can be slower than serial execution, while private reduction combines correctness with scalability. "
        "Third, scheduling is a balance between load distribution and dispatch cost. Fourth, memory layout changes performance even when the algorithm and logical variables are unchanged. Fifth, task parallelism needs a cutoff because useful work must dominate task-management overhead."
    )
    doc.add_paragraph(
        "The measured Apple M2 results are primarily overhead studies because the quick profile is intentionally small. That limitation is visible in the data rather than hidden: process startup reduces reduction speedup beyond four workers, dynamic scheduling cannot amortize its queue at high P, and the merge-sort optimum collapses to one optimized leaf. "
        "The supplied full profile preserves the manual's large parameters for a longer run, while the raw CSV schema keeps problem size, trial number, and execution status attached to every observation."
    )

    doc.add_heading("9 Reproduction guide and references", level=1)
    doc.add_heading("9.1 Reproduction commands", level=2)
    for command in (
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "python -m pip install -r requirements.txt",
        "PYTHONPATH=. .venv/bin/python run_all.py --profile quick",
        "PYTHONPATH=. .venv/bin/python run_all.py --profile full",
    ):
        paragraph = doc.add_paragraph(style="No Spacing")
        paragraph.paragraph_format.left_indent = Inches(0.35)
        run = paragraph.add_run(command)
        run.font.name = "Menlo"
        run._element.rPr.rFonts.set(qn("w:ascii"), "Menlo")
        run._element.rPr.rFonts.set(qn("w:hAnsi"), "Menlo")
        run.font.size = Pt(8.5)
    doc.add_paragraph()
    doc.add_page_break()
    doc.add_heading("9.2 Submission file map", level=2)
    add_table(
        doc,
        ["Path", "Purpose"],
        [
            ["lab1 to lab5", "Commented Python source for each required lab"],
            ["data", "Raw console output, CSV timings, and system information"],
            ["plots", "Figures regenerated from the CSV files"],
            ["README.md", "Setup, quick profile, full profile, and platform notes"],
            ["report", "Formal DOCX and PDF report"],
        ],
        widths=[1.7, 4.5],
    )
    doc.add_heading("9.3 References", level=2)
    references = (
        "S. bin Uzayr. Parallel Programming Laboratory Practice Manual: Shared-Memory Concurrency and OpenMP Paradigms. Supplied course manual, 2026.",
        "OpenMP Architecture Review Board. OpenMP Application Programming Interface Specification.",
        "Python Software Foundation. concurrent.futures, multiprocessing, and threading library documentation.",
        "Numba project. Automatic parallelization and prange documentation.",
        "OpenJDK project. Contended annotation and HotSpot field-padding implementation documentation.",
    )
    for number, reference in enumerate(references, start=1):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.25)
        paragraph.paragraph_format.first_line_indent = Inches(-0.25)
        paragraph.add_run(f"{number}. {reference}")

    core = doc.core_properties
    core.title = "OpenMP Paradigms in Python"
    core.subject = "Parallel Programming Laboratory Report for Labs 1 through 5"
    core.author = "Student"
    core.keywords = "OpenMP, Python, parallel programming, fork join, reduction, scheduling, false sharing, merge sort"
    core.comments = "Generated from measured benchmark CSV data."

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
