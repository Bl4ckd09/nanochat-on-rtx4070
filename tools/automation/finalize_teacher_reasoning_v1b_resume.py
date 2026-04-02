#!/usr/bin/env python3
import csv
import math
import re
import sys
import time
from pathlib import Path

notes = Path('/home/sun0115/nanochat-learn/notes')
results_path = notes / 'teacher_reasoning_v1b_2026-04-02_1621_results.tsv'
decision_path = notes / 'teacher_reasoning_v1b_2026-04-02_1621_decision.md'
baseline_summary = notes / 'd24_r32_adamw_partial_fr20_mixv2_2026-03-29_1040_s768_gc_best_s300_confirm_summary_1000_2026-03-29_1200.txt'
seed42_chain = notes / 'sft_teacher_reasoning_v1b_seed42_2026-04-02_1621_chain.log'

seed43_candidates = sorted(notes.glob('sft_teacher_reasoning_v1b_seed43_*_chain.log'))
if not seed43_candidates:
    raise SystemExit('seed43 chain log not found')
seed43_chain = seed43_candidates[-1]

PROMOTE_PASS_COUNT_MIN = 2
PROMOTE_GSM8K_MIN = 4.60
PROMOTE_MMLU_MIN = 27.40

OOM_FAIL_RE = re.compile(r'^\[fail\] all partial full-tune attempts exhausted ', re.M)
FULL_DONE_RE = re.compile(r'^\[done\] completed at ', re.M)
QUICK_FAIL_RE = re.compile(r'^\[gate\] quick gate failed ', re.M)
NON_OOM_RE = re.compile(r'^\[error\] non-OOM failure ', re.M)
SUMMARY_RE = re.compile(r'^\[summary\]\s+(.+)$', re.M)


def extract_summary(path: Path) -> str:
    text = path.read_text() if path.exists() else ''
    matches = SUMMARY_RE.findall(text)
    return matches[-1] if matches else ''


def metric(text: str, label: str) -> float:
    m = re.search(rf"{re.escape(label)}:\s+\d+/\d+\s+=\s+([0-9.]+)%", text)
    return float(m.group(1)) if m else math.nan


def status_for(path: Path) -> str:
    text = path.read_text() if path.exists() else ''
    if FULL_DONE_RE.search(text):
        return 'full_confirm_complete'
    if QUICK_FAIL_RE.search(text):
        return 'quick_gate_failed'
    if NON_OOM_RE.search(text):
        return 'failed_non_oom'
    if OOM_FAIL_RE.search(text):
        return 'failed_attempts_exhausted'
    return 'no_summary'


def rc_for(status: str) -> int:
    if status == 'full_confirm_complete':
        return 0
    if status == 'failed_attempts_exhausted':
        return 2
    return 1


def row_for(seed: int, chain: Path):
    status = status_for(chain)
    summary = extract_summary(chain)
    pass1 = pass8 = mmlu = spelling = 'nan'
    if summary and Path(summary).exists():
        text = Path(summary).read_text()
        vals = [
            metric(text, 'GSM8K pass@1'),
            metric(text, 'GSM8K pass@8'),
            metric(text, 'MMLU'),
            metric(text, 'SpellingBee'),
        ]
        fmt = lambda v: 'nan' if math.isnan(v) else f'{v:.2f}'
        pass1, pass8, mmlu, spelling = [fmt(v) for v in vals]
    return {
        'seed': str(seed),
        'rc': str(rc_for(status)),
        'status': status,
        'run_base': chain.name.replace('sft_', '').replace('_chain.log', ''),
        'chain_log': str(chain),
        'summary': summary,
        'gsm8k_pass1_pct': pass1,
        'gsm8k_pass8_pct': pass8,
        'mmlu_pct': mmlu,
        'spellingbee_pct': spelling,
    }


def run_finished(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    return any([
        FULL_DONE_RE.search(text),
        QUICK_FAIL_RE.search(text),
        NON_OOM_RE.search(text),
        OOM_FAIL_RE.search(text),
    ])


def write_results(rows):
    with results_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seed','rc','status','run_base','chain_log','summary','gsm8k_pass1_pct','gsm8k_pass8_pct','mmlu_pct','spellingbee_pct'], delimiter='\t')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_decision(rows):
    baseline = baseline_summary.read_text() if baseline_summary.exists() else ''
    baseline_gsm = metric(baseline, 'GSM8K pass@8')
    baseline_mmlu = metric(baseline, 'MMLU')
    baseline_spell = metric(baseline, 'SpellingBee')
    full_rows = [r for r in rows if r['status'] == 'full_confirm_complete']
    threshold_rows = []
    for r in full_rows:
        try:
            gsm = float(r['gsm8k_pass8_pct'])
            mmlu = float(r['mmlu_pct'])
        except Exception:
            continue
        if gsm >= PROMOTE_GSM8K_MIN and mmlu >= PROMOTE_MMLU_MIN:
            threshold_rows.append(r)
    if len(threshold_rows) >= PROMOTE_PASS_COUNT_MIN:
        decision = 'promote_recipe'
        reason = f'{len(threshold_rows)} seeds met the replication thresholds (required {PROMOTE_PASS_COUNT_MIN})'
    elif len(full_rows) >= 1:
        decision = 'hold_provisional'
        reason = 'some seeds reached full confirm, but not enough met the promotion thresholds'
    else:
        decision = 'recipe_failed_quick_gate'
        reason = 'no seed produced a full-confirm result that met the thresholds'
    lines = [
        '# teacher_reasoning_v1b_2026-04-02_1621 decision',
        '',
        f'- decision: `{decision}`',
        f'- reason: {reason}',
        f'- seeds_total: `{len(rows)}`',
        f'- full_confirm_count: `{len(full_rows)}`',
        f'- threshold_pass_count: `{len(threshold_rows)}`',
        f'- threshold_required: `{PROMOTE_PASS_COUNT_MIN}`',
        f'- promote_gsm8k_pass8_min: `{PROMOTE_GSM8K_MIN:.2f}%`',
        f'- promote_mmlu_min: `{PROMOTE_MMLU_MIN:.2f}%`',
    ]
    if not math.isnan(baseline_gsm):
        lines.append(f'- baseline_gsm8k_pass8: `{baseline_gsm:.2f}%`')
    if not math.isnan(baseline_mmlu):
        lines.append(f'- baseline_mmlu: `{baseline_mmlu:.2f}%`')
    if not math.isnan(baseline_spell):
        lines.append(f'- baseline_spellingbee: `{baseline_spell:.2f}%`')
    lines += ['', '## Seed Results', '']
    for r in rows:
        lines.append(f"- seed {r['seed']}: status=`{r['status']}` gsm8k_pass8=`{r['gsm8k_pass8_pct']}` mmlu=`{r['mmlu_pct']}` summary=`{r['summary']}`")
    decision_path.write_text('\n'.join(lines) + '\n')
    print(decision)


while not run_finished(seed43_chain):
    time.sleep(30)

rows = [row_for(42, seed42_chain), row_for(43, seed43_chain)]
write_results(rows)
write_decision(rows)
print(results_path)
print(decision_path)
