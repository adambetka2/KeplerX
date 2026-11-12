"""Render the live pending PSF queue for visual inspection, without refitting."""

from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
QUEUE = ROOT / "other/boi-v3-20261025/psf-campaign-20261027"
BLUE, ORANGE, GREY = "#2368ae", "#c57028", "#788596"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bins(x, y, error, lo, hi, number):
    selected = (x >= lo) & (x <= hi)
    xx, yy, ee = x[selected], y[selected], error[selected]
    label = np.minimum(((xx - lo) / (hi - lo) * number).astype(int), number - 1)
    w = 1 / ee**2
    total = np.bincount(label, weights=w, minlength=number)
    used = total > 0
    return dict(
        x=np.bincount(label, weights=w * xx, minlength=number)[used] / total[used],
        y=np.bincount(label, weights=w * yy, minlength=number)[used] / total[used],
        error=1 / np.sqrt(total[used]),
        count=np.bincount(label, minlength=number)[used],
    )


def limits(binned):
    lower = min(float(np.min(b["y"] - b["error"])) for b in binned if len(b["x"]))
    upper = max(float(np.max(b["y"] + b["error"])) for b in binned if len(b["x"]))
    padding = max((upper - lower) * 0.12, 1.0)
    return lower - padding, upper + padding


def axis_style(ax, xlabel, lo, hi):
    ax.set(xlabel=xlabel, ylabel="Relative flux (ppm)", xlim=(lo, hi))
    ax.axhline(0, color="#c2cbd5", lw=0.65, zorder=0)
    ax.axvline(0, color="#c2cbd5", lw=0.65, zorder=0)
    ax.tick_params(labelsize=8, colors="#48586b")
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b3bdca")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def draw(ax, signal, mode="transit", title=None):
    if mode == "orbit":
        xx, bb, lo, hi = signal["phase"], signal["orbit_bins"], -0.5, 0.5
        xlabel = "Orbital phase (transit = 0)"
        span = signal["duration_days"] / signal["period_days"] / 2
    else:
        xx, bb = signal["hours"], signal["transit_bins"]
        lo, hi = -signal["half_window_hours"], signal["half_window_hours"]
        xlabel = "Hours from supplied mid-transit"
        span = signal["duration_days"] * 12
    use = (xx >= lo) & (xx <= hi)
    ax.scatter(xx[use], signal["flux_ppm"][use], s=2, color=GREY, alpha=0.13,
               edgecolors="none", rasterized=True)
    ax.errorbar(bb["x"], bb["y"], yerr=bb["error"], fmt="o", ms=2.8,
                color=BLUE, ecolor=BLUE, elinewidth=0.65, capsize=0, zorder=3)
    ax.axvspan(-span, span, color=BLUE, alpha=0.055, zorder=0)
    ylo, yhi = limits([bb])
    ax.set_ylim(ylo, yhi)
    off = int(np.sum(use & ((signal["flux_ppm"] < ylo) | (signal["flux_ppm"] > yhi))))
    ax.text(0.99, 0.97, f"{int(use.sum()):,} cadences · {off:,} beyond y-range",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color="#69798d")
    axis_style(ax, xlabel, lo, hi)
    if title:
        ax.set_title(title, loc="left", fontsize=10, pad=11, color="#192c42")


def draw_parity(ax, signal):
    lo, hi = -signal["half_window_hours"], signal["half_window_hours"]
    both = []
    for remainder, color, label in [(0, BLUE, "Even events"), (1, ORANGE, "Odd events")]:
        use = signal["event_number"] % 2 == remainder
        b = bins(signal["hours"][use], signal["flux_ppm"][use], signal["error_ppm"][use], lo, hi, 50)
        both.append(b)
        ax.errorbar(b["x"], b["y"], yerr=b["error"], fmt="o", ms=2.6,
                    color=color, elinewidth=0.6, capsize=0, label=label)
    ax.set_ylim(*limits(both))
    axis_style(ax, "Hours from supplied mid-transit", lo, hi)
    ax.set_title("Odd / even events", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)


def serial_bins(b):
    return {k: v.tolist() for k, v in b.items()}


def main():
    (OUT / "plots").mkdir(exist_ok=True)
    q, status = read(QUEUE / "queue.json"), read(QUEUE / "status.json")
    waiting = [e for e in q["entries"] if status["states"][e["boi_id"]]["stage"] == "NATIVE"]
    running = [e for e in q["entries"] if status["states"][e["boi_id"]]["stage"] == "NATIVE_RUNNING"]
    ranks = {e["boi_id"]: e for e in read(ROOT / "results/boi-validation/queued-candidate-screen-20261028/ranking.json")["signals"]}
    generated = datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(timespec="seconds")
    signals, records = [], []
    for index, entry in enumerate(running + waiting, 1):
        boi = entry["boi_id"]
        base = Path(entry["psf_case"]) / "native-v3"
        config = read(base / "candidate.json")
        clock = config["recovered"]
        review = read(base / "input/input-review.json")
        photons = base / config["photometry"]
        assert sha(photons) == review["photons"]["sha256"], boi
        with np.load(photons, allow_pickle=False) as archive:
            assert str(archive["source_id"].item()) == config["source_id"]
            valid = archive["valid"]
            assert valid.dtype == bool
            time, flux, error, quarter = [np.array(archive[k][valid], dtype=float)
                                        for k in ("time_bkjd", "flux", "flux_err", "quarter")]
        assert np.isfinite([time, flux, error, quarter]).all() and np.all(error > 0)
        normalization = {}
        for qr in np.unique(quarter):
            use = quarter == qr
            baseline = float(np.median(flux[use]))
            assert np.isfinite(baseline) and baseline > 0
            flux[use] /= baseline
            error[use] /= baseline
            normalization[str(int(qr))] = baseline
        p, t0, duration = [float(clock[k]) for k in ("period_days", "epoch_bkjd", "duration_days")]
        assert 0 < duration < p
        phase_days = (time - t0 + p / 2) % p - p / 2
        phase, hours = phase_days / p, phase_days * 24
        y, err = (flux - 1) * 1e6, error * 1e6
        half_window = min(2 * duration, p / 2) * 24
        old = ranks.get(boi, {})
        kic = old.get("kic")
        if kic is None:
            primary = fits.getheader(review["original"]["path"])
            kic = primary.get("KEPLERID")
        kic = int(kic) if str(kic).strip().isdecimal() and int(kic) > 0 else None
        label = f"KIC {kic}" if kic else f"Gaia DR3 {config['source_id']}"
        if boi == "BOI-2104817618979591168-1":
            label = "TIC 1716526053"
        if boi == "BOI-2073666564790289664-3":
            label += " · signal 3"
        active = entry in running
        position = None if active else waiting.index(entry) + 1
        stage_label = "Fitting" if active else f"Waiting #{position}"
        s = dict(index=index, boi_id=boi, source_id=config["source_id"], label=label,
                 kic=kic, state="fitting" if active else "waiting", position=position,
                 stage_label=stage_label, period_days=p, epoch_bkjd=t0, duration_days=duration,
                 half_window_hours=half_window, phase=phase, hours=hours, flux_ppm=y,
                 error_ppm=err, event_number=np.floor((time - t0) / p + 0.5).astype(np.int64),
                 transit_bins=bins(hours, y, err, -half_window, half_window, 70),
                 orbit_bins=bins(phase, y, err, -0.5, 0.5, 200),
                 valid_cadences=len(time), quarters=len(normalization))
        signals.append(s)
        for mode in ("transit", "orbit"):
            fig, ax = plt.subplots(figsize=(5.8, 3.3), layout="constrained")
            draw(ax, s, mode)
            fig.savefig(OUT / f"plots/{boi}-{mode}.png", dpi=140, facecolor="white")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.8, 3.3), layout="constrained")
        draw_parity(ax, s)
        fig.savefig(OUT / f"plots/{boi}-parity.png", dpi=140, facecolor="white")
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        draw(axes[0], s, "transit", "Transit close-up")
        draw(axes[1], s, "orbit", "Full orbital phase")
        draw_parity(axes[2], s)
        fig.suptitle(f"{label}  ·  {stage_label}\nP = {p:.8f} d   |   supplied duration = {duration*24:.3f} h   |   {boi}",
                     fontsize=13, color="#192c42")
        fig.subplots_adjust(left=0.055, right=0.99, bottom=0.20, top=0.76, wspace=0.28)
        fig.text(0.5, 0.014, "Delivered PSF_FLAT · quarter median normalization · supplied clock · all accepted cadences · no additional detrending or fitted model\nBlue/orange: inverse-variance bins with formal errors; grey: individual cadences. Pale band: supplied transit duration. Independent y scales.",
                 ha="center", fontsize=8, color="#5d6c7d")
        fig.savefig(OUT / f"plots/{boi}-detail.png", dpi=145, facecolor="white", bbox_inches="tight", pad_inches=0.16)
        plt.close(fig)
        record = {k: s[k] for k in ("index", "boi_id", "source_id", "label", "kic", "state", "position", "stage_label",
                                    "period_days", "epoch_bkjd", "duration_days", "valid_cadences", "quarters", "half_window_hours")}
        record.update(photons=str(photons), photons_sha256=review["photons"]["sha256"],
                      candidate=str(base / "candidate.json"), candidate_sha256=sha(base / "candidate.json"),
                      normalization_quarter_medians=normalization,
                      transit_bins=serial_bins(s["transit_bins"]), orbit_bins=serial_bins(s["orbit_bins"]))
        records.append(record)
        print(f"{index}/32 rendered {label} ({stage_label})", flush=True)

    sheets = []
    for start in range(0, len(signals), 12):
        group = signals[start:start + 12]
        nrows = int(np.ceil(len(group) / 3))
        fig, axes = plt.subplots(nrows, 3, figsize=(17.4, 3.55 * nrows + 1), squeeze=False,
                                 layout="constrained")
        for ax, s in zip(axes.flat, group):
            title = f"{s['index']:02d} · {s['label']} · {s['stage_label']}\nP = {s['period_days']:.6f} d  |  duration = {s['duration_days']*24:.2f} h"
            draw(ax, s, "transit", title)
            if s["kic"] == 8758214:
                ax.set_facecolor("#f5f1ff")
        for ax in list(axes.flat)[len(group):]:
            ax.set_visible(False)
        page = start // 12 + 1
        fig.suptitle(f"Pending PSF signals · {start+1}–{start+len(group)} of {len(signals)} · page {page}\n{len(waiting)} waiting + {len(running)} fitting | supplied clocks | quarter medians | independent y scales",
                     fontsize=15, color="#192c42")
        fig.get_layout_engine().set(rect=(0, 0.025, 1, 0.95))
        fig.text(0.5, 0.009, "Grey: accepted cadences. Blue: inverse-variance bins and formal errors. Pale band: supplied duration. Individual points outside the displayed y-range are counted in each panel.",
                 ha="center", fontsize=8, color="#5d6c7d")
        filename = f"matrix-page-{page}.png"
        fig.savefig(OUT / filename, dpi=155, facecolor="white", bbox_inches="tight", pad_inches=0.16)
        plt.close(fig)
        sheets.append(filename)

    manifest = dict(created_warsaw=generated, queue_status_utc=status["utc"],
                    waiting=len(waiting), fitting=len(running), total=len(signals),
                    queue=str(QUEUE / "queue.json"), queue_sha256=sha(QUEUE / "queue.json"),
                    status_snapshot=status, source="Authenticated staged PSF_FLAT input arrays",
                    method="Each delivered-quality-mask cadence is retained. Flux and error are divided by the median accepted flux of its quarter. Supplied discovery clocks; no fitted clock, additional detrending, clipping, companion masking or reclassification. Equal-width inverse-variance bins; formal errors assume independent delivered errors. All bins are visible. Individual cadences beyond the y-range are counted on each plot. Odd/even labels use the supplied epoch.",
                    matrix_pages=sheets, signals=records)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cards = []
    for s in signals:
        boi, label = s["boi_id"], html.escape(s["label"])
        highlight = " highlight" if s["kic"] == 8758214 else ""
        cards.append(f'''<article class="card{highlight}" data-state="{s['state']}" data-search="{html.escape(s['label']+' '+boi+' '+s['source_id'])}">
          <button class="open" data-id="{boi}" aria-label="Inspect {label}">
            <div class="cardtop"><span class="index">{s['index']:02d}</span><span class="badge {s['state']}">{s['stage_label']}</span></div>
            <h2>{label}</h2><p class="clock">P {s['period_days']:.6f} d <span>·</span> duration {s['duration_days']*24:.2f} h</p>
            <img loading="lazy" src="plots/{boi}-transit.png" alt="Folded PSF light curve for {label}" data-boi="{boi}">
            <div class="cardfoot"><span>{s['valid_cadences']:,} cadences · {s['quarters']} quarters</span><span>Enlarge ↗</span></div>
          </button></article>''')
    public = [{k: r[k] for k in ("boi_id", "label", "state", "stage_label", "period_days", "duration_days", "valid_cadences", "quarters", "source_id")} for r in records]
    document = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Queued transit matrix · SpaceLab</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#182d43;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{background:#152a40;color:#f2f6fa;padding:28px 4vw 24px}.eyebrow{font-size:12px;letter-spacing:.13em;color:#b5cbde;text-transform:uppercase}h1{font-size:29px;letter-spacing:-.025em;font-weight:600;margin:5px 0}header p{margin:6px 0 0;color:#c7d7e5;max-width:1000px}.stamp{font-size:12px;opacity:.8}.controls{position:sticky;top:0;z-index:2;padding:15px 4vw;background:#eef2f6f5;border-bottom:1px solid #d5dfe9;display:flex;flex-wrap:wrap;gap:10px;align-items:center}.controls input{width:280px;max-width:100%;padding:9px 12px;border:1px solid #bbc9d7;border-radius:7px;font:inherit}.controls select,.controls button,.download{padding:9px 12px;border:1px solid #bbc9d7;border-radius:7px;background:white;color:#20394e;font:inherit}.controls button{cursor:pointer}.controls .active{background:#203e60;color:white;border-color:#203e60}#count{margin-left:auto;font-size:13px;color:#50647a}.guide{padding:14px 4vw 2px;color:#50647a;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px;padding:18px 4vw 30px}.card{min-width:0;background:white;border:1px solid #d4dfe9;border-radius:10px;overflow:hidden;box-shadow:0 2px 5px #17345208}.card.highlight{border:2px solid #9272c5}.card button{background:none;border:0;padding:15px 14px 10px;width:100%;text-align:left;color:inherit;cursor:pointer;font:inherit}.card button:hover{background:#f9fbfd}.cardtop{display:flex;align-items:center;justify-content:space-between}.index{font-size:12px;color:#708196}.badge{padding:3px 8px;border-radius:5px;font-size:11px;background:#edf2f8;color:#48657e}.badge.fitting{background:#dceefa;color:#185b84}.card h2{font-size:16px;font-weight:600;margin:7px 0 2px;overflow-wrap:anywhere}.clock{font-size:12px;color:#536b81;margin:0}.clock span{padding:0 5px}.card img{width:100%;display:block;margin-top:7px}.cardfoot{display:flex;justify-content:space-between;gap:8px;color:#6b7f91;font-size:11px}.cardfoot span:last-child{color:#2368ae}.downloads{padding:5px 4vw 35px;font-size:13px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}.downloads a{color:#2368ae}dialog{width:calc(100vw - 32px);max-width:1800px;max-height:95vh;border:1px solid #b3c3d3;border-radius:10px;padding:0;box-shadow:0 15px 80px #061b3c55}dialog::backdrop{background:#10243dc9}.modalhead{position:sticky;top:0;background:#f8fafc;padding:13px 18px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #d3dfeb}.modalhead h2{font-size:17px;margin:0;margin-right:auto}.modalhead button{font:inherit;background:white;border:1px solid #b9c9d8;border-radius:5px;padding:5px 12px;cursor:pointer}.modalbody{padding:12px 18px 20px}.modalbody img{width:100%;display:block}.modalbody p{font-size:13px;color:#526b80;margin:8px 0}.modalbody a{color:#2368ae}.empty{display:none;padding:30px 4vw;color:#536b81}footer{font-size:12px;color:#60778b;padding:0 4vw 30px;max-width:1300px}button:focus-visible,a:focus-visible{outline:3px solid #9174c5;outline-offset:3px}@media(max-width:700px){.grid{grid-template-columns:1fr;padding:14px}.controls{padding:12px 14px}.controls input{width:100%}header{padding:22px 18px}h1{font-size:25px}#count{margin-left:0}.guide{padding:12px 18px 0}.modalhead{flex-wrap:wrap}.modalhead h2{width:100%}}
</style></head><body>
<header><div class="eyebrow">SpaceLab · visual inspection</div><h1>Queued transit matrix</h1>
<p>__WAITING__ waiting signals + __RUNNING__ active fits. Inspect each supplied-clock fold, then enlarge it for the full orbit and odd/even comparison.</p>
<p class="stamp">Snapshot: __DATE__ Warsaw · KIC 8758214 is outlined in purple, waiting #__POSITION__.</p></header>
<div class="controls"><input id="search" type="search" placeholder="Find KIC, TIC, Gaia or BOI…" aria-label="Find a signal">
<select id="state" aria-label="Queue state"><option value="all">All pending signals</option><option value="waiting">Waiting only</option><option value="fitting">Fitting now</option></select>
<button id="transit" class="active" aria-pressed="true">Transit close-ups</button><button id="orbit" aria-pressed="false">Full orbital phase</button><span id="count"></span></div>
<p class="guide">Grey: individual accepted cadences. Blue: inverse-variance bins with formal error bars. Pale band: supplied transit duration. <strong>Each panel has its own y scale.</strong> Click any card to inspect it.</p>
<main class="grid">__CARDS__</main><p class="empty" id="empty">No signals match this search.</p>
<div class="downloads"><strong>Save the matrices:</strong>__DOWNLOADS__<a href="manifest.json">Data and methods (JSON)</a></div>
<footer>Delivered PSF_FLAT flux, divided by each quarter’s median. No additional detrending, refitting, clipping or scientific classification. Every accepted cadence contributes to the bins; points beyond the displayed y-range are counted in the plots. Formal bin errors assume independent delivered errors. Odd/even numbering is relative to the supplied epoch. Other detected signals are not masked. These figures show the data used for visual inspection, not the eventual fitted model. Queue status is a dated snapshot.</footer>
<dialog id="detail"><div class="modalhead"><h2 id="detail-title"></h2><button id="prev" aria-label="Previous signal">← Previous</button><button id="next" aria-label="Next signal">Next →</button><button id="close" aria-label="Close inspection">Close ×</button></div><div class="modalbody"><img id="detail-img" alt="Detailed signal inspection"><p id="detail-meta"></p><p><a id="detail-file" target="_blank">Open full-resolution figure ↗</a></p></div></dialog>
<script>
const signals=__DATA__;const cards=[...document.querySelectorAll('.card')];const search=document.querySelector('#search');const state=document.querySelector('#state');const dialog=document.querySelector('#detail');let selected=0;
function filter(){let n=0;const query=search.value.trim().toLowerCase();cards.forEach(c=>{const show=c.dataset.search.toLowerCase().includes(query)&&(state.value==='all'||c.dataset.state===state.value);c.hidden=!show;if(show)n++});document.querySelector('#count').textContent=n+' of '+signals.length+' signals';document.querySelector('#empty').style.display=n?'none':'block'}
search.addEventListener('input',filter);state.addEventListener('change',filter);
function mode(which){document.querySelectorAll('.card img').forEach(img=>{img.src='plots/'+img.dataset.boi+'-'+which+'.png'});['transit','orbit'].forEach(x=>{document.getElementById(x).classList.toggle('active',x===which);document.getElementById(x).setAttribute('aria-pressed',x===which?'true':'false')})}
document.querySelector('#transit').onclick=()=>mode('transit');document.querySelector('#orbit').onclick=()=>mode('orbit');
function inspect(i){selected=(i+signals.length)%signals.length;const s=signals[selected];document.querySelector('#detail-title').textContent=(selected+1)+' / '+signals.length+' · '+s.label+' · '+s.stage_label;const src='plots/'+s.boi_id+'-detail.png';document.querySelector('#detail-img').src=src;document.querySelector('#detail-file').href=src;document.querySelector('#detail-meta').textContent=s.boi_id+' · Gaia DR3 '+s.source_id+' · '+s.valid_cadences.toLocaleString()+' accepted cadences in '+s.quarters+' quarters';if(!dialog.open)dialog.showModal()}
document.querySelectorAll('.open').forEach(b=>b.onclick=()=>inspect(signals.findIndex(s=>s.boi_id===b.dataset.id)));document.querySelector('#close').onclick=()=>dialog.close();document.querySelector('#prev').onclick=()=>inspect(selected-1);document.querySelector('#next').onclick=()=>inspect(selected+1);document.addEventListener('keydown',e=>{if(dialog.open&&e.key==='ArrowRight')inspect(selected+1);if(dialog.open&&e.key==='ArrowLeft')inspect(selected-1)});dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close()});filter();
</script></body></html>'''
    position875 = next(s["position"] for s in signals if s["kic"] == 8758214)
    replacements = {"__WAITING__": str(len(waiting)), "__RUNNING__": str(len(running)),
                    "__DATE__": html.escape(generated[:19].replace("T", " ")),
                    "__POSITION__": str(position875), "__CARDS__": "\n".join(cards),
                    "__DOWNLOADS__": " ".join(f'<a href="{p}" target="_blank">Page {i+1} (PNG)</a>' for i, p in enumerate(sheets)),
                    "__DATA__": json.dumps(public).replace("</", "<\\/")}
    for key, value in replacements.items():
        document = document.replace(key, value)
    # Keep individual scientific panels readable in the desktop side panel.
    document = document.replace('minmax(330px,1fr)', 'minmax(260px,1fr)')
    document = document.replace('.grid{grid-template-columns:1fr;padding:14px}', '.grid{padding:14px}')
    document = document.replace('</style>', '.detailfigs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.detailfigs h3{font-size:14px;margin:6px 0}.detailfigs img{width:100%}@media(max-width:1000px){.detailfigs{grid-template-columns:1fr}}@media(max-width:540px){.grid{grid-template-columns:1fr}}</style>')
    document = document.replace('<img id="detail-img" alt="Detailed signal inspection">', '<div class="detailfigs"><section><h3>Transit close-up</h3><img id="detail-img" alt="Transit close-up"></section><section><h3>Full orbital phase</h3><img id="detail-orbit" alt="Full orbital phase"></section><section><h3>Odd / even comparison</h3><img id="detail-parity" alt="Odd and even events"></section></div>')
    document = document.replace("document.querySelector('#detail-img').src=src;", "document.querySelector('#detail-img').src='plots/'+s.boi_id+'-transit.png';document.querySelector('#detail-orbit').src='plots/'+s.boi_id+'-orbit.png';document.querySelector('#detail-parity').src='plots/'+s.boi_id+'-parity.png';")
    (OUT / "index.html").write_text(document)
    print(json.dumps(dict(total=len(signals), waiting=len(waiting), fitting=len(running), output=str(OUT)), indent=2))


if __name__ == "__main__":
    main()
