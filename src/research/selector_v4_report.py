"""Slovenský vývojový report selektora V4 s vektorovým grafom bez šrafovania."""
from __future__ import annotations
from html import escape
from pathlib import Path


def pct(value):
    return "nedefinovaná" if value is None else f"{100 * value:.2f} %".replace(".", ",")


def write_report(output, selection, calibration, artifact):
    output = Path(output)
    colors = ("#2563eb", "#ea580c", "#059669", "#9333ea", "#ca8a04", "#0891b2", "#db2777", "#475569", "#dc2626")
    ymax = max(.05, max(p["risk"] for c in selection["candidates"] for p in c["curve"]) * 1.05)
    w, h, left, top, gw, gh = 1060, 640, 75, 35, 630, 500
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial,sans-serif" font-size="14" fill="#1e293b">']
    for i in range(6):
        x, y = left + gw * i / 5, top + gh * (1 - i / 5)
        svg += [f'<path d="M {x} {top} V {top+gh}" stroke="#e2e8f0"/>',
                f'<path d="M {left} {y} H {left+gw}" stroke="#e2e8f0"/>',
                f'<text x="{x}" y="{top+gh+24}" text-anchor="middle">{i*20} %</text>',
                f'<text x="{left-10}" y="{y+5}" text-anchor="end">{100*ymax*i/5:.1f} %</text>']
    for i, c in enumerate(selection["candidates"]):
        points = c["curve"]
        # Downsample display only; tabulated AURC is from all tie-block steps.
        step = max(1, len(points) // 1500)
        sampled = points[::step] + ([points[-1]] if points[-1] != points[::step][-1] else [])
        path = []
        previous = None
        for p in sampled:
            x, y = left + gw*p["coverage"], top + gh*(1-p["risk"]/ymax)
            if previous is None:
                path.append(f"M {left:.3f} {y:.3f} H {x:.3f}")
            else:
                path.append(f"V {y:.3f} H {x:.3f}")
            previous = p
        color = colors[i % len(colors)]
        svg.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="{3 if c["id"] == selection["chosen_id"] else 1.7}"/>')
        yy = top + 25 + i*33
        svg += [f'<path d="M 730 {yy} H 760" stroke="{color}" stroke-width="3"/>',
                f'<text x="770" y="{yy+5}" font-size="12">{escape(c["id"])}</text>']
    svg += [f'<path d="M {left} {top} V {top+gh} H {left+gw}" fill="none" stroke="#334155"/>',
            f'<text x="{left+gw/2}" y="{top+gh+65}" text-anchor="middle">Pokrytie – podiel prijatých rozhodnutí</text>',
            f'<text transform="translate(20 {top+gh/2}) rotate(-90)" text-anchor="middle">Chybovosť prijatých rozhodnutí</text>',
            '<text x="75" y="625" font-size="12">Výberová množina: vývojové porovnanie, nie nezávislý externý test.</text>', '</g></svg>']
    (output / "risk_coverage.svg").write_text("\n".join(svg), encoding="utf-8")
    rows = []
    for c in sorted(selection["candidates"], key=lambda x: (x["selection_aurc"], x["id"])):
        rows.append(f'<tr><td>{escape(c["id"])}{" – vybrané" if c["id"] == selection["chosen_id"] else ""}</td><td>{c["selection_aurc"]:.6f}</td><td>{pct(c["threshold"]["coverage"])}</td><td>{pct(c["threshold"]["risk"])}</td></tr>')
    warnings = "".join(f'<li>{escape(w)}</li>' for w in selection["warnings"])
    status = "Cieľ rizika bol potvrdený v rozsahu kalibračnej distribúcie." if calibration["status"] == "passed" else "Cieľ rizika sa nepotvrdil. Nasadzované pravidlo odovzdáva všetky prípady na overenie."
    text = f'''<!doctype html><html lang="sk"><meta charset="utf-8"><title>Selektor V4 – učenie a overenie</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#172033;background:#f8fafc;max-width:1120px;margin:40px auto;padding:0 24px}}h1,h2{{line-height:1.2}}section{{background:white;border:1px solid #dbe3ee;border-radius:12px;padding:24px;margin:24px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e2e8f0}}img{{width:100%}}code{{overflow-wrap:anywhere}}.status{{border-left:5px solid {'#059669' if calibration['status']=='passed' else '#d97706'};padding:16px;background:#f1f5f9}}</style>
<h1>Selektor V4: výber a jednorazové overenie</h1>
<p>Detektor bol počas učenia selektora zmrazený. Selektor predikuje jeho chybu; nemení rozhodnutie real/fake. Výsledky tejto stránky opisujú vývoj a overenie pevného prahu. Externý test sa týmto skriptom nevykonal.</p>
<section><h2>Čo bolo naučené</h2><p>Tréning obsahoval {selection['training_rows']:,} profilov z {selection['training_groups']:,} skupín; {selection['training_error_groups']:,} skupín obsahovalo aspoň jednu chybu detektora. Každá skupina mala v stratovej funkcii celkovú váhu 1. Výberová množina obsahovala {selection['selection_rows']:,} nezávislých reprezentantov a {selection['selection_detector_errors']:,} chýb detektora.</p>
<p>Vopred stanovené porovnanie obsahovalo tri neučené referencie a tri logistické rodiny, každú pri C = 0,01 a 0,1. Model dostal presné logitové rozdiely, profil degradácií a podľa rodiny predikovanú triedu. Skutočná trieda, zdroj ani generátor neboli vstupnými príznakmi. Skutočný štítok slúžil iba na určenie cieľovej chyby.</p><ul>{warnings}</ul></section>
<section><h2>Vývojový výber</h2><p>Víťaz <strong>{escape(selection['chosen_id'])}</strong> mal najnižšiu výberovú AURC. AURC je súčet pravých krokov krivky rizika a pokrytia po celých blokoch rovnakého skóre. Pri zhode rozhodoval názov kandidáta. Zo všetkých pevných prahov víťaza sa zvolilo najvyššie pokrytie pri empirickej chybovosti najviac 1 % a pokrytí aspoň 10 %. Bez vyhovujúceho bodu je prah −1 a nič sa neprijíma.</p>
<table><thead><tr><th>Kandidát</th><th>AURC ↓</th><th>Pokrytie prahu</th><th>Riziko prahu</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Nasledujúci graf ukazuje vývojové krivky. Víťaz bol zvolený podľa tej istej množiny, preto jeho krivka nie je nezaujatým odhadom externého výkonu.</p><img src="risk_coverage.svg" alt="Vývojové krivky rizika a pokrytia kandidátov selektora V4"></section>
<section><h2>Jednorazové overenie zmrazeného pravidla</h2><p class="status">{status}</p>
<table><tbody><tr><th>Prijaté / všetky</th><td>{calibration['accepted']} / {calibration['total']}</td></tr><tr><th>Pokrytie</th><td>{pct(calibration['coverage'])}</td></tr><tr><th>Chyby medzi prijatými</th><td>{calibration['errors']}</td></tr><tr><th>Empirické riziko</th><td>{pct(calibration['risk'])}</td></tr><tr><th>Jednostranná 95 % horná hranica Clopper–Pearson</th><td>{pct(calibration['risk_upper_bound'])}</td></tr><tr><th>Cieľ rizika</th><td>{pct(calibration['target_risk'])}</td></tr><tr><th>Výskumný prah</th><td>{artifact['research_threshold']:.12g}</td></tr><tr><th>Nasadzovaný prah</th><td>{artifact['accept_threshold']:.12g}</td></tr></tbody></table>
<p>Overil sa iba jeden vopred zmrazený model a jeden prah. Po zistení kalibračných výsledkov sa model, cieľ ani prah neupravovali. Na úspech bolo potrebných aspoň 30 prijatých nezávislých reprezentantov a horná hranica najviac 2 %. Pri prázdnej prijatej množine je empirické riziko nedefinované; nulové pokrytie nie je dôkazom presnosti.</p></section>
<section><h2>Rozsah a obmedzenia</h2><p>Binomické overenie predpokladá nezávislé reprezentanty a rovnakú budúcu cieľovú distribúciu ako pri kalibrácii. Nie je zárukou pri novom generátore, inom zdroji alebo zmene zastúpenia tried. Tento test nevaliduje celú multimodálnu aplikáciu. Artefakt V4 sa automaticky nenasadil a nenahradil V3.</p><p>Presná teplota detektora: {artifact['temperature']:.12g}. Identita modelu: <code>{escape(artifact['model_sha256'])}</code>. Presné vstupy, zdrojový kód, vybraný model a potvrdenia sú uzamknuté kontrolnými súčtami v tomto priečinku.</p></section></html>'''
    (output / "report.html").write_text(text, encoding="utf-8")
