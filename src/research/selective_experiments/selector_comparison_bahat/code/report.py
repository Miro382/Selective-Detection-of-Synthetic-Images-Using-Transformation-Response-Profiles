from collections import Counter
from pathlib import Path
import csv
import html
import numpy as np
from .common import write, read, sha, check_receipt, load_data
from .features import TRANSFORMS
from .models import score, extra_data
from .metrics import curve, summarize, paired_bootstrap


def contrasts(suite):
    pairs = [("frozen_v4","confidence"), ("frozen_v4","tta_mean"), ("lr_full","lr_confidence"),
             ("lr_full","lr_no_summaries"), ("lr_full","lr_no_orientation"),
             ("lr_full","lr_no_confidence"), ("lr_full","lr_summaries_only"),
             ("lr_full","lr_jpeg_only"), ("lr_full","lr_resize_only")]
    pairs += [("lr_full", "lr_without_"+name) for name in TRANSFORMS]
    pairs += [("lr_balanced_no_summaries", "lr_no_summaries")]
    pairs += [(f"mlp_margins_s{s}", "lr_balanced_no_summaries") for s in (17,29,43)]
    pairs += [("frozen_v4", f"mlp_margins_s{s}") for s in (17,29,43)]
    if suite == "full":
        for s in (17,29,43):
            pairs += [(f"mlp_ours_logits_s{s}",f"mlp_bahat_logits_s{s}"),
                      ("frozen_v4",f"mlp_bahat_logits_s{s}"),
                      (f"mlp_ours_logits_s{s}",f"mlp_margins_s{s}")]
    return pairs


def figures(folder, cohort, y, scores, summaries, suite):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10, "axes.spines.top":False,"axes.spines.right":False})
    selected = ["frozen_v4","confidence","tta_mean","lr_no_summaries","mlp_margins_s17"]
    if suite == "full":
        selected += ["mlp_ours_logits_s17","mlp_bahat_logits_s17"]
    fig, ax = plt.subplots(figsize=(8.5,4.6))
    for name in selected:
        c = curve(y, scores[name]); ax.step(c["coverage"]*100,c["risk"]*100,where="pre",label=name,lw=1.7)
    ax.set(xlabel="Pokrytie (%)", ylabel="Chybovosť prijatých rozhodnutí (%)", title=cohort + " – riziko a pokrytie")
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(folder/(cohort+"_risk.svg")); fig.savefig(folder/(cohort+"_risk.png"),dpi=180); plt.close(fig)
    selected = ["lr_full","lr_confidence","lr_no_summaries","lr_no_orientation","lr_no_confidence",
                "lr_summaries_only","lr_jpeg_only","lr_resize_only"]
    fig, ax = plt.subplots(figsize=(8.5,4.6))
    ax.barh(selected,[summaries[n]["aurc"] for n in selected],color=["#2274a5"]+["#73a6c4"]*7)
    ax.invert_yaxis(); ax.set(xlabel="AURC (nižšie je lepšie)",title=cohort+" – kontrolované ablácie, LR C = 0,1")
    ax.grid(axis="x",alpha=.2); fig.tight_layout()
    fig.savefig(folder/(cohort+"_ablation.svg")); fig.savefig(folder/(cohort+"_ablation.png"),dpi=180); plt.close(fig)


def evaluate(run, protocol, suite):
    run = Path(run); psha = sha(run/"protocol.json")
    check_receipt(run/"models",suite+".complete.json",psha)
    folder = run/"reports"/suite; folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():
        check_receipt(folder,"complete.json",psha)
        print("Report už je kompletný: " + str(folder/"index.html"),flush=True)
        return
    data, meta = load_data(run,"external")
    extra = extra_data(run,"external",psha) if suite == "full" else None
    models = {s["id"]:read(run/"models"/(s["id"]+".json")) for s in protocol["methods"][suite]}
    scores = {name:score(model,data,extra) for name,model in models.items()}
    if any(v.shape != data["y"].shape or not np.isfinite(v).all() for v in scores.values()):
        raise ValueError("Neplatné výstupné skóre.")
    np.savez_compressed(folder/"scores.npz",**scores)
    ids = np.array([r["sample_id"] for r in meta]); groups = np.array([r["group"] for r in meta])
    cohorts = np.array([r["cohort"] for r in meta]); pairs = contrasts(suite)
    report = {"protocol_sha256":psha,"suite":suite,"scope":protocol["scope"],"cohorts":{},
              "method_specs":protocol["methods"][suite],"statistics":protocol["statistics"],
              "sources":["https://arxiv.org/abs/1804.00657","https://arxiv.org/abs/2006.16705"]}
    csvrows = []; parts = []
    esc = html.escape
    for cohort in sorted(set(cohorts)):
        mask = cohorts == cohort; y = data["y"][mask]; ss = {k:v[mask] for k,v in scores.items()}
        accepted = int((ss["frozen_v4"] <= protocol["frozen_threshold"]).sum())
        print(f"Vyhodnocujem {cohort}: {len(y)} obrázkov, {len(set(groups[mask]))} skupín; párový bootstrap...",flush=True)
        summary = {k:summarize(y,v,ids[mask],models[k]["threshold"],accepted) for k,v in ss.items()}
        boot = paired_bootstrap(y,ss,ids[mask],groups[mask],pairs,
                                protocol["statistics"]["bootstrap"],protocol["statistics"]["seed"])
        contrast_summary = {}
        for a,b in pairs:
            key = a+"__minus__"+b
            contrast_summary[key] = {"aurc_difference":summary[a]["aurc"]-summary[b]["aurc"],
                                     "risk90_difference":summary[a]["risk_at_coverage"]["0.9"]["risk"]-summary[b]["risk_at_coverage"]["0.9"]["risk"],
                                     "pointwise_group_bootstrap":boot[key]}
        seeds = {}
        families = ["mlp_margins"] + (["mlp_ours_logits","mlp_bahat_logits"] if suite == "full" else [])
        for family in families:
            values = [summary[f"{family}_s{s}"]["aurc"] for s in (17,29,43)]
            seeds[family] = {"mean_aurc":float(np.mean(values)),"std_aurc":float(np.std(values,ddof=1)),"seeds":[17,29,43]}
        sources = {}
        for generator in sorted({r["generator"] for r in meta if r["cohort"] == cohort}):
            subset = mask & np.array([r["generator"] == generator for r in meta])
            sources[generator] = {"n":int(subset.sum()),"detector_errors":int(data["y"][subset].sum()),
                                   "methods":{k:{"aurc":curve(data["y"][subset],v[subset])["aurc"]} for k,v in scores.items()}}
        report["cohorts"][cohort] = {"n":len(y),"groups":len(set(groups[mask])),"detector_errors":int(y.sum()),
                                     "detector_accuracy":float(1-y.mean()),"methods":summary,
                                     "contrasts":contrast_summary,"seed_variability":seeds,"generators":sources}
        figures(folder,cohort,y,ss,summary,suite)
        parts.append(f"<h2>{esc(cohort)}: {len(y):,} obrázkov</h2><p>Detektor urobil {int(y.sum())} chýb. "
                     f"Porovnanie pri pôvodnom pokrytí prijíma pri každej metóde presne {accepted} obrázkov.</p>")
        parts.append("<table><tr><th>Metóda</th><th>AURC ↓</th><th>Chybovosť pri 90 %</th><th>Chyby pri rovnakom pokrytí</th><th>Výberové AURC</th></tr>")
        for name, result in summary.items():
            match = result["matched_frozen_coverage"]; r90=result["risk_at_coverage"]["0.9"]
            csvrows.append({"cohort":cohort,"method":name,"n":len(y),"aurc":result["aurc"],
                            "risk90":r90["risk"],"matched_accepted":match["accepted"],"matched_errors":match["errors"],
                            "matched_risk":match["risk"],"selection_aurc":models[name]["selection_aurc"]})
            parts.append(f"<tr><td>{esc(name)}</td><td>{result['aurc']:.6f}</td><td>{100*r90['risk']:.3f} %</td>"
                         f"<td>{match['errors']} / {match['accepted']}</td><td>{models[name]['selection_aurc']:.6f}</td></tr>")
        parts.append(f'</table><img src="{cohort}_risk.svg"><img src="{cohort}_ablation.svg">')
        parts.append("<h3>Rozdiely AURC a neistota</h3><p>Záporný rozdiel zvýhodňuje prvú metódu. Intervaly sú párové skupinové, bodové 95 %, bez korekcie mnohonásobných porovnaní. Slúžia na prieskumné vyhodnotenie.</p><table><tr><th>Prvá mínus druhá</th><th>Rozdiel</th><th>95 % interval</th></tr>")
        for name,result in contrast_summary.items():
            ci=result["pointwise_group_bootstrap"]["aurc"]
            parts.append(f"<tr><td>{esc(name)}</td><td>{result['aurc_difference']:.6f}</td><td>[{ci['lower_95']:.6f}; {ci['upper_95']:.6f}]</td></tr>")
        parts.append("</table><h3>Variabilita učenia pomocnej siete</h3><p>Uvádzame všetky tri semená a ich priemer; najlepší výsledok testu sa nepoužíva na výber modelu.</p><ul>")
        for name,v in seeds.items():
            parts.append(f"<li>{esc(name)}: priemerné AURC {v['mean_aurc']:.6f}, smerodajná odchýlka {v['std_aurc']:.6f}.</li>")
        parts.append("</ul>")
    write(folder/"results.json",report)
    with (folder/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    intro = """<!doctype html><html lang="sk"><meta charset="utf-8"><title>Porovnanie selektorov</title>
<style>body{font:16px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#192d3e}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{padding:9px;border-bottom:1px solid #d3dde6;text-align:left}th{background:#eaf1f6}img{width:100%;max-width:900px}h2{margin-top:50px}code{background:#eef3f7}</style>
<h1>Porovnanie selektorov a prínosu príznakov</h1>
<p>Detektor aj pôvodný selektor zostali zmrazené. Všetky metódy posudzujú chybu tej istej pôvodnej predikcie; nemenia predikovanú triedu. Nové pomocné modely sa učili na existujúcich tréningových profiloch. Výber epochy a pracovného prahu používal iba pôvodnú výberovú množinu.</p>
<p><strong>Rozsah dôkazu:</strong> ide o retrospektívne porovnanie na už vyhodnotených dátach DeepShield a OpenFake, nie o novú nezávislú certifikáciu. Pôvodné kalibračné profily sa pri novom učení ani teste nepoužili. Pre nové modely nie je overená garancia rizika.</p>
<p><code>tta_mean</code> je adaptácia základného priemerovania pravdepodobnosti pôvodnej triedy podľa Bahata a Shakhnarovicha (2020), s našimi transformáciami a zmrazenou teplotou. Neimplementuje všetky rozšírenia ich článku. <code>mlp_margins</code> používa pomocnú sieť podľa Bahata a kol. (2018), ale naše orientované rozdiely skóre; preto nie je presnou reprodukciou pôvodnej metódy.</p>
<p>Voliteľný režim <code>full</code> porovnáva rovnakú pomocnú sieť na skutočných spoločne zoradených logitoch pre našich päť transformácií a päť transformácií z článku z roku 2018. Aj toto je adaptácia na binárny detektor, s vlastným optimalizátorom, škálovaním a tréningovými vstupmi. Kalibrované rozdiely logitov sa nevydávajú za pôvodné dvojice logitov.</p>
<p>Ablácie používajú rovnakú logistickú regresiu a pevné C = 0,1. Pri vynechaní transformácie sa prepočítajú aj všetky súhrny. <code>lr_balanced_no_summaries</code> navyše vyvažuje triedy chyby rovnako ako pomocná sieť, aby porovnanie siete s regresiou nebolo iba porovnaním váženia tried. <code>no_confidence</code> odstraňuje explicitný príznak istoty; ostatné príznaky môžu stále niesť nepriamu informáciu o istote. Grafy siete ukazujú vopred určené semeno 17; tabuľky obsahujú všetky semená.</p>
<p>Nižšie AURC a nižšia chybovosť pri rovnakom pokrytí sú lepšie. Na testovacích štítkoch sa nevyberá prah ani model. Rovnaké pokrytie je diagnostika poradia skóre, nie nový nasadzovaný prah. Súhrnné skóre nie je dôkaz pôvodu konkrétneho obrázka.</p>
<p>Zdroje: <a href="https://arxiv.org/abs/1804.00657">Bahat, Irani a Shakhnarovich (2018)</a>; <a href="https://arxiv.org/abs/2006.16705">Bahat a Shakhnarovich (2020)</a>. Úplné výsledky: <a href="results.json">JSON</a>, <a href="comparison.csv">CSV</a>.</p>"""
    (folder/"index.html").write_text(intro+"\n".join(parts)+"</html>",encoding="utf-8")
    write(folder/"complete.json",{"protocol_sha256":psha,"files":{f.name:sha(f) for f in folder.iterdir() if f.is_file() and f.name != "complete.json"}})
    print("Hotovo: " + str(folder/"index.html"),flush=True)
