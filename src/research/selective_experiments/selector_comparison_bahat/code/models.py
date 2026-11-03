from pathlib import Path
import math
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .common import read, write, sha, load_data, check_receipt
from .features import representation, baseline, logistic_score, weights, sorted_logits
from .metrics import curve, threshold


def fit_lr(x, y, w, C=.1, scaler_weights=None):
    scaler = StandardScaler().fit(x, sample_weight=w if scaler_weights is None else scaler_weights)
    estimator = LogisticRegression(C=C, solver="lbfgs", max_iter=5000, tol=1e-9)
    estimator.fit(scaler.transform(x), y, sample_weight=w)
    if int(estimator.n_iter_.max()) >= 5000:
        raise ValueError("Logistická regresia nekonvergovala.")
    return {"scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
            "coef": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
            "iterations": int(estimator.n_iter_.max())}


def network(dimension):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(dimension,70), torch.nn.ReLU(), torch.nn.BatchNorm1d(70),
                               torch.nn.Dropout(.5), torch.nn.Linear(70,70), torch.nn.ReLU(),
                               torch.nn.BatchNorm1d(70), torch.nn.Dropout(.5), torch.nn.Linear(70,1))


def fit_mlp(x, y, w, vx, vy, seed, settings):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    scaler = StandardScaler().fit(x, sample_weight=w)
    tx = torch.tensor(scaler.transform(x), dtype=torch.float32)
    tv = torch.tensor(scaler.transform(vx), dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    totals = np.bincount(y, weights=w, minlength=2)
    if np.any(totals <= 0):
        raise ValueError("Učenie potrebuje správne aj chybné rozhodnutia.")
    balanced = w / totals[y]
    balanced /= balanced.mean()
    tw = torch.tensor(balanced, dtype=torch.float32)
    net = network(x.shape[1]); optimizer = torch.optim.Adam(net.parameters(), lr=settings["lr"])
    best, history, state, stale = math.inf, [], None, 0
    batch_size = settings["batch_size"]
    for epoch in range(1, settings["epochs"]+1):
        net.train(); loss_sum = 0.
        for idx in torch.tensor_split(torch.randperm(len(y)), max(1, math.ceil(len(y)/batch_size))):
            if len(idx) < 2:
                raise ValueError("Tréningový minibatch musí mať aspoň dva záznamy.")
            optimizer.zero_grad(set_to_none=True)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(tx[idx]).ravel(), ty[idx], reduction="none") * tw[idx]).mean()
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach())*len(idx)
        net.eval()
        with torch.inference_mode():
            score = torch.sigmoid(net(tv).ravel()).numpy().astype(float)
        area = curve(vy, score)["aurc"]
        history.append({"epoch": epoch, "loss": loss_sum/len(y), "selection_aurc": area})
        if area < best:
            best, stale = area, 0
            state = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
            best_epoch = epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"    epocha {epoch}: výberové AURC {area:.6f}", flush=True)
        if stale >= settings["patience"]:
            break
    return {"scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
            "dimension": x.shape[1], "state": {k: v.tolist() for k,v in state.items()},
            "seed": seed, "best_epoch": best_epoch, "history": history,
            "adaptation": "Bahat 2018: 2x70, ReLU, BatchNorm, dropout .5, balanced loss; own optimizer/scaling/training inputs"}


def mlp_score(payload, x):
    import torch
    torch.set_num_threads(4)
    net = network(payload["dimension"])
    template = net.state_dict()
    net.load_state_dict({k: torch.tensor(v, dtype=template[k].dtype) for k,v in payload["state"].items()})
    net.eval()
    z = (x - payload["scaler"]["mean"]) / np.asarray(payload["scaler"]["scale"])
    with torch.inference_mode():
        return np.concatenate([torch.sigmoid(net(torch.tensor(a, dtype=torch.float32)).ravel()).numpy().astype(float)
                               for a in np.array_split(z, max(1, math.ceil(len(z)/4096)))])


def extra_data(run, split, protocol_sha):
    folder = Path(run) / "logits"
    check_receipt(folder, split + ".complete.json", protocol_sha)
    with np.load(folder / (split + ".npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def matrix(data, name, extra=None):
    if name.endswith("_logits"):
        if extra is None:
            raise ValueError("Najprv dokončite extract_selector_comparison.py.")
        return sorted_logits(extra[name])
    return representation(data["margins"], name)


def score(model, data, extra=None):
    spec = model["spec"]
    if spec["kind"] == "baseline":
        return baseline(data["margins"], spec["id"])
    if spec["kind"] == "frozen":
        return data["frozen_scores"]
    x = matrix(data, spec["features"], extra)
    return (mlp_score if spec["kind"] == "mlp" else logistic_score)(model["parameters"], x)


def train(run, protocol, suite):
    run = Path(run); psha = sha(run / "protocol.json")
    folder = run / "models"; folder.mkdir(exist_ok=True)
    manifest_path = suite + ".complete.json"
    if (folder / manifest_path).exists():
        check_receipt(folder, manifest_path, psha)
        print("Pomocné modely už sú zmrazené: " + suite, flush=True)
        return
    # Táto funkcia neotvára externé testovacie dáta ani kalibračné profily.
    data, meta = load_data(run, "train"); val, _ = load_data(run, "selection")
    y = data["y"]; w = weights([r["group"] for r in meta])
    if len(np.unique(y)) != 2:
        raise ValueError("Tréning má iba jednu triedu chyby.")
    extra = extra_data(run, "train", psha) if suite == "full" else None
    vextra = extra_data(run, "selection", psha) if suite == "full" else None
    completed = {}
    for i, spec in enumerate(protocol["methods"][suite], 1):
        name = spec["id"]; filename = name + ".json"; receipt = name + ".receipt.json"
        print(f"[{i}/{len(protocol['methods'][suite])}] {name}", flush=True)
        if (folder / receipt).exists():
            check_receipt(folder, receipt, psha)
            if read(folder / filename)["spec"] != spec:
                raise ValueError("Zmenená špecifikácia existujúceho modelu.")
        else:
            model = {"spec": spec, "protocol_sha256": psha}
            kind = spec["kind"]
            if kind in ("lr", "mlp"):
                x = matrix(data, spec["features"], extra)
                if kind == "lr":
                    fit_weights = w
                    if spec.get("balance_classes"):
                        totals = np.bincount(y, weights=w, minlength=2)
                        fit_weights = w / totals[y] * w.sum() / 2
                    model["parameters"] = fit_lr(x, y, fit_weights, spec["C"], scaler_weights=w)
                else:
                    model["parameters"] = fit_mlp(x, y, w, matrix(val, spec["features"], vextra), val["y"],
                                                   spec["seed"], protocol["training"])
            vs = score(model, val, vextra)
            if name == "lr_full" and not np.allclose(vs, val["frozen_scores"], rtol=1e-8, atol=1e-8):
                raise ValueError("Kontrolné opätovné učenie nereprodukuje pôvodný model; porovnanie sa zastavilo.")
            model["selection_aurc"] = curve(val["y"], vs)["aurc"]
            model["threshold"] = protocol["frozen_threshold"] if kind == "frozen" else threshold(val["y"], vs)
            model["threshold_status"] = "original_frozen_rule" if kind == "frozen" else "development_only_not_certified"
            write(folder / filename, model)
            write(folder / receipt, {"protocol_sha256": psha, "files": {filename: sha(folder / filename)}})
        completed[filename] = sha(folder / filename)
    write(folder / manifest_path, {"protocol_sha256": psha, "files": completed, "suite": suite,
                                   "external_used_for_fitting": False, "calibration_used": False})
    print("Učenie dokončené. Všetky porovnávané pravidlá sú zmrazené pred vyhodnotením.", flush=True)
