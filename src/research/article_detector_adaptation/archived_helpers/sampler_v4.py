"""Vyvážené vzorkovanie V4 s tvrdým limitom opakovaní nezávislej skupiny.

Hmotnosti sú hierarchické: trieda, rodina/doména, zdroj, generátor,
skupina. Malý zdroj po vyčerpaní skupín ustúpi ostatným zdrojom; limit
opakovaní sa nikdy neuvoľňuje iba kvôli požadovanej dĺžke epochy.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import random


class _Tree:
    def __init__(self, values):
        self.n = len(values)
        self.tree = [0.0] * (self.n + 1)
        self.values = list(values)
        for i, value in enumerate(values):
            self.add(i, value)

    def add(self, i, value):
        i += 1
        while i <= self.n:
            self.tree[i] += value
            i += i & -i

    def total(self):
        i, value = self.n, 0.0
        while i:
            value += self.tree[i]
            i -= i & -i
        return value

    def draw(self, uniform):
        i = 0
        step = 1 << (self.n.bit_length() - 1)
        while step:
            j = i + step
            if j <= self.n and self.tree[j] <= uniform:
                uniform -= self.tree[j]
                i = j
            step >>= 1
        if i >= self.n:
            raise RuntimeError("Numerická chyba vzorkovania.")
        return i


def group_key(row):
    return str(row.get("dependency_key") or row["group"])


def sampling_cell(row):
    label = int(row["label"])
    family = str(row.get("family", ""))
    if label not in (0, 1) or family not in {"real", "gan", "diffusion", "historical_unknown"}:
        raise ValueError("Neplatná trieda alebo rodina vo vzorkovači V4.")
    if (label == 0 and family != "real") or (label == 1 and family == "real"):
        raise ValueError("Rodina nezodpovedá triede.")
    level = str(row.get("domain", "unknown")) if label == 0 else family
    if not row.get("source") or not row.get("generator"):
        raise ValueError("Vzorka potrebuje zdroj aj generátor/pôvod.")
    return (label, level, str(row["source"]), str(row["generator"]))


def sampling_plan(rows):
    rows = list(rows)
    cells = defaultdict(lambda: defaultdict(list))
    labels_by_group = defaultdict(set)
    ids = set()
    for index, row in enumerate(rows):
        if row["sample_id"] in ids:
            raise ValueError("Duplicitné sample_id vo vzorkovači.")
        ids.add(row["sample_id"])
        key, cell = group_key(row), sampling_cell(row)
        if not key:
            raise ValueError("Prázdna závislostná skupina.")
        labels_by_group[key].add(cell[0])
        cells[cell][key].append(index)
    if set().union(*labels_by_group.values()) != {0, 1}:
        raise ValueError("Tréning vyžaduje obe triedy.")
    row_weights = [0.0] * len(rows)
    descriptions = []
    for cell, groups in sorted(cells.items()):
        label, level, source, generator = cell
        n_level = len({c[1] for c in cells if c[0] == label})
        n_source = len({c[2] for c in cells if c[:2] == cell[:2]})
        n_gen = len({c[3] for c in cells if c[:3] == cell[:3]})
        mass = 0.5 / n_level / n_source / n_gen
        for indices in groups.values():
            for index in indices:
                row_weights[index] = mass / len(groups) / len(indices)
        descriptions.append({"label": label, "level": level, "source": source,
                             "generator": generator, "target_mass_before_caps": mass,
                             "groups": len(groups), "images": sum(map(len, groups.values()))})
    if not math.isclose(sum(row_weights), 1.0, abs_tol=1e-9):
        raise RuntimeError("Neplatný súčet hmotností.")
    return row_weights, descriptions


class CappedBalancedSampler:
    """Deterministická epocha; celá závislostná skupina najviac cap krát.

    Počty tried sú presne rovnaké. Ak nie je dostatok skupín, epocha sa
    skráti a skutočný počet sa zaznamená. Hmotnosti zdrojov sú ciele pred
    uplatnením kapacity, nie sľub presného zastúpenia po vyčerpaní zdroja.
    """
    def __init__(self, rows, samples_per_epoch=24000, max_group_repeats=3, seed=0):
        self.rows = list(rows)
        if samples_per_epoch < 2 or max_group_repeats < 1:
            raise ValueError("Epocha potrebuje aspoň dve vzorky a kladný limit.")
        weights, cells = sampling_plan(self.rows)
        by_class = {0: defaultdict(list), 1: defaultdict(list)}
        for index, row in enumerate(self.rows):
            by_class[int(row["label"])][group_key(row)].append(index)
        # A reference group may contain one real and several generated images.
        # Its capacity is shared across both classes, never duplicated by label.
        all_groups = set(by_class[0]) | set(by_class[1])
        shared_groups = set(by_class[0]) & set(by_class[1])
        quota = min(samples_per_epoch // 2, len(all_groups) * max_group_repeats // 2,
                    *(len(groups) * max_group_repeats for groups in by_class.values()))
        if quota < 1:
            raise ValueError("Skupinový limit neumožňuje ani jeden príklad z každej triedy.")
        rng, indices = random.Random(seed), []
        remaining_capacity = {key: max_group_repeats for key in all_groups}
        available = [len(by_class[label]) * max_group_repeats for label in (0, 1)]
        needed = [quota, quota]
        trees, keys_by_tree, lookup = {}, {}, {}
        for label in (0, 1):
            for shared in (False, True):
                keys = sorted(key for key in by_class[label] if (key in shared_groups) == shared)
                tree_key = (label, shared)
                keys_by_tree[tree_key] = keys
                trees[tree_key] = _Tree([sum(weights[index] for index in by_class[label][key]) for key in keys])
                for gi, key in enumerate(keys):
                    lookup[(label, key)] = (tree_key, gi)
        labels = [0] * quota + [1] * quota
        rng.shuffle(labels)
        for label in labels:
            # Reserve shared capacity whenever the other class needs every one
            # of its remaining slots. This maintains a feasible balanced flow.
            allow_shared = available[1 - label] > needed[1 - label]
            exclusive_mass = max(0.0, trees[(label, False)].total())
            shared_mass = max(0.0, trees[(label, True)].total()) if allow_shared else 0.0
            total = exclusive_mass + shared_mass
            if total <= 1e-14:
                raise RuntimeError("Vyčerpaná kapacita triedy pred dokončením kvóty.")
            draw = rng.random() * total
            shared = draw >= exclusive_mass
            tree_key = (label, shared)
            gi = trees[tree_key].draw(draw - exclusive_mass if shared else draw)
            key = keys_by_tree[tree_key][gi]
            candidates = by_class[label][key]
            indices.append(rng.choices(candidates, weights=[weights[i] for i in candidates], k=1)[0])
            remaining_capacity[key] -= 1
            needed[label] -= 1
            for affected_label in (0, 1):
                if key in by_class[affected_label]:
                    available[affected_label] -= 1
                    if remaining_capacity[key] == 0:
                        affected_tree, affected_index = lookup[(affected_label, key)]
                        tree = trees[affected_tree]
                        tree.add(affected_index, -tree.values[affected_index])
                        tree.values[affected_index] = 0.0
        rng.shuffle(indices)
        self.indices = indices
        group_counts = Counter(group_key(self.rows[i]) for i in indices)
        image_counts = Counter(self.rows[i]["sample_id"] for i in indices)
        cell_counts = Counter(sampling_cell(self.rows[i]) for i in indices)
        for cell in cells:
            key = (cell["label"], cell["level"], cell["source"], cell["generator"])
            cell["actual_draws"] = cell_counts[key]
            cell["actual_mass"] = cell_counts[key] / len(indices)
        self.report = {
            "schema_version": "v4-capped-sampler-1", "seed": seed,
            "requested_samples": samples_per_epoch, "actual_samples": len(indices),
            "shortened": len(indices) != samples_per_epoch,
            "max_group_repeats": max_group_repeats,
            "max_observed_group_repeats": max(group_counts.values()),
            "unique_images": len(image_counts), "unique_dependency_groups": len(group_counts),
            "available_images": len(self.rows),
            "available_dependency_groups": len(all_groups),
            "mixed_label_dependency_groups": len(shared_groups),
            "groups_at_repeat_cap": sum(v == max_group_repeats for v in group_counts.values()),
            "group_repeat_histogram": dict(sorted(Counter(group_counts.values()).items())),
            "image_repeat_histogram": dict(sorted(Counter(image_counts.values()).items())),
            "class_draws": {"real": quota, "fake": quota}, "cells": cells,
            "note": "Zdrojové ciele platia pred kapacitným limitom. Po vyčerpaní skupiny sa hmotnosť rozdelí medzi zostávajúce skupiny tej istej triedy. Limit sa zdieľa aj medzi real a fake obrázkami tej istej predlohy; potrebná kapacita druhej triedy sa rezervuje.",
        }

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
