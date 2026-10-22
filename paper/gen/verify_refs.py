# -*- coding: utf-8 -*-
"""Verify every bibliography entry against arXiv and Crossref."""

import json
import re
import xml.etree.ElementTree as ET

import time

import requests

PROXIES = {"http": "http://127.0.0.1:18081", "https": "http://127.0.0.1:18081"}

ARXIV = {
    "2402.03300": "shao2024deepseekmath",
    "2412.15115": "qwen2025qwen25",
    "2110.14168": "cobbe2021gsm8k",
    "2307.10635": "wang2023scibench",
    "2311.12022": "rein2023gpqa",
    "2210.03629": "yao2023react",
    "2302.04761": "schick2023toolformer",
    "2211.09110": "liang2022helm",
    "2206.04615": "srivastava2022bigbench",
    "2305.20050": "lightman2023verify",
    "2411.15124": "lambert2024tulu3",
    "2409.12122": "yang2024qwenmath",
    "2410.23326": "bushuiev2024massspecgym",
}

NS = {"a": "http://www.w3.org/2005/Atom"}


def arxiv_check(ids):
    import time
    out = {}
    for aid in ids:
        url = "http://export.arxiv.org/api/query?id_list=" + aid + "&max_results=1"
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "ref-check/1.0"})
        except Exception as e:
            out[aid] = ("TIMEOUT", [], "")
            continue
        if r.status_code != 200 or not r.text.startswith("<?xml"):
            out[aid] = ("ERROR " + str(r.status_code), [], "")
            time.sleep(1)
            continue
        root = ET.fromstring(r.text)
        for entry in root.findall("a:entry", NS):
            found = re.sub(r"v\d+$", "", entry.find("a:id", NS).text.split("/abs/")[-1])
            title = re.sub(r"\s+", " ", entry.find("a:title", NS).text).strip()
            authors = [a.find("a:name", NS).text for a in entry.findall("a:author", NS)]
            published = entry.find("a:published", NS).text[:10]
            out[aid] = (title, authors, published)
        time.sleep(1)
    return out


def crossref_check(title):
    url = "https://api.crossref.org/works"
    r = requests.get(url, params={"query.bibliographic": title, "rows": 1}, timeout=60)
    item = r.json()["message"]["items"][0]
    return (
        item.get("title", [""])[0],
        item.get("container-title", [""])[0],
        item.get("issued", {}).get("date-parts", [[None]])[0][0],
        item.get("volume", ""),
        item.get("page", ""),
        [f"{a.get('given','')} {a.get('family','')}".strip() for a in item.get("author", [])][:5],
    )


def main():
    print("=== arXiv entries ===")
    res = arxiv_check(list(ARXIV))
    for aid, key in ARXIV.items():
        if aid in res:
            title, authors, pub = res[aid]
            print(f"[{key}] {aid} | {pub} | {title[:80]}")
            print(f"    authors: {', '.join(authors[:4])}{' ...' if len(authors) > 4 else ''} ({len(authors)})")
        else:
            print(f"[{key}] {aid} NOT FOUND")

    print()
    print("=== journal entries ===")
    for key, title in [
        ("hinnov2013cyclo", "Cyclostratigraphy and its revolutionizing applications in the earth and planetary sciences"),
        ("hinnov2007ats", "Cyclostratigraphy and the astronomical time scale"),
        ("li2019acycle", "Acycle time-series analysis software for paleoclimate research and education"),
    ]:
        try:
            t, journal, year, vol, pages, authors = crossref_check(title)
            print(f"[{key}] {year} | {journal} | vol {vol} | pp {pages}")
            print(f"    title: {t[:90]}")
            print(f"    authors: {', '.join(authors)}")
        except Exception as e:
            print(f"[{key}] ERROR {repr(e)[:100]}")


if __name__ == "__main__":
    main()
