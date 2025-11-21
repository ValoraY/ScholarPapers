import json
from scholarly import scholarly
import traceback
import requests
from bs4 import BeautifulSoup
import html
import os
from datetime import datetime
import time
import random
import textwrap

# -----------------------------------
# Logging Helpers
# -----------------------------------
def info(msg):    print(f"\033[94m[INFO]\033[0m {msg}")
def success(msg): print(f"\033[92m[SUCCESS]\033[0m {msg}")
def warn(msg):    print(f"\033[93m[WARN]\033[0m {msg}")
def error(msg):   print(f"\033[91m[ERROR]\033[0m {msg}")

# ================================================================
# 1. Load Minimal config.json
# ================================================================
info("Loading config.json ...")

with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Required fields (override by GitHub Actions if config_override.json exists)
YEAR_START = cfg.get("year_start", 2020)
YEAR_END = datetime.now().year
AUTHORS = cfg.get("authors", [])
INCREMENTAL_LIMIT = cfg.get("incremental_limit", 20)

# ================================================================
# 2. Load GitHub override (optional)
# ================================================================
if os.path.exists("config_override.json"):
    info("Loading config_override.json ...")
    with open("config_override.json", "r", encoding="utf-8") as f:
        override = json.load(f)

    YEAR_START = override.get("year_start", YEAR_START)
    YEAR_END = override.get("year_end", YEAR_END)
    AUTHORS = override.get("authors", AUTHORS)
    INCREMENTAL_LIMIT = override.get("incremental_limit", INCREMENTAL_LIMIT)

info("Final Config:")
info(f"  YEAR_START = {YEAR_START}")
info(f"  YEAR_END   = {YEAR_END}")
info(f"  AUTHORS    = {AUTHORS}")
info(f"  INCREMENTAL_LIMIT = {INCREMENTAL_LIMIT}")

# ================================================================
# 3. Hard-coded defaults
# ================================================================
DATA_DIR = "data/author_jsons"
ALL_MD_DIR = "vitepress-project/docs/authors"
ALL_MD_FILE = "vitepress-project/docs/authors/index.md"

    
MAX_PAPERS_PER_AUTHOR = 150      # Hardcoded default
ENABLE_INCREMENTAL_MODE = True   # Hardcoded default
ENABLE_TRUNCATED_CHECK = True    # Hardcoded default
USE_ARXIV = True                 # Hardcoded default

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ALL_MD_DIR, exist_ok=True)


# ================================================================
# Helper functions
# ================================================================
def is_truncated(abs_text):
    if not abs_text:
        return False
    return abs_text.strip().endswith("...") or abs_text.strip().endswith("…")

def clean_html(raw):
    if not raw:
        return raw
    clean = BeautifulSoup(raw, "html.parser").get_text()
    clean = html.unescape(clean).strip()
    return clean

def load_state(author_id, author_name):
    safe = author_name.replace(" ", "").replace("/", "_")
    path = os.path.join(DATA_DIR, f"{safe}_{author_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(author_id, author_name, papers):
    safe = author_name.replace(" ", "").replace("/", "_")
    path = os.path.join(DATA_DIR, f"{safe}_{author_id}.json")
    papers = sorted(papers, key=lambda x: x["year"], reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

# =========================
# Retry Helper
# =========================
def retry(max_attempts=5, initial_wait=3, backoff=2):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            wait = initial_wait
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    warn(f"[RETRY] Attempt {attempt}/{max_attempts} failed: {e}")

                    if attempt == max_attempts:
                        error("[RETRY] Max retry attempts reached. Aborting.")
                        raise

                    sleep_time = wait + random.uniform(0, 1)
                    info(f"[RETRY] Waiting {sleep_time:.1f}s before next attempt ...")
                    time.sleep(sleep_time)
                    wait *= backoff
        return wrapper
    return decorator

@retry(max_attempts=5, initial_wait=3)
def safe_search_author(author_id):
    # return scholarly.search_author_id(author_id)
    return scholarly.search_author_id(author_id, sortby="year")

@retry(max_attempts=5, initial_wait=3)
def safe_fill(*args, **kwargs):
    return scholarly.fill(*args, **kwargs)


# ================================================================
# Full abstract only for arXiv
# ================================================================
def fetch_arxiv_abstract(url):
    try:
        paper_id = url.split("/")[-1]
        api = f"https://export.arxiv.org/api/query?id_list={paper_id}"
        r = requests.get(api, timeout=10)
        xml = r.text
        s = xml.find("<summary>") + len("<summary>")
        e = xml.find("</summary>")
        summary = xml[s:e].strip()
        return clean_html(summary)
    except:
        return None

def fetch_full_abstract(url, fallback_abs):
    if not ENABLE_TRUNCATED_CHECK or not is_truncated(fallback_abs):
        return fallback_abs

    info("Truncated abstract detected. Checking arXiv...")

    if USE_ARXIV and "arxiv.org" in url:
        full = fetch_arxiv_abstract(url)
        if full:
            info("Full arXiv abstract retrieved.")
            return full

    warn("Full abstract not found, using shortened version.")
    return fallback_abs

# ================================================================
# Fetch papers
# ================================================================
def fetch_author_papers(author_id, author_name):
    state = load_state(author_id, author_name)
    is_first = (len(state) == 0)

    info(f"Fetching papers for: {author_name}")

    author = scholarly.search_author_id(author_id)
    # safe fill with retry
    # author = safe_fill(author, sections=["publications"])
    author = safe_fill(author, sections=["publications"], sortby="year")

    papers = author["publications"]
    total = len(papers)
    info(f"Total papers found: {total}")

    # Limit
    papers = papers[:MAX_PAPERS_PER_AUTHOR]

    if not is_first and ENABLE_INCREMENTAL_MODE:
        papers = papers[:INCREMENTAL_LIMIT]
        info(f"Incremental mode enabled: checking top {INCREMENTAL_LIMIT} papers")

    existing_titles = {p["title"] for p in state}
    new_results = []

    for idx, pub in enumerate(papers, 1):
        info(f" ({idx}/{len(papers)}) Loading paper metadata ...")

        try:
            # safe search with retry
            pub_filled = safe_fill(pub)
            bib = pub_filled.get("bib", {})

            year_raw = bib.get("pub_year")
            if not year_raw:
                continue
            try:
                year = int(year_raw)
            except:
                continue

            if not (YEAR_START <= year <= YEAR_END):
                info(f"   Skipped (year {year} outside range {YEAR_START}-{YEAR_END})")
                continue

            title = bib.get("title", "Unknown Title")
            info(f" → {title} ({year})")

            short_abs = bib.get("abstract", "")
            link = pub_filled.get("pub_url", "")

            if not short_abs:
                warn(f"⚠ No abstract for: {title}")
                short_abs = "Abstract unavailable. This publication does not provide a summary using scholarly."
            if not link:
                warn(f"⚠ No link for: {title}")

            if not is_first and title in existing_titles:
                info("   Already exists in local cache. Skipped.")
                continue

            abstract = fetch_full_abstract(link, short_abs)

            new_results.append({
                "year": year,
                "title": title,
                "abstract": abstract,
                "link": link
            })

        except Exception:
            warn("Failed to load paper:")
            warn(traceback.format_exc())
            continue
    

    # 旧数据一定优先
    merged_dict = {}

    # 先写入旧数据（最完整）
    for p in state:
        merged_dict[p["title"]] = p

    # 再写入新数据（只有旧数据没有的才会加进去）
    for p in new_results:
        if p["title"] not in merged_dict:
            merged_dict[p["title"]] = p

    # 转回列表
    final = list(merged_dict.values())

    save_state(author_id, author_name, final)

    success(f"Added {len(new_results)} new papers. Total stored: {len(final)}")

    return final



# ================================================================
# Markdown
# ================================================================
def generate_md(author_name, papers):
    # 按年份降序
    papers.sort(key=lambda p: p["year"], reverse=True)

    grouped = {}
    for p in papers:
        grouped.setdefault(p["year"], []).append(p)

    md = f"## 📑 {author_name} Papers\n\n论文按年份分组（点击年份或空白区域可展开/折叠该年份的论文）\n\n"

    for year, group in grouped.items():
        md += textwrap.dedent(f"""
        <details class="year-block" open>
        <summary class="year-summary"><span class="icon">📅</span>{year}</summary>
        """)

        for p in group:
            # 1. 清理换行
            abs_clean = " ".join(p["abstract"].split())
            # 2. 对摘要做 HTML 转义，防止出现裸的 <、>、&
            abs_clean = html.escape(abs_clean)

            title_clean = html.escape(p["title"])

            md += textwrap.dedent(f"""
            <div class="paper-card">

            <h3 class="paper-title">{title_clean}</h3>

            <div class="paper-meta">📄 {p['year']}</div>

            <a class="paper-link" href="{p['link']}" target="_blank">🔗 Read Paper</a>

            <p class="paper-abstract">
            {abs_clean}
            </p>

            </div>
            """)

        md += "</details>\n\n"

    return md



# ================================================================
# Main
# ================================================================
def main():
    info("Starting fetch_papers.py ...")
    info("README.md will NOT be touched. Output is in /data/...")

    for author in AUTHORS:
        name = author["name"]
        author_id = author["id"]

        info("="*60)
        info(f"Processing: {name}")
        info("="*60)

        papers = fetch_author_papers(author_id, name)
        papers = [p for p in papers if YEAR_START <= p["year"] <= YEAR_END]
        papers.sort(key=lambda x: x["year"], reverse=True)

        # Individual MD
        safe_name = name.replace(" ", "").replace("/", "_")
        md_path = os.path.join(ALL_MD_DIR, f"{safe_name}.md")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generate_md(name, papers))

        success(f"Saved: {md_path}")

    # ❗ 不再生成 authors/index.md，保持静态自定义内容
    success("All author markdown files generated. index.md was NOT modified.")
    success("🎉 All done!")


if __name__ == "__main__":
    main()