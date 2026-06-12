import json
import os
import re
import sys
import time
import logging
from pathlib import Path
from html.parser import HTMLParser
from typing import Optional, List

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://kb.wisc.edu"
DATA_DIR = Path(__file__).parent.parent / "data" / "kbs"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
HEADERS = {"User-Agent": "DoIT-KB-Agent/1.0 (pallavisharm@wisc.edu; educational research)"}
REQUEST_DELAY = 0.5  # seconds between requests

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/pipeline/feature-extraction"

# Each entry: (search_query, category_slug, category_label, max_articles)
CATEGORIES = [
    # --- Original 6 ---
    ("O365 Outlook email affiliation deactivation", "microsoft365", "O365", 25),
    ("NetID activation expiry password reset", "iam", "NetID", 20),
    ("Duo MFA reactivation setup bypass passcode", "iam", "Duo_MFA", 20),
    ("GlobalProtect VPN setup troubleshooting", "helpdesk", "VPN", 15),
    ("eduroam UWNet WiFi wireless setup", "ns", "WiFi", 15),
    ("printing campus printers quota WiscPrint", "helpdesk", "Printing", 15),

    # --- Tier 1: Highest ticket volume ---
    ("Canvas LMS course access quiz grades student", "learn@uw", "Canvas", 20),
    ("Microsoft Office install download license activation", "microsoft365", "Office_Install", 20),
    ("Adobe Creative Cloud activation license student", "helpdesk", "Adobe_CC", 15),
    ("phishing email security URL defense suspicious", "security", "Phishing", 15),
    ("Zoom getting started meeting recording troubleshoot", "helpdesk", "Zoom", 20),
    ("Box OneDrive Google Drive storage quota file sharing", "helpdesk", "Cloud_Storage", 20),
    ("campus software library download SPSS SAS license", "helpdesk", "Software_Library", 15),
    ("antivirus endpoint protection Trend Micro Windows Defender", "security", "Antivirus", 15),

    # --- Tier 2: Moderate volume ---
    ("remote desktop RDS connection access campus", "helpdesk", "Remote_Desktop", 15),
    ("Microsoft Teams meeting chat cache troubleshoot", "microsoft365", "Teams", 15),
    ("macOS Windows OS update troubleshoot setup", "helpdesk", "OS_Support", 15),
    ("computer repair loaner laptop DoIT lending program", "helpdesk", "Computer_Repair", 10),
    ("Student Center SIS enrollment transcript registration", "helpdesk", "Student_Center", 15),
    ("Cisco VoIP campus phone voicemail PIN setup", "telecom", "VoIP", 15),
    ("SPSS SAS statistical software install Mac Windows", "helpdesk", "Stats_Software", 10),

    # --- Tier 3: Specialized ---
    ("ResearchDrive research data storage quota", "helpdesk", "ResearchDrive", 10),
    ("Workspace ONE device enrollment MDM macOS Windows", "helpdesk", "Endpoint_Mgmt", 10),
    ("classroom AV projector audio video technology support", "helpdesk", "Classroom_AV", 10),
    ("Kaltura lecture capture video upload Canvas media", "learn@uw", "Kaltura", 10),
    ("DoIT web hosting WordPress domain database", "helpdesk", "Web_Hosting", 10),
    ("Campus Active Directory AD join permissions IAM", "iam", "Active_Directory", 10),
    ("data classification sensitive restricted policy encryption", "security", "Data_Policy", 10),
    ("Cisco Webex video conferencing hybrid room meeting", "telecom", "Webex", 10),
    ("Google Workspace Gmail Drive Meet UW-Madison", "helpdesk", "Google_Workspace", 15),
]


# HTML parsing helpers

class _DocBodyExtractor(HTMLParser):
    """Extracts text inside class="doc-body", skipping script/style tags."""

    def __init__(self):
        super().__init__()
        self._in_doc = False
        self._depth = 0
        self._skip_tag = False  # inside <script> or <style>
        self.text_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if attrs_d.get("class") == "doc-body":
            self._in_doc = True
            self._depth = 1
            return
        if self._in_doc:
            if tag in ("script", "style"):
                self._skip_tag = True
            self._depth += 1

    def handle_endtag(self, tag):
        if self._in_doc:
            if tag in ("script", "style"):
                self._skip_tag = False
            self._depth -= 1
            if self._depth <= 0:
                self._in_doc = False

    def handle_data(self, data):
        if self._in_doc and not self._skip_tag:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(stripped)


def _extract_body(html: str) -> str:
    # Truncate at doc-info to avoid footer/metadata bleed
    cutoff = html.find('class="doc-info"')
    if cutoff > 0:
        html = html[:cutoff]
    p = _DocBodyExtractor()
    p.feed(html)
    return " ".join(p.text_parts)


def _extract_title(html: str) -> str:
    m = re.search(r"<title>([^<]+)</title>", html)
    if not m:
        return ""
    title = m.group(1).strip()
    # Strip trailing " - University of Wisconsin KnowledgeBase" suffix
    title = re.sub(r"\s*[-|]\s*University of Wisconsin.*$", "", title)
    return title


def _extract_canonical_url(html: str) -> str:
    m = re.search(r'<link rel="canonical" href="([^"]+)"', html)
    return m.group(1) if m else ""


def _extract_group_from_canonical(canonical: str) -> str:
    """kb.wisc.edu/iam/1140 → 'iam'"""
    m = re.match(r"https://kb\.wisc\.edu/([^/]+)/\d+", canonical)
    return m.group(1) if m else ""


# Search: get article IDs for a query

def _search_article_ids(query: str, limit: int = 50) -> list[int]:
    """Return article IDs from KB search (table format, server-rendered)."""
    url = f"{BASE_URL}/search.php"
    params = {"q": query, "flt": "1", "format": "table", "limit": limit}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Search request failed for '%s': %s", query, e)
        return []

    # Article rows look like: <a href="12345">Title</a> where href is just a numeric ID
    ids = re.findall(r'<a href="(\d{3,6})">', resp.text)
    seen = set()
    result = []
    for id_str in ids:
        aid = int(id_str)
        if aid not in seen:
            seen.add(aid)
            result.append(aid)
    return result


# Article fetch + parse

def _fetch_article(article_id: int, category: str) -> Optional[dict]:
    url = f"{BASE_URL}/{article_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Failed to fetch article %s: %s", article_id, e)
        return None

    html = resp.text
    title = _extract_title(html)
    body = _extract_body(html)
    canonical = _extract_canonical_url(html)

    if not title or not body or "Page Not Found" in title:
        log.debug("Skipping %s — no content", article_id)
        return None

    # Prefer the canonical URL; fall back to the redirect URL
    final_url = canonical if canonical else resp.url

    return {
        "id": str(article_id),
        "title": title,
        "body": body,
        "category": category,
        "url": final_url,
    }


# Scrape pipeline

def scrape_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total_written = 0

    for query, group_slug, category_label, max_articles in CATEGORIES:
        log.info("=== Scraping category: %s (query: '%s') ===", category_label, query)
        ids = _search_article_ids(query, limit=max_articles + 10)
        log.info("  Found %d candidate article IDs", len(ids))

        written = 0
        for article_id in ids:
            if written >= max_articles:
                break

            out_path = DATA_DIR / f"{article_id}.json"
            if out_path.exists():
                log.debug("  Skipping %s (already scraped)", article_id)
                written += 1
                continue

            time.sleep(REQUEST_DELAY)
            article = _fetch_article(article_id, category_label)
            if article is None:
                continue

            out_path.write_text(json.dumps(article, ensure_ascii=False, indent=2))
            log.info("  Saved %s: %s", article_id, article["title"][:60])
            written += 1

        log.info("  Wrote %d articles for %s", written, category_label)
        total_written += written

    log.info("Scraping complete. Total articles: %d", total_written)


# HuggingFace Inference API — embeddings

def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY", "")
    if not token:
        raise EnvironmentError(
            "Set HF_TOKEN (or HUGGINGFACE_API_KEY) to your HuggingFace token before indexing."
        )
    return token


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Call HF feature-extraction pipeline; return one embedding vector per text."""
    resp = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {_hf_token()}"},
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    # HF returns shape [n_texts, seq_len, hidden] for feature-extraction;
    # mean-pool over the sequence dimension to get [n_texts, hidden].
    embeddings = []
    for item in result:
        if isinstance(item[0], list):
            # shape [seq_len, hidden] → mean pool
            seq = item
            vec = [sum(tok[i] for tok in seq) / len(seq) for i in range(len(seq[0]))]
        else:
            # already [hidden]
            vec = item
        embeddings.append(vec)
    return embeddings


def embed_query(text: str) -> List[float]:
    """Embed a single query string at retrieval time. Called by retriever.py."""
    return _embed_batch([text])[0]


# ChromaDB index pipeline

def index_all() -> None:
    try:
        import chromadb
    except ImportError:
        log.error("chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    # Validate token early before doing any work
    _hf_token()

    files = list(DATA_DIR.glob("*.json"))
    if not files:
        log.error("No scraped articles found in %s. Run scrape first.", DATA_DIR)
        sys.exit(1)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # embedding_function=None → we supply raw embeddings ourselves
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        "kb_articles",
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 16  # HF free tier: keep batches small to avoid timeouts
    batch_ids, batch_docs, batch_embeddings, batch_metas = [], [], [], []

    for f in files:
        article = json.loads(f.read_text())
        doc_text = f"{article['title']}\n\n{article['body']}"
        batch_ids.append(article["id"])
        batch_docs.append(doc_text)
        batch_metas.append({
            "title": article["title"],
            "category": article["category"],
            "url": article["url"],
        })

        if len(batch_ids) >= BATCH:
            log.info("Embedding batch of %d articles via HF API...", BATCH)
            batch_embeddings = _embed_batch(batch_docs)
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
            )
            log.info("Upserted %d articles", BATCH)
            batch_ids, batch_docs, batch_embeddings, batch_metas = [], [], [], []

    if batch_ids:
        log.info("Embedding final batch of %d articles via HF API...", len(batch_ids))
        batch_embeddings = _embed_batch(batch_docs)
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas,
        )
        log.info("Upserted %d articles", len(batch_ids))

    log.info("Indexing complete. Collection count: %d", collection.count())


# Entry point

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "scrape":
        scrape_all()
    elif cmd == "index":
        index_all()
    else:
        scrape_all()
        index_all()
