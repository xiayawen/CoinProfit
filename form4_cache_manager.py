import time
import random
import requests
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# ============================================================
# 0. Config
# ============================================================

SEC_HEADERS = {
    "User-Agent": "Yawen Wang yawen20040129@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. SEC request helper with retry / rate-limit handling
# ============================================================

def sec_get(url, headers=SEC_HEADERS, timeout=30, max_retries=5, base_sleep=1.0):
    """
    Safer SEC GET request:
    - Handles 429 rate limit
    - Retries with longer waiting time
    - Sleeps briefly after every successful request
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)

            if r.status_code == 429:
                wait = 10 * (attempt + 1) + random.uniform(0, 3)
                print(f"SEC rate limited. Sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()

            time.sleep(base_sleep + random.uniform(0, 0.5))
            return r

        except requests.exceptions.RequestException as e:
            wait = 5 * (attempt + 1) + random.uniform(0, 3)
            print(f"Request error: {e}. Sleeping {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Too many failed requests for url: {url}")


# ============================================================
# 2. Ticker -> CIK cache
# ============================================================

_COMPANY_TICKERS_CACHE = None


def get_company_tickers_json():
    global _COMPANY_TICKERS_CACHE

    if _COMPANY_TICKERS_CACHE is None:
        url = "https://www.sec.gov/files/company_tickers.json"
        r = sec_get(url)
        _COMPANY_TICKERS_CACHE = r.json()

    return _COMPANY_TICKERS_CACHE


def get_cik_from_ticker(ticker: str) -> str:
    data = get_company_tickers_json()

    ticker = ticker.upper().strip()
    for rec in data.values():
        if rec["ticker"].upper() == ticker:
            return str(rec["cik_str"]).zfill(10)

    raise ValueError(f"Ticker not found: {ticker}")


# ============================================================
# 3. SEC submissions / filing metadata
# ============================================================

def get_company_submissions(cik_10: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik_10}.json"
    r = sec_get(url)
    return r.json()


def get_form4_filings_df(submissions_json: dict) -> pd.DataFrame:
    recent = submissions_json["filings"]["recent"]
    df = pd.DataFrame(recent)

    df = df[df["form"].isin(["4", "4/A"])].copy()

    keep_cols = [
        "filingDate",
        "acceptanceDateTime",
        "accessionNumber",
        "primaryDocument",
        "form",
    ]
    df = df[keep_cols]

    df["acceptanceDateTime"] = pd.to_datetime(
        df["acceptanceDateTime"], errors="coerce"
    )
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")

    df = df.sort_values("acceptanceDateTime", ascending=False).reset_index(drop=True)
    return df


def filing_index_url(cik_10: str, accession_number: str) -> str:
    cik_no_zero = str(int(cik_10))
    accession_nodash = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zero}/{accession_nodash}/{accession_number}-index.html"
    )


# ============================================================
# 4. Find and parse Form 4 XML
# ============================================================

def find_xml_document_from_index(index_url: str) -> str | None:
    r = sec_get(index_url)
    soup = BeautifulSoup(r.text, "html.parser")

    candidates = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if href.lower().endswith(".xml"):
            full_url = urljoin(index_url, href)
            candidates.append(full_url)

    for xml_url in candidates:
        try:
            rx = sec_get(xml_url)
            text = rx.text.strip()

            if "<ownershipDocument>" in text or "<ownershipDocument " in text:
                return xml_url

        except Exception as e:
            print(f"Skipping XML candidate due to error: {e}")
            continue

    return None


def text_or_none(node, path):
    if node is None:
        return None
    x = node.find(path)
    return x.text.strip() if x is not None and x.text is not None else None


def parse_form4_xml(xml_url: str):
    r = sec_get(xml_url)
    root = ET.fromstring(r.content)

    issuer_cik = text_or_none(root, ".//issuer/issuerCik")
    issuer_ticker = text_or_none(root, ".//issuer/issuerTradingSymbol")
    owner_name = text_or_none(root, ".//reportingOwner/reportingOwnerId/rptOwnerName")

    relationship = root.find(".//reportingOwner/reportingOwnerRelationship")
    is_director = text_or_none(relationship, "isDirector")
    is_officer = text_or_none(relationship, "isOfficer")
    officer_title = text_or_none(relationship, "officerTitle")
    is_ten_pct = text_or_none(relationship, "isTenPercentOwner")

    non_derivative_rows = []
    for txn in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        non_derivative_rows.append({
            "issuer_cik": issuer_cik,
            "issuer_ticker": issuer_ticker,
            "owner_name": owner_name,
            "is_director": is_director,
            "is_officer": is_officer,
            "officer_title": officer_title,
            "is_ten_percent_owner": is_ten_pct,
            "security_title": text_or_none(txn, "securityTitle/value"),
            "transaction_date": text_or_none(txn, "transactionDate/value"),
            "transaction_code": text_or_none(txn, "transactionCoding/transactionCode"),
            "shares": text_or_none(txn, "transactionAmounts/transactionShares/value"),
            "price_per_share": text_or_none(txn, "transactionAmounts/transactionPricePerShare/value"),
            "acquired_disposed_code": text_or_none(txn, "transactionAmounts/transactionAcquiredDisposedCode/value"),
            "shares_owned_following": text_or_none(txn, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"),
            "ownership_nature": text_or_none(txn, "ownershipNature/directOrIndirectOwnership/value"),
        })

    derivative_rows = []
    for txn in root.findall(".//derivativeTable/derivativeTransaction"):
        derivative_rows.append({
            "issuer_cik": issuer_cik,
            "issuer_ticker": issuer_ticker,
            "owner_name": owner_name,
            "is_director": is_director,
            "is_officer": is_officer,
            "officer_title": officer_title,
            "is_ten_percent_owner": is_ten_pct,
            "security_title": text_or_none(txn, "securityTitle/value"),
            "transaction_date": text_or_none(txn, "transactionDate/value"),
            "transaction_code": text_or_none(txn, "transactionCoding/transactionCode"),
            "shares": text_or_none(txn, "transactionAmounts/transactionShares/value"),
            "price_per_share": text_or_none(txn, "transactionAmounts/transactionPricePerShare/value"),
            "acquired_disposed_code": text_or_none(txn, "transactionAmounts/transactionAcquiredDisposedCode/value"),
            "shares_owned_following": text_or_none(txn, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"),
            "ownership_nature": text_or_none(txn, "ownershipNature/directOrIndirectOwnership/value"),
            "conversion_or_exercise_price": text_or_none(txn, "conversionOrExercisePrice/value"),
        })

    return pd.DataFrame(non_derivative_rows), pd.DataFrame(derivative_rows)


# ============================================================
# 5. Parse selected filings for one ticker
# ============================================================

def parse_form4_filings_for_ticker(ticker: str, filings: pd.DataFrame):
    ticker = ticker.upper().strip()
    cik_10 = get_cik_from_ticker(ticker)

    all_non_deriv = []
    all_deriv = []
    total = len(filings)

    for i, (_, row) in enumerate(filings.iterrows(), start=1):
        print(f"  {ticker}: parsing filing {i}/{total}")

        try:
            idx_url = filing_index_url(cik_10, row["accessionNumber"])
            xml_url = find_xml_document_from_index(idx_url)

            if xml_url is None:
                continue

            non_df, der_df = parse_form4_xml(xml_url)

            for df in [non_df, der_df]:
                if not df.empty:
                    df["ticker"] = ticker
                    df["filing_date"] = row["filingDate"]
                    df["acceptance_datetime"] = row["acceptanceDateTime"]
                    df["accession_number"] = row["accessionNumber"]
                    df["xml_url"] = xml_url

            if not non_df.empty:
                all_non_deriv.append(non_df)
            if not der_df.empty:
                all_deriv.append(der_df)

        except Exception as e:
            print(f"  Failed filing {row['accessionNumber']} for {ticker}: {e}")
            continue

    common = pd.concat(all_non_deriv, ignore_index=True) if all_non_deriv else pd.DataFrame()
    options = pd.concat(all_deriv, ignore_index=True) if all_deriv else pd.DataFrame()

    return common, options


# ============================================================
# 6. Crawl Form 4 data for one ticker
# ============================================================

def get_form4_data_for_ticker(ticker: str, n: int | None = 50):
    """
    Get Form 4 / 4-A data for one ticker.

    n=50: latest 50 Form 4 filings
    n=None: all Form 4 filings available in SEC recent submissions
    """
    ticker = ticker.upper().strip()
    cik_10 = get_cik_from_ticker(ticker)
    submissions = get_company_submissions(cik_10)

    filings = get_form4_filings_df(submissions)
    if n is not None:
        filings = filings.head(n).copy()

    filings = filings.copy()
    filings["ticker"] = ticker

    common, options = parse_form4_filings_for_ticker(ticker, filings)
    return filings, common, options


# ============================================================
# 7. Local cache manager
# ============================================================

def get_form4_cache_paths(ticker: str, data_dir: Path = DATA_DIR):
    ticker = ticker.upper().strip()
    return {
        "filings": data_dir / f"{ticker}_filings.csv",
        "common": data_dir / f"{ticker}_common.csv",
        "options": data_dir / f"{ticker}_options.csv",
    }


def read_cached_form4(ticker: str, data_dir: Path = DATA_DIR):
    paths = get_form4_cache_paths(ticker, data_dir)

    filings = pd.read_csv(paths["filings"]) if paths["filings"].exists() else pd.DataFrame()
    common = pd.read_csv(paths["common"]) if paths["common"].exists() else pd.DataFrame()
    options = pd.read_csv(paths["options"]) if paths["options"].exists() else pd.DataFrame()

    for df in [filings, common, options]:
        if df.empty:
            continue
        for col in ["acceptanceDateTime", "filingDate", "filing_date", "transaction_date", "acceptance_datetime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return filings, common, options


def save_cached_form4(ticker: str, filings: pd.DataFrame, common: pd.DataFrame, options: pd.DataFrame, data_dir: Path = DATA_DIR):
    paths = get_form4_cache_paths(ticker, data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    filings.to_csv(paths["filings"], index=False)
    common.to_csv(paths["common"], index=False)
    options.to_csv(paths["options"], index=False)

    print(f"Saved cache for {ticker} to {data_dir}/")


def get_latest_sec_form4_metadata(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    cik_10 = get_cik_from_ticker(ticker)
    submissions = get_company_submissions(cik_10)
    filings_sec = get_form4_filings_df(submissions)
    filings_sec["ticker"] = ticker
    return filings_sec


def load_or_update_form4_for_ticker(ticker: str, n_if_no_cache: int | None = None, data_dir: Path = DATA_DIR):
    """
    Main cache function.

    If data/{ticker}_*.csv does not exist:
        crawl from scratch and save to data/.

    If local cache exists:
        check SEC filing metadata.
        If new accessionNumber exists, only crawl the new filings and append.
        Otherwise, directly return cached data.

    n_if_no_cache:
        None = crawl all Form 4 filings available in SEC recent submissions.
        50/100/etc = crawl only the latest n filings when no cache exists.
    """
    ticker = ticker.upper().strip()
    paths = get_form4_cache_paths(ticker, data_dir)

    has_filings_cache = paths["filings"].exists()

    if not has_filings_cache:
        print(f"No cache found for {ticker}. Crawling from scratch...")
        filings, common, options = get_form4_data_for_ticker(ticker, n=n_if_no_cache)
        save_cached_form4(ticker, filings, common, options, data_dir)
        return filings, common, options

    print(f"Cache found for {ticker}. Checking for updates...")
    filings_cached, common_cached, options_cached = read_cached_form4(ticker, data_dir)

    filings_sec = get_latest_sec_form4_metadata(ticker)

    if filings_cached.empty or "accessionNumber" not in filings_cached.columns:
        print(f"Cache for {ticker} is incomplete. Re-crawling from scratch...")
        filings, common, options = get_form4_data_for_ticker(ticker, n=n_if_no_cache)
        save_cached_form4(ticker, filings, common, options, data_dir)
        return filings, common, options

    cached_accessions = set(filings_cached["accessionNumber"].dropna().astype(str))
    new_filings = filings_sec[
        ~filings_sec["accessionNumber"].astype(str).isin(cached_accessions)
    ].copy()

    # parse older first, newer last; final output will be sorted descending
    new_filings = new_filings.sort_values("acceptanceDateTime", ascending=True)

    if new_filings.empty:
        print(f"{ticker}: cache is already up to date.")
        return filings_cached, common_cached, options_cached

    print(f"{ticker}: found {len(new_filings)} new Form 4 filings. Updating cache...")
    common_new, options_new = parse_form4_filings_for_ticker(ticker, new_filings)

    filings_updated = pd.concat([filings_cached, new_filings], ignore_index=True)
    filings_updated = filings_updated.drop_duplicates(subset=["accessionNumber"], keep="last")
    filings_updated = filings_updated.sort_values("acceptanceDateTime", ascending=False).reset_index(drop=True)

    common_updated = pd.concat([common_cached, common_new], ignore_index=True)
    options_updated = pd.concat([options_cached, options_new], ignore_index=True)

    # safer dedup keys for parsed transaction tables
    common_dedup_cols = [c for c in ["ticker", "accession_number", "owner_name", "transaction_date", "transaction_code", "shares", "price_per_share", "acquired_disposed_code"] if c in common_updated.columns]
    options_dedup_cols = [c for c in ["ticker", "accession_number", "owner_name", "transaction_date", "transaction_code", "shares", "price_per_share", "conversion_or_exercise_price", "acquired_disposed_code"] if c in options_updated.columns]

    if common_dedup_cols:
        common_updated = common_updated.drop_duplicates(subset=common_dedup_cols, keep="last")
    if options_dedup_cols:
        options_updated = options_updated.drop_duplicates(subset=options_dedup_cols, keep="last")

    save_cached_form4(ticker, filings_updated, common_updated, options_updated, data_dir)
    return filings_updated, common_updated, options_updated


# ============================================================
# 8. Load/update multiple tickers
# ============================================================

def load_or_update_form4_for_tickers(tickers, n_if_no_cache: int | None = None, data_dir: Path = DATA_DIR):
    all_filings = []
    all_common = []
    all_options = []

    for ticker in tickers:
        ticker = ticker.upper().strip()
        print(f"\nLoading/updating Form 4 data for {ticker}...")

        try:
            filings, common_df, options_df = load_or_update_form4_for_ticker(
                ticker,
                n_if_no_cache=n_if_no_cache,
                data_dir=data_dir,
            )

            if not filings.empty:
                all_filings.append(filings)
            if not common_df.empty:
                all_common.append(common_df)
            if not options_df.empty:
                all_options.append(options_df)

            sleep_time = 5 + random.uniform(0, 2)
            print(f"Finished {ticker}. Sleeping {sleep_time:.1f}s before next ticker...")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Failed for {ticker}: {e}")

    filings_all = pd.concat(all_filings, ignore_index=True) if all_filings else pd.DataFrame()
    common_all = pd.concat(all_common, ignore_index=True) if all_common else pd.DataFrame()
    options_all = pd.concat(all_options, ignore_index=True) if all_options else pd.DataFrame()

    print("filings_all:", filings_all.shape)
    print("common_all:", common_all.shape)
    print("options_all:", options_all.shape)

    return filings_all, common_all, options_all


# ============================================================
# 9. Example run
# ============================================================

if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "MSFT", "AMD"]

    filings_all, common_all, options_all = load_or_update_form4_for_tickers(
        tickers,
        n_if_no_cache=None,   # None = no cache 时从头到尾爬；如果被限速，可以先改成 50/100
        data_dir=DATA_DIR,
    )
