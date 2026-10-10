"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

All data wrangling lives here so the rest of the code just calls these
functions. Every cleaning decision is appended to CLEANING_LOG so that
run_all can print (and the report can quote) exactly what was done.

Pipeline (see run_all.py):
    df            = clean(load_raw())
    tr, va, te    = train_val_test_split_indices(...)        # stratified
    builder       = FeatureBuilder().fit(df.iloc[tr])        # TRAIN stats only
    X, y, names   = builder.transform(df)                    # y = log(price)
    y_tier, thr   = make_tier_label(price, threshold from TRAIN median)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/house_sale.csv"   # Kaggle: sehriyarmemmedli/binaaz-sale-project

# Columns that LEAK the target (arithmetic functions of price). Dropped.
#   unit_price  = price / area   ("3 450 AZN/m²")
#   total_price = price restated (identical to price in 99.99% of rows)
LEAKAGE_COLUMNS: list[str] = ["unit_price", "total_price"]

# Identifiers / free text / scrape metadata — not features (privacy, leakage
# risk or simply no signal available at listing time).
DROP_COLUMNS: list[str] = [
    "id_x", "id_y", "estate_id", "estate_details_id_x", "estate_details_id_y",
    "rel_url", "estate_rel_url_x", "estate_rel_url_y", "estate_rel_url",
    "img_url", "owner_name", "shop_name", "address",
    "description",          # free text, frequently restates the price
    "attributes",           # restatement of rooms / area / floor (parsed instead)
    "city_when", "day_x", "hour_x", "day_y", "hour_y", "updated",
    "datetime_scrape_x", "datetime_scrape_y",
    "views",                # accumulates AFTER listing -> not available at prediction time
    "currency_x", "currency_y",   # constant "AZN"
    "shop_title",           # duplicate of products_label
    "Binanın növü",         # 99.8 % missing
    "repair",               # duplicate of Təmir
    # (raw bill_of_sale / mortgage duplicate Çıxarış / İpoteka; they are
    #  overwritten in clean() by the parsed 0/1 versions of those columns)
]

# Azerbaijani -> English names for the columns we keep.
RENAME = {
    "Kateqoriya": "category",
    "Mərtəbə": "floor_raw",
    "Otaq sayı": "rooms",
    "Sahə": "area_raw",
    "Torpaq sahəsi": "land_raw",
    "Təmir": "repaired_raw",
    "Çıxarış": "bill_of_sale_raw",
    "İpoteka": "mortgage_raw",
}

CATEGORY_EN = {
    "Yeni tikili": "new_build",
    "Köhnə tikili": "old_build",
    "Həyət evi/Bağ evi": "house",
}

# Fixed, domain-based outlier rules (NOT data quantiles, so no statistic
# is learned from the test rows).
PRICE_MIN, PRICE_MAX = 10_000, 5_000_000          # AZN
AREA_MIN, AREA_MAX = 15, 1_500                    # m²
PPM2_MIN, PPM2_MAX = 150, 15_000                  # AZN per m² (cleaning only!)
LAND_MAX = 200                                    # sot (1 sot = 100 m²)
ROOMS_MAX = 15
BAKU_LAT = (40.20, 40.65)
BAKU_LNG = (49.55, 50.40)
BAKU_CENTER = (40.3703, 49.8352)                  # Fountain Square (fixed constant)

CLEANING_LOG: list[str] = []


def _log(msg: str) -> None:
    CLEANING_LOG.append(msg)


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame (UTF-8). Return it unmodified."""
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


# --- small parsers ----------------------------------------------------------
def _num(s: pd.Series) -> pd.Series:
    """'1 234.5 m²' -> 1234.5 ; strips units / thousands spaces."""
    s = s.astype("string").str.replace(r"[^\d.,]", "", regex=True).str.replace(",", ".")
    return pd.to_numeric(s, errors="coerce").astype(float)


def _floor_split(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'7 / 9' -> (7, 9)."""
    parts = s.astype("string").str.extract(r"^\s*(-?\d+)\s*/\s*(\d+)\s*$")
    return (pd.to_numeric(parts[0], errors="coerce").astype(float),
            pd.to_numeric(parts[1], errors="coerce").astype(float))


def _tokens(extra_info: pd.Series) -> pd.Series:
    return extra_info.fillna("").str.split(r"\s*\*\s*")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw frame (see CLEANING_LOG for the full, ordered record):
      scope -> deduplicate -> parse types -> outliers -> drop leakage/ids.
    Returns a tidy frame with English column names and a `price` column.
    """
    CLEANING_LOG.clear()
    n0 = len(df)
    _log(f"raw rows: {n0:,}, columns: {df.shape[1]}")

    # Currency: every row is AZN in this dump; assert instead of assume.
    cur = set(df["currency_x"].dropna().unique()) | set(df["currency_y"].dropna().unique())
    assert cur == {"AZN"}, f"unexpected currencies {cur}"
    _log("currency: all prices are AZN (checked currency_x/currency_y) -> no conversion")

    # 1) scope: residential listings in Baku
    df = df[df["Kateqoriya"].isin(CATEGORY_EN)].copy()
    _log(f"keep residential categories {list(CATEGORY_EN)} "
         f"(drop land/office/garage/commercial — priced per sot or a different market): {len(df):,}")
    df = df[df["city"] == "bakı"]
    _log(f"keep city == 'bakı' (99.7 % of rows): {len(df):,}")

    # 2) duplicates: the same listing is scraped on many days -> keep newest
    df["_ts"] = pd.to_datetime(df["datetime_scrape_x"], format="mixed", utc=True)
    df = df.sort_values(["_ts", "estate_rel_url_x"], ascending=[False, True])
    n = len(df)
    df = df.drop_duplicates("estate_rel_url_x", keep="first")
    _log(f"drop repeated scrapes of the same listing URL (keep newest): -{n - len(df):,} -> {len(df):,}")

    # 3) type coercion (numbers arrive as strings with units)
    df = df.rename(columns=RENAME)
    df["area_m2"] = _num(df["area_raw"])
    df["land_sot"] = _num(df["land_raw"])
    df["floor"], df["total_floors"] = _floor_split(df["floor_raw"])
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    df["repaired"] = df["repaired_raw"].map({"var": 1.0, "yoxdur": 0.0})   # NaN stays NaN
    df["bill_of_sale"] = (df["bill_of_sale_raw"] == "var").astype(float)
    df["mortgage"] = (df["mortgage_raw"] == "var").astype(float)            # NaN = not offered
    df["vip"] = df["vip"].notna().astype(float)
    df["featured"] = df["featured"].notna().astype(float)
    df["is_agent"] = (df["owner_title"] == "vasitəçi (agent)").astype(float)
    df["is_complex"] = (df["products_label"] == "Kompleks").astype(float)
    df["category"] = df["category"].map(CATEGORY_EN)
    _log("parse 'Sahə' ('145 m²'), 'Torpaq sahəsi' ('1.3 sot'), 'Mərtəbə' ('7 / 9' -> floor, total_floors); "
         "yes/no columns -> 0/1 (missing İpoteka/vip/featured = not offered)")

    # location: district (rayon, 'r.') and metro proximity from extra_info tokens
    toks = _tokens(df["extra_info"])
    df["district"] = toks.apply(lambda t: next((x for x in t if x.endswith(" r.")), "unknown"))
    df["near_metro"] = toks.apply(lambda t: any(x.endswith(" m.") for x in t)).astype(float)
    df["neighbourhood"] = df["location"].fillna("unknown")
    _log("location: district = the 'r.' token of extra_info; near_metro = any 'm.' token; "
         "neighbourhood = `location` (metro / settlement name)")

    # second duplicate pass: re-posts of the same flat under a new URL
    key = ["price", "area_m2", "rooms", "floor", "total_floors", "lat", "lng", "category"]
    n = len(df)
    df = df.drop_duplicates(key)
    _log(f"drop re-posted duplicates (same price, area, rooms, floor, coordinates, category): "
         f"-{n - len(df):,} -> {len(df):,}")

    # 4) outliers / impossible values (fixed domain rules)
    rules = {
        f"price outside [{PRICE_MIN:,}, {PRICE_MAX:,}] AZN": ~df["price"].between(PRICE_MIN, PRICE_MAX),
        f"area outside [{AREA_MIN}, {AREA_MAX}] m²": ~df["area_m2"].between(AREA_MIN, AREA_MAX),
        f"price per m² outside [{PPM2_MIN}, {PPM2_MAX:,}] AZN (typos / rent ads)":
            ~(df["price"] / df["area_m2"]).between(PPM2_MIN, PPM2_MAX),
        f"land area > {LAND_MAX} sot": df["land_sot"] > LAND_MAX,
        f"rooms > {ROOMS_MAX}": df["rooms"] > ROOMS_MAX,
        "floor > total floors or floor < 0": (df["floor"] > df["total_floors"]) | (df["floor"] < 0),
        "coordinates outside the Baku bounding box":
            ~(df["lat"].between(*BAKU_LAT) & df["lng"].between(*BAKU_LNG)),
    }
    for name, mask in rules.items():
        n = len(df)
        df = df[~mask.reindex(df.index, fill_value=False)]
        _log(f"outlier rule — {name}: -{n - len(df):,} -> {len(df):,}")

    # 5) leakage + identifiers
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    _log(f"drop LEAKAGE columns {LEAKAGE_COLUMNS} (functions of the target)")
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    _log("drop identifiers / free text / scrape metadata: owner_name, shop_name, address, ids, urls, "
         "description, views, timestamps")

    keep = ["price", "area_m2", "rooms", "floor", "total_floors", "land_sot", "lat", "lng",
            "category", "district", "neighbourhood", "near_metro", "repaired", "bill_of_sale",
            "mortgage", "vip", "featured", "is_agent", "is_complex"]
    df = df[keep].reset_index(drop=True)
    _log(f"final cleaned rows: {len(df):,} ({len(df) / n0:.1%} of raw)")
    return df


# --- features ----------------------------------------------------------------
NUMERIC = ["area_m2", "rooms", "floor", "total_floors", "land_sot", "lat", "lng"]
BINARY = ["near_metro", "repaired", "bill_of_sale", "mortgage", "vip", "featured",
          "is_agent", "is_complex"]


class FeatureBuilder:
    """
    Turns the cleaned frame into a numeric design matrix.

    Every statistic (imputation medians, which categories get their own
    one-hot column) is learned in fit() from the TRAINING rows only and
    then reused unchanged for validation / test rows.

    Encoding choice: one-hot. Trees split on single columns, so one-hot
    lets a split isolate one district; the SVM is linear, so an ordinal
    code for a nominal variable would impose a fake ordering.
    """

    def __init__(self, min_count: int = 100):
        self.min_count = min_count   # rarer neighbourhoods -> 'other'

    def fit(self, df_train: pd.DataFrame) -> "FeatureBuilder":
        self.medians_ = {c: float(df_train[c].median()) for c in NUMERIC + ["repaired"]}
        self.categories_ = sorted(df_train["category"].unique())
        self.districts_ = sorted(df_train["district"].value_counts()
                                 .loc[lambda s: s >= self.min_count].index)
        self.neighbourhoods_ = sorted(df_train["neighbourhood"].value_counts()
                                      .loc[lambda s: s >= self.min_count].index)
        return self

    def transform(self, df: pd.DataFrame):
        cols: dict[str, np.ndarray] = {}
        for c in NUMERIC:
            v = df[c].to_numpy(dtype=float)
            if c in ("floor", "total_floors", "land_sot", "rooms"):
                cols[f"{c}_missing"] = np.isnan(v).astype(float)
            if c == "land_sot":
                v = np.where(np.isnan(v), 0.0, v)   # flats have no land: 0 is the true value
            cols[c] = np.where(np.isnan(v), self.medians_[c], v)
        # engineered numeric features
        cols["floor_ratio"] = cols["floor"] / np.maximum(cols["total_floors"], 1)
        cols["is_top_floor"] = ((cols["floor"] == cols["total_floors"]) & (cols["floor_missing"] == 0)).astype(float)
        cols["is_ground_floor"] = ((cols["floor"] <= 1) & (cols["floor_missing"] == 0)).astype(float)
        cols["area_per_room"] = cols["area_m2"] / np.maximum(cols["rooms"], 1)
        cols["dist_center_km"] = _haversine_km(cols["lat"], cols["lng"], *BAKU_CENTER)
        for c in BINARY:
            v = df[c].to_numpy(dtype=float)
            if c == "repaired":
                cols["repaired_missing"] = np.isnan(v).astype(float)
                v = np.where(np.isnan(v), self.medians_["repaired"], v)
            cols[c] = v
        for cat in self.categories_:
            cols[f"cat={cat}"] = (df["category"] == cat).to_numpy(dtype=float)
        dist = df["district"].where(df["district"].isin(self.districts_), "other")
        for d in self.districts_ + ["other"]:
            cols[f"district={d}"] = (dist == d).to_numpy(dtype=float)
        nb = df["neighbourhood"].where(df["neighbourhood"].isin(self.neighbourhoods_), "other")
        for d in self.neighbourhoods_ + ["other"]:
            cols[f"nbhd={d}"] = (nb == d).to_numpy(dtype=float)

        names = list(cols)
        X = np.column_stack([cols[k] for k in names])
        y = np.log(df["price"].to_numpy(dtype=float))
        return X, y, names


def _haversine_km(lat, lng, lat0, lng0):
    lat, lng, lat0, lng0 = map(np.radians, (lat, lng, lat0, lng0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lng - lng0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def make_features(df: pd.DataFrame, train_idx=None):
    """
    Convenience wrapper: fit a FeatureBuilder on the training rows (all rows
    if train_idx is None) and transform the whole frame.
    Returns (X, y, feature_names) with y = log(price).
    """
    fb = FeatureBuilder().fit(df if train_idx is None else df.iloc[train_idx])
    return fb.transform(df)


def make_tier_label(y_price, threshold=None):
    """
    Price TIER: premium = 1 (price > threshold), standard = 0.
    Call it first on the TRAINING prices with threshold=None -> the
    threshold is the training median; then pass that threshold for val/test.
    Returns (y_tier, threshold).
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), float(threshold)


def stratified_split_indices(strata, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic stratified split of row indices into train / val / test.
    Within each stratum rows are shuffled and cut in the same proportions,
    so every split has the same stratum mix.
    """
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    tr, va, te = [], [], []
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        n_te = int(round(test_size * len(idx)))
        n_va = int(round(val_size * len(idx)))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    return tuple(np.sort(np.concatenate(p)) for p in (tr, va, te))


def price_strata(price, n_bins: int = 10):
    """
    Stratification key: price decile. Used ONLY to balance the split (each
    split gets the same price mix, hence the same tier balance); it is not
    a feature and does not set the tier threshold, which is re-computed
    from the training split alone.
    """
    price = np.asarray(price, dtype=float)
    edges = np.quantile(price, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, price, side="right")


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED, strata=None):
    """
    Deterministic split into train / validation / test (stratified on
    `strata` if given, else on price deciles of y).
    Returns (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    strata = price_strata(y) if strata is None else strata
    tr, va, te = stratified_split_indices(strata, val_size, test_size, seed)
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def standardize(X_tr, *others):
    """
    Standardize with TRAIN mean/std only, then apply the same transform to
    the other splits. Constant columns get std 1 (left centred at 0).
    Returns the scaled arrays in the same order as given.
    """
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    out = [(X_tr - mu) / sd] + [(X - mu) / sd for X in others]
    return out if others else out[0]
