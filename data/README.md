# Data

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`).

## Where to get it
Kaggle: **"binaaz-sale-project"** by `sehriyarmemmedli`, **version 1**
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

Archive downloaded on 2026-10-11; it contains one file, `house_sale.csv`
(141.9 MB, 100,775 rows × 51 columns, listings scraped 2024-10-05 … 2024-11-06).

## How to place it
Either run (from the repo root)

```bash
python -m src.get_data      # kagglehub download -> data/house_sale.csv
```

or download the archive by hand and extract it so the CSV is at

```
data/house_sale.csv
```

If your filename differs, change `DATA_PATH` at the top of `src/data_prep.py`.

## Notes
- Column names are Azerbaijani (`Sahə`, `Otaq sayı`, `Mərtəbə`, `Təmir`, …);
  read with UTF-8. They are translated in `data_prep.RENAME`.
- **Leakage:** `unit_price` (AZN per m²) and `total_price` (= `price`) are
  arithmetic functions of the target and are dropped (`data_prep.LEAKAGE_COLUMNS`).
- Identifiers and personal fields (`owner_name`, `shop_name`, `address`,
  ids, URLs) are dropped and never written to any output.
