#!/usr/bin/env python3

import os
import sys
import re
import pathlib
from typing import List
from datetime import datetime

# PySpark imports
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, CountVectorizer, IDF,
    StringIndexer, OneHotEncoder, VectorAssembler, PCA
)

# ============================================================================
# 1. Configuration & Constants
# ============================================================================
APP_NAME = "Wine-Reviews-Preprocessing"
DATA_DIR = pathlib.Path("data")  # Data folder is in the same directory as the script
OUTPUT_DIR = pathlib.Path("outputs")

# Candidate CSV paths
DEFAULT_CSV_PATHS = [
    DATA_DIR / "winemag-data-130k-v2.csv",
    DATA_DIR / "winemag-data_first150k.csv",
    pathlib.Path("winemag-data-130k-v2.csv"),
]

# Text columns
TEXT_COLS = [
    "country", "province", "region_1", "region_2",
    "variety", "winery", "taster_name", "designation", "title"
]

# Core data columns (to be saved after cleaning)
CORE_COLS_TO_SAVE = [
    "country", "country_norm", "province", "province_norm",
    "region_1", "region_1_norm", "region_2", "region_2_norm",
    "variety", "variety_norm", "variety_canonical",
    "winery", "winery_norm", "designation", "designation_norm",
    "title", "title_norm", "taster_name", "taster_name_norm",
    "taster_twitter_handle", "description", "points",
    "price", "price_imputed", "price_winsor"
]

# Variety synonym mapping
VARIETY_MAP = {
    "cabernet sauvignon": ["cabernet sauv.", "cab sauv", "cabernet"],
    "sauvignon blanc": ["sauv. blanc"],
    "pinot noir": ["spatburgunder"],
    "pinot grigio": ["pinot gris"],
    "syrah": ["shiraz"],
    "grenache": ["garnacha"],
    "chenin blanc": ["steen"],
    "gruner veltliner": ["gruener veltliner", "gruner veltliner"],
}


# ============================================================================
# 2. Helper Functions
# ============================================================================

def log(msg: str):
    """Log output with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def find_input_csv() -> str:
    """Search for input CSV file"""
    for path in DEFAULT_CSV_PATHS:
        if path.exists():
            log(f"✓ Found data file: {path}")
            return str(path)

    raise FileNotFoundError(
        "No data file found. Please ensure one of the following paths exists:\n" +
        "\n".join(f"  - {p}" for p in DEFAULT_CSV_PATHS)
    )


def normalize_text_py(s: str) -> str:
    """Text normalization (Python UDF)"""
    if s is None:
        return None
    try:
        from unidecode import unidecode
        s = unidecode(str(s))
    except ImportError:
        pass  # Skip if unidecode is not installed

    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def null_report(df):
    """Missing value report"""
    total = df.count()
    stats = [
        F.count(F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), c)).alias(c)
        for c in df.columns
    ]
    row = df.select(*stats).collect()[0].asDict()
    return sorted(
        [(k, v, round(v / total, 4)) for k, v in row.items()],
        key=lambda x: x[1],
        reverse=True
    )


# ============================================================================
# 3. Main Processing Pipeline
# ============================================================================

def main():
    log("=" * 80)
    log("Wine Reviews - Data Preprocessing Task Started")
    log("=" * 80)

    # ------------------------------------------------------------------------
    # 3.1 Create Spark Session
    # ------------------------------------------------------------------------
    log("Creating Spark Session...")

    spark = (
        SparkSession.builder
        .appName(APP_NAME)
        # master will be passed by spark-submit --master
        .config("spark.driver.memory", "16g")
        .config("spark.executor.memory", "12g")
        .config("spark.memory.fraction", "0.85")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.autoBroadcastJoinThreshold", -1)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.legacy.OneHotEncoder.vectorOutput", "true")
    spark.sparkContext.setLogLevel("WARN")

    log("✓ Spark Session created successfully")
    log(f"  Version: {spark.version}")
    log(f"  Master: {spark.sparkContext.master}")
    log(f"  App ID: {spark.sparkContext.applicationId}")

    # ------------------------------------------------------------------------
    # 3.2 Locate & Read Data
    # ------------------------------------------------------------------------
    input_path = find_input_csv()
    OUTPUT_DIR.mkdir(exist_ok=True)

    log(f"Reading data: {input_path}")
    df_raw = (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(input_path, inferSchema=True)
    )

    log(f"✓ Data read complete, raw rows: {df_raw.count():,}")

    # Remove junk index columns
    for junk in ["Unnamed: 0", "_c0", "index"]:
        if junk in df_raw.columns:
            df_raw = df_raw.drop(junk)

    # Type corrections
    df = (
        df_raw
        .withColumn("points", F.col("points").cast(T.IntegerType()))
        .withColumn("price", F.regexp_replace(F.col("price").cast(T.StringType()), ",", ""))
        .withColumn("price", F.col("price").cast(T.DoubleType()))
    )

    log(f"✓ Type correction complete, column count: {len(df.columns)}")

    # ------------------------------------------------------------------------
    # 3.3 String Normalization
    # ------------------------------------------------------------------------
    log("Performing string normalization...")

    normalize_udf = F.udf(normalize_text_py, T.StringType())

    for c in TEXT_COLS:
        if c in df.columns:
            df = df.withColumn(f"{c}_norm", normalize_udf(F.col(c)))

    log("✓ String normalization done")

    # ------------------------------------------------------------------------
    # 3.4 Deduplication
    # ------------------------------------------------------------------------
    log("Performing deduplication...")

    df = df.withColumn(
        "desc_fingerprint",
        F.lower(F.regexp_replace(F.col("description"), r"[^A-Za-z0-9]+", " "))
    )
    df = df.withColumn(
        "row_fingerprint",
        F.sha2(F.concat_ws("||",
                           F.coalesce(F.col("title"), F.lit("")),
                           F.coalesce(F.col("winery"), F.lit("")),
                           F.coalesce(F.col("desc_fingerprint"), F.lit(""))
                           ), 256)
    )

    before = df.count()
    df_dedup = df.dropDuplicates(["row_fingerprint"]).drop("row_fingerprint", "desc_fingerprint")
    after = df_dedup.count()

    log(f"✓ Deduplication complete: {before:,} -> {after:,} (removed {before - after:,})")

    # ------------------------------------------------------------------------
    # 3.5 Missing Value & Outlier Handling
    # ------------------------------------------------------------------------
    log("Handling missing values and outliers...")

    df_clean = df_dedup

    # Handle invalid price
    df_clean = df_clean.withColumn(
        "price",
        F.when(F.col("price").isNotNull() & (F.col("price") <= 0), None).otherwise(F.col("price"))
    )

    # Clean empty text fields
    for c in TEXT_COLS:
        norm = f"{c}_norm"
        if norm in df_clean.columns:
            df_clean = df_clean.withColumn(
                norm,
                F.when(F.trim(F.col(norm)) == "", None).otherwise(F.col(norm))
            )

    # Median imputation by group
    groupby_keys = [c for c in ["country_norm", "variety_norm"] if c in df_clean.columns]
    if groupby_keys:
        med = (
            df_clean
            .groupBy(*groupby_keys)
            .agg(F.expr("percentile_approx(price, 0.5)").alias("med_price"))
        )
        df_clean = (
            df_clean.join(med, on=groupby_keys, how="left")
            .withColumn("price_imputed",
                        F.when(F.col("price").isNull(), F.col("med_price")).otherwise(F.col("price")))
            .drop("med_price")
        )
    else:
        df_clean = df_clean.withColumn("price_imputed", F.col("price"))

    # Winsorization (1%-99%)
    q = df_clean.approxQuantile("price_imputed", [0.01, 0.99], 0.01)
    lower, upper = (q + [None, None])[:2]
    if lower is not None and upper is not None:
        df_clean = df_clean.withColumn(
            "price_winsor",
            F.when(F.col("price_imputed") < lower, F.lit(lower))
            .when(F.col("price_imputed") > upper, F.lit(upper))
            .otherwise(F.col("price_imputed"))
        )
    else:
        df_clean = df_clean.withColumn("price_winsor", F.col("price_imputed"))

    log(f"✓ Missing value handling complete, final row count: {df_clean.count():,}")

    # Print missing value report (top 5)
    null_stats = null_report(df_clean)[:5]
    log("  Missing Values TOP5:")
    for col, cnt, pct in null_stats:
        log(f"    {col}: {cnt:,} ({pct:.2%})")

    # ------------------------------------------------------------------------
    # 3.6 Variety Normalization
    # ------------------------------------------------------------------------
    log("Normalizing variety names...")

    alias_to_canonical = {}
    for canon, aliases in VARIETY_MAP.items():
        alias_to_canonical[canon] = canon
        for a in aliases:
            alias_to_canonical[a] = canon

    broadcast_map = spark.sparkContext.broadcast(alias_to_canonical)

    @F.udf(T.StringType())
    def map_variety_norm(v):
        if v is None:
            return None
        v = v.strip().lower()
        if v in broadcast_map.value:
            return broadcast_map.value[v]
        if v.endswith(" blend"):
            return v.replace(" blend", "").strip()
        return v

    if "variety_norm" in df_clean.columns:
        df_clean = df_clean.withColumn("variety_canonical", map_variety_norm(F.col("variety_norm")))
    else:
        df_clean = df_clean.withColumn("variety_canonical", F.col("variety"))

    log("✓ Variety normalization complete")

    # ------------------------------------------------------------------------
    # 3.7 Save Cleaned Data
    # ------------------------------------------------------------------------
    log("Saving cleaned data...")

    core_cols = [c for c in CORE_COLS_TO_SAVE if c in df_clean.columns]
    record_count = df_clean.count()

    (df_clean
     .select(*core_cols)
     .write.mode("overwrite")
     .option("compression", "snappy")
     .option("parquet.block.size", 64 * 1024 * 1024)
     .parquet(str(OUTPUT_DIR / "wine_clean.parquet"))
     )

    log(f"✓ Cleaned data saved: {OUTPUT_DIR / 'wine_clean.parquet'} ({record_count:,} rows)")

    # ------------------------------------------------------------------------
    # 3.8 TF-IDF Text Features
    # ------------------------------------------------------------------------
    log("Extracting text features (TF-IDF)...")

    df_text = df_clean

    # Clean description
    df_text = df_text.withColumn(
        "description_clean",
        F.lower(F.regexp_replace(F.col("description"), r"[^A-Za-z\s]+", " "))
    )

    tokenizer = RegexTokenizer(
        inputCol="description_clean",
        outputCol="tokens_raw",
        pattern="\\W+"
    )
    remover = StopWordsRemover(
        inputCol="tokens_raw",
        outputCol="tokens"
    )
    cv = CountVectorizer(
        inputCol="tokens",
        outputCol="tf",
        minDF=10
    )
    idf = IDF(
        inputCol="tf",
        outputCol="tfidf",
        minDocFreq=10
    )

    text_pipe = Pipeline(stages=[tokenizer, remover, cv, idf])

    log("  Training TF-IDF model...")
    text_model = text_pipe.fit(df_text)
    df_text_feats = text_model.transform(df_text)

    log("✓ TF-IDF extraction complete")

    # ------------------------------------------------------------------------
    # 3.9 PCA Reduction
    # ------------------------------------------------------------------------
    log("Performing PCA on TF-IDF...")

    tfidf_dim = df_text_feats.select("tfidf").first()["tfidf"].size
    k = max(100, min(500, tfidf_dim))

    log(f"  TF-IDF original dimension: {tfidf_dim}, PCA target dimension: {k}")

    pca = PCA(k=k, inputCol="tfidf", outputCol="tfidf_pca")
    pca_model = pca.fit(df_text_feats)
    df_text_feats = pca_model.transform(df_text_feats)

    log("✓ PCA reduction complete")

    # ------------------------------------------------------------------------
    # 3.10 Structured Feature Encoding
    # ------------------------------------------------------------------------
    log("Encoding structured features...")

    categorical_cols = [
        c for c in ["country_norm", "variety_canonical", "province_norm"]
        if c in df_text_feats.columns
    ]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_oh"])
        for c in categorical_cols
    ]

    struct_pipe = Pipeline(stages=[*indexers, *encoders])

    log("  Training structured feature model...")
    struct_model = struct_pipe.fit(df_text_feats)
    df_ready_feats = struct_model.transform(df_text_feats)

    log("✓ Structured feature encoding complete")

    # ------------------------------------------------------------------------
    # 3.11 Feature Assembly
    # ------------------------------------------------------------------------
    log("Assembling final feature vector...")

    numeric_cols = [c for c in ["points", "price_winsor"] if c in df_ready_feats.columns]

    assembler_struct = VectorAssembler(
        inputCols=[*numeric_cols, *[f"{c}_oh" for c in categorical_cols]],
        outputCol="struct_features",
        handleInvalid="keep"
    )
    df_struct = assembler_struct.transform(df_ready_feats)

    if "features" in df_struct.columns:
        df_struct = df_struct.drop("features")

    assembler_full = VectorAssembler(
        inputCols=["struct_features", "tfidf_pca"],
        outputCol="features",
        handleInvalid="keep"
    )
    df_ready = assembler_full.transform(df_struct)

    log("✓ Feature assembly complete")

    # ------------------------------------------------------------------------
    # 3.12 Save ML-Ready Data
    # ------------------------------------------------------------------------
    log("Saving ML-ready data...")

    df_ready_to_save = df_ready.select(
        *[c for c in core_cols if c in df_ready.columns],
        "tfidf_pca", "features"
    )

    df_ready_to_save.persist(StorageLevel.MEMORY_AND_DISK)

    (df_ready_to_save
     .repartition(8)
     .write.mode("overwrite")
     .option("compression", "snappy")
     .option("parquet.block.size", 64 * 1024 * 1024)
     .parquet(str(OUTPUT_DIR / "wine_ml_ready.parquet"))
     )

    log(f"✓ ML-ready data saved: {OUTPUT_DIR / 'wine_ml_ready.parquet'}")

    df_ready_to_save.unpersist()

    # ------------------------------------------------------------------------
    # 3.13 Save Preprocessing Models
    # ------------------------------------------------------------------------
    log("Saving preprocessing models...")

    models_dir = OUTPUT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    text_model.write().overwrite().save(str(models_dir / "text_pipeline"))
    struct_model.write().overwrite().save(str(models_dir / "struct_pipeline"))
    pca_model.write().overwrite().save(str(models_dir / "pca_model"))

    log(f"✓ Models saved: {models_dir}")
    log("  - text_pipeline (tokenization / TF-IDF)")
    log("  - struct_pipeline (indexing / encoding)")
    log("  - pca_model (PCA dimensionality reduction)")

    # ------------------------------------------------------------------------
    # 3.14 Output Summary
    # ------------------------------------------------------------------------
    log("=" * 80)
    log("Processing Summary:")
    log(f"  Raw data: {df_raw.count():,} rows")
    log(f"  After deduplication: {after:,} rows")
    log(f"  Final output: {record_count:,} rows")
    log(f"  Feature dimension: {df_ready.select('features').first()['features'].size}")
    log("=" * 80)

    # ------------------------------------------------------------------------
    # 3.15 Cleanup & Stop
    # ------------------------------------------------------------------------
    log("Stopping Spark Session...")
    spark.stop()
    log("✓ Task complete!")
    log("=" * 80)


# ============================================================================
# 4. Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        log(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
