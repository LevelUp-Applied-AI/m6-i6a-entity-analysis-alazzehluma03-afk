"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Build a corpus-level entity analysis pipeline that preprocesses
climate articles (with language-aware handling), extracts entities,
computes statistics, and produces visualizations.

Run: python entity_analysis.py
"""

import unicodedata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy


def load_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    #  Load the CSV and return the DataFrame unchanged
    df = pd.read_csv(filepath)
    return df


def preprocess_corpus(df):
    """Add a language-aware `processed_text` column to the corpus.

    For every row, apply Unicode NFC normalization to `text` so that
    visually identical characters (composed vs. decomposed diacritics)
    compare equal downstream. The processed form preserves
    capitalization and punctuation — those are signals NER depends on.

    For Arabic rows (`language == 'ar'`), do not attempt English NLP
    processing: either pass the NFC-normalized text through unchanged
    or store an empty string. Either choice must not crash the
    pipeline.

    Args:
        df: DataFrame returned by load_corpus.

    Returns:
        Copy of df with a new `processed_text` column. The original
        `text` column is left intact so NER can still consume it.
    """
    #  Copy df, apply unicodedata.normalize('NFC', t) to each
    #       text, branch on language for English vs. Arabic handling,
    #       write results into a new `processed_text` column
    df_copy = df.copy()

    def process_row(row):

        normalized_text = unicodedata.normalize(
            "NFC",
            str(row["text"])
        )

        if row["language"] == "en":
            return normalized_text

        elif row["language"] == "ar":
            return normalized_text

        return ""

    df_copy["processed_text"] = df_copy.apply(
        process_row,
        axis=1
    )

    return df_copy


def run_ner_pipeline(df, nlp):
    """Run spaCy NER on the English rows of a preprocessed corpus.

    Args:
        df: DataFrame with columns id, text, language, processed_text.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    #  Filter df to language == 'en', process each text with nlp,
    #       collect entities into rows, return as a DataFrame
    english_df = df[df["language"] == "en"]

    entity_rows = []

    for _, row in english_df.iterrows():

        doc = nlp(row["text"])

        for ent in doc.ents:
            entity_rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(entity_rows)


def aggregate_entity_stats(entity_df, articles_df):
    """Compute frequency, co-occurrence, and per-category statistics.

    Args:
        entity_df: DataFrame with columns text_id, entity_text,
                   entity_label.
        articles_df: The source corpus DataFrame (with columns id,
                     category, ...). Used to join category onto
                     each entity for per-category aggregation.

    Returns:
        Dictionary with keys:
          'top_entities': DataFrame of top 20 entities by frequency
                          (columns: entity_text, entity_label, count)
          'label_counts': dict of entity_label -> total count
          'co_occurrence': DataFrame of entity pairs appearing in the
                           same text (columns: entity_a, entity_b,
                           co_count). Cap at top 50 pairs by co_count
                           (or filter to co_count >= 2) so the result
                           stays readable on the full corpus.
          'per_category': DataFrame of entity-label counts broken out
                          by article category (columns: category,
                          entity_label, count)
    """
    #  Count entity frequencies (top 20), compute label totals,
    #       build co-occurrence pairs, and join on articles_df.id to
    #       compute per-category entity-label counts
    
    # top entities
    top_entities = (
        entity_df.groupby(
            ["entity_text", "entity_label"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
        .head(20)
    )

    # label counts
    label_counts = (
        entity_df["entity_label"]
        .value_counts()
        .to_dict()
    )

    # co-occurrence
    pair_counts = {}

    for text_id, group in entity_df.groupby("text_id"):

        entities = sorted(
            group["entity_text"]
            .drop_duplicates()
            .tolist()
        )

        for i in range(len(entities)):
            for j in range(i+1, len(entities)):

                pair = (
                    entities[i],
                    entities[j]
                )

                pair_counts[pair] = (
                    pair_counts.get(pair, 0) + 1
                )

    co_occurrence = pd.DataFrame(
        [
            {
                "entity_a": a,
                "entity_b": b,
                "co_count": c
            }
            for (a,b), c in pair_counts.items()
            if c >= 2
        ]
    )

    if not co_occurrence.empty:
        co_occurrence = (
            co_occurrence
            .sort_values(
                "co_count",
                ascending=False
            )
            .head(50)
        )

    # per category
    merged = entity_df.merge(
        articles_df[
            ["id", "category"]
        ],
        left_on="text_id",
        right_on="id"
    )

    per_category = (
        merged.groupby(
            ["category", "entity_label"]
        )
        .size()
        .reset_index(name="count")
    )

    print("\nEntity Statistics Summary")
    print(top_entities.head())
    print(label_counts)

    return {
        "top_entities": top_entities,
        "label_counts": label_counts,
        "co_occurrence": co_occurrence,
        "per_category": per_category
    }


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create a bar chart of the top 20 entities by frequency.

    Args:
        stats: Dictionary from aggregate_entity_stats (must contain
               'top_entities' DataFrame).
        output_path: File path to save the chart.
    """
    #  Create a horizontal bar chart of top entities, colored or
    #       grouped by entity type, save to output_path
    top_entities = stats["top_entities"]

    plt.figure(
        figsize=(12,8)
    )

    plt.barh(
        top_entities["entity_text"],
        top_entities["count"]
    )

    plt.xlabel("Frequency")
    plt.ylabel("Entity")

    plt.title(
        "Top 20 Climate Entities"
    )

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()


def generate_report(stats, co_occurrence):
    """Generate a text summary of entity analysis findings.

    Args:
        stats: Dictionary from aggregate_entity_stats.
        co_occurrence: Co-occurrence DataFrame from stats.

    Returns:
        String containing a structured report with: entity counts
        per type, top 5 most frequent entities, top 3 co-occurring
        pairs, and a brief summary.
    """
    
    #  Build a formatted report string from the statistics
    label_section = "\n".join(
        [
            f"{k}: {v}"
            for k,v in stats["label_counts"].items()
        ]
    )

    top5 = stats["top_entities"].head(5)

    top_entity_section = "\n".join(
        [
            f"{r.entity_text} ({r.entity_label}) - {r.count}"
            for _,r in top5.iterrows()
        ]
    )

    if (
        co_occurrence is not None
        and not co_occurrence.empty
    ):

        top3_pairs = co_occurrence.head(3)

        pair_section = "\n".join(
            [
                f"{r.entity_a} + {r.entity_b}: {r.co_count}"
                for _,r in top3_pairs.iterrows()
            ]
        )

    else:
        pair_section = "No significant co-occurrence pairs"

    report = f"""
ENTITY ANALYSIS REPORT

Entity Counts by Type:
{label_section}

Top 5 Entities:
{top_entity_section}

Top 3 Co-occurring Pairs:
{pair_section}

Summary:
    The entity analysis shows that the climate corpus is mainly driven by temporal (DATE), organizational (ORG), and geographical (GPE) entities. This reflects a strong focus on timelines, institutions, and countries in climate discussions. Numerical entities (CARDINAL, PERCENT) are also highly frequent, indicating data-heavy reporting. Co-occurrence patterns show meaningful relationships between countries, time targets, and climate actions, suggesting that entities are interconnected rather than isolated.
"""

    return report


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    # Load and preprocess the corpus
    raw = load_corpus()
    if raw is not None:
        corpus = preprocess_corpus(raw)
        if corpus is not None:
            print(f"Corpus: {len(corpus)} articles")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")
            print(f"Categories: {corpus['category'].value_counts().to_dict()}")

            # Run NER on English rows
            entities = run_ner_pipeline(corpus, nlp)
            if entities is not None:
                print(f"\nExtracted {len(entities)} entities")

                # Aggregate statistics
                stats = aggregate_entity_stats(entities, corpus)
                if stats is not None:
                    print(f"\nLabel counts: {stats['label_counts']}")
                    print(f"\nTop 5 entities:")
                    print(stats["top_entities"].head())
                    print(f"\nPer-category counts (head):")
                    print(stats["per_category"].head())

                    # Visualize
                    visualize_entity_distribution(stats)
                    print("\nVisualization saved to entity_distribution.png")

                    # Generate report
                    report = generate_report(stats, stats.get("co_occurrence"))
                    if report is not None:
                        print(f"\n{'='*50}")
                        print(report)
