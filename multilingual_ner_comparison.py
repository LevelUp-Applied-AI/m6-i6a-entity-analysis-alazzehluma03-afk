# ================================
# Multilingual NER Comparison
# ================================

import pandas as pd
import spacy
from transformers import pipeline
from collections import Counter

def load_data(filepath):
    df = pd.read_csv(filepath)
    assert "text" in df.columns
    assert "language" in df.columns
    return df

def split_languages(df, n=20):
    df["language"] = df["language"].str.lower()
    df_en = df[df["language"] == "en"].sample(n, random_state=42)
    df_ar = df[df["language"] == "ar"].sample(n, random_state=42)
    return df_en["text"].tolist(), df_ar["text"].tolist()

def load_spacy_model():
    return spacy.load("xx_ent_wiki_sm")

def load_hf_model():
    return pipeline(
        "ner",
        model="Davlan/xlm-roberta-base-wikiann-ner",
        aggregation_strategy="simple"
    )

def run_spacy(texts, nlp):
    results = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        results.append({"entities": entities, "count": len(entities), "words": len(text.split())})
    return results

def run_hf(texts, model):
    results = []
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        preds = model(text)
        entities = [(p["word"], p["entity_group"]) for p in preds]
        results.append({"entities": entities, "count": len(entities), "words": len(text.split())})
    return results

def analyze(results):
    total_entities = sum(r["count"] for r in results)
    total_words = sum(r["words"] for r in results)
    labels = Counter(ent[1] for r in results for ent in r["entities"])
    examples = [f"{text}({label})" for r in results for text, label in r["entities"][:3]][:3]
    density = round((total_entities / total_words * 100) if total_words else 0, 2)
    zero = round(sum(1 for r in results if r["count"] == 0) / len(results), 2)
    return {"total": total_entities, "density": density, "labels": labels, "examples": examples, "zero": zero}

def main():
    df = load_data("data/climate_articles.csv")
    en_texts, ar_texts = split_languages(df)

    spacy_model = load_spacy_model()
    hf_model = load_hf_model()

    sp_en = analyze(run_spacy(en_texts, spacy_model))
    sp_ar = analyze(run_spacy(ar_texts, spacy_model))
    hf_en = analyze(run_hf(en_texts, hf_model))
    hf_ar = analyze(run_hf(ar_texts, hf_model))

    # Create clean table
    data = {
        "Language": ["English", "Arabic", "English", "Arabic"],
        "Model": ["spaCy", "spaCy", "HF", "HF"],
        "Total": [sp_en["total"], sp_ar["total"], hf_en["total"], hf_ar["total"]],
        "Density": [sp_en["density"], sp_ar["density"], hf_en["density"], hf_ar["density"]],
        "Zero Rate": [sp_en["zero"], sp_ar["zero"], hf_en["zero"], hf_ar["zero"]],
        "Entity Types": [
            ", ".join([f"{k}:{v}" for k,v in sp_en["labels"].most_common(4)]),
            ", ".join([f"{k}:{v}" for k,v in sp_ar["labels"].most_common(4)]),
            ", ".join([f"{k}:{v}" for k,v in hf_en["labels"].most_common(4)]),
            ", ".join([f"{k}:{v}" for k,v in hf_ar["labels"].most_common(4)])
        ],
        "Examples": [
            " | ".join(sp_en["examples"]),
            " | ".join(sp_ar["examples"]),
            " | ".join(hf_en["examples"]),
            " | ".join(hf_ar["examples"])
        ]
    }

    table = pd.DataFrame(data)

    print("\n=== Multilingual NER Comparison ===\n")
    pd.set_option('display.max_colwidth', 60)
    print(table.to_string(index=False))

    table.to_csv("ner_comparison_table.csv", index=False, encoding='utf-8')


    print("\nTable saved successfully!")
    print("   ner_comparison_table.csv")

if __name__ == "__main__":
    main()