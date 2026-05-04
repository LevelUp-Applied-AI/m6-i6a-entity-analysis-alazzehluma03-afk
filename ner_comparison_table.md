# Multilingual NER Comparison Table

**Run Date:** 2026-05-02

| Language | Model   | Total | Density | Zero Rate | Entity Types                  | Examples |
|----------|---------|-------|---------|-----------|-------------------------------|----------|
| English  | spaCy   | 81    | 6.67    | 0.0       | LOC:37, ORG:21, MISC:19, PER:4 | British Antarctic Survey(ORG) \| Antarctic(LOC) \| Antarctic(MISC) |
| Arabic   | spaCy   | 16    | 1.7     | 0.4       | PER:9, LOC:3, MISC:2, ORG:2   | ويعاني الحوض(PER) \| وأفادت دائرة(PER) \| ويرتبط(PER) |
| English  | HF      | 88    | 7.25    | 0.0       | LOC:45, ORG:42, PER:1         | British Antarctic Survey(ORG) \| Antarctic sea ice(LOC) \| Antarctic ice dynamic(LOC) |
| Arabic   | HF      | 64    | 6.8     | 0.0       | ORG:35, LOC:29                | جامعة الأردن(ORG) \| حوض عمّان-الزرقاء(LOC) \| دائرة الأرصاد الجوية الأردنية(ORG) |

**Notes:**
- HF model performs significantly better on Arabic text.
- spaCy struggles with Arabic (low entity count).