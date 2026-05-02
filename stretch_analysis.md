# Stretch: Multilingual NER Comparison

## (a) What entity types are harder in Arabic vs English, and why?

Multilingual NER models showed noticeably different behavior across English and Arabic texts. In English articles, both models consistently detected entities such as “UN”, “EU”, and “Paris Agreement” as organizations. However, in Arabic texts, equivalent entities like "الأمم المتحدة" and "اتفاقية باريس" were either missed or inconsistently labeled, particularly by the spaCy model. This difference is largely due to the lack of capitalization in Arabic, which removes an important signal used in English NER. Additionally, Arabic’s morphological complexity—where prefixes and suffixes attach directly to words—makes entity boundaries harder to detect, leading to partial or fragmented entity recognition.

## (b) What this means for bilingual NLP applications in the MENA region

These findings highlight a critical challenge for NLP systems in the MENA region. A pipeline that performs well on English climate reports may fail to extract key entities from Arabic news articles, leading to incomplete or biased downstream analysis. For example, missing organization or location entities in Arabic could affect climate policy tracking or information extraction systems. To build robust bilingual applications in Jordan and similar contexts, developers must either fine-tune multilingual models on Arabic-specific data or incorporate preprocessing techniques tailored to Arabic text.

**Conclusion:** Multilingual capabilities are not optional but essential for effective NLP solutions in the MENA region.