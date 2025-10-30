# LangTest Compatibility Matrix - Complete Guide

## Overview
This guide provides a comprehensive mapping of compatibility between Tasks, Test Categories, Hubs, and Data formats in LangTest. Use this reference to understand which combinations are supported for your NLP model testing needs.

---

## Quick Reference: Task Support Matrix

| Task | Test Categories | Compatible Hubs | Data Formats |
|------|----------------|-----------------|--------------|
| **[NER](https://langtest.org/docs/pages/task/ner)** | Accuracy, Bias, Fairness, Robustness, Representation | John Snow Labs, HuggingFace, SpaCy | CoNLL, CSV, HuggingFace datasets |
| **[Text Classification](https://langtest.org/docs/pages/task/text-classification)** | Accuracy, Bias, Fairness, Robustness, Representation, Grammar | John Snow Labs, HuggingFace, SpaCy, Custom frameworks | CSV, HuggingFace datasets |
| **[Question Answering](https://langtest.org/docs/pages/task/question-answering)** | Accuracy, Bias, Fairness, Representation, Grammar, Robustness, Factuality, Ideology, Legal, Sensitivity, Stereoset, Sycophancy | HuggingFace, HuggingFace API, OpenAI, Cohere, AI21, Azure OpenAI, LM Studio | Benchmark datasets, CSV, HuggingFace datasets |
| **[Summarization](https://langtest.org/docs/pages/task/summarization)** | Accuracy, Bias, Fairness, Robustness, Representation | HuggingFace, HuggingFace API, OpenAI, Cohere, AI21, Azure OpenAI, LM Studio | XSum, MultiLexSum, HuggingFace datasets |
| **[Text Generation](https://langtest.org/docs/pages/task/text-generation)** | Clinical, Disinformation, Security, Toxicity | HuggingFace, HuggingFace API, OpenAI, Cohere, AI21, Azure OpenAI, LM Studio | Specific benchmark datasets per category |
| **[Fill Mask](https://langtest.org/docs/pages/task/fill-mask)** | Stereotype (Wino-Bias, CrowS-Pairs) | HuggingFace | Wino-test, CrowS-Pairs |
| **[Translation](https://langtest.org/docs/pages/task/translation)** | Robustness | John Snow Labs, HuggingFace, SpaCy | Translation benchmark dataset |

---

## Detailed Compatibility Breakdown

### 1. [Named Entity Recognition (NER)](https://langtest.org/docs/pages/task/ner)

#### Supported Test Categories
- ✅ **[Accuracy](https://langtest.org/docs/pages/docs/test_categories#accuracy-tests)**: Precision, Recall, F1 Score, Micro/Macro/Weighted F1
- ✅ **[Bias](https://langtest.org/docs/pages/docs/test_categories#bias-tests)**: Gender, ethnicity, religion, country replacement tests
- ✅ **[Fairness](https://langtest.org/docs/pages/docs/test_categories#fairness-tests)**: Male, female, unknown group evaluations
- ✅ **[Robustness](https://langtest.org/docs/pages/docs/test_categories#robustness-tests)**: Typos, case changes, perturbations
- ✅ **[Representation](https://langtest.org/docs/pages/docs/test_categories#representation-tests)**: Population representation analysis

#### Compatible Hubs
- `johnsnowlabs`
- `huggingface`
- `spacy`

#### Data Formats
- **CoNLL Format**: Standard token-label format
- **CSV Format**: 
  - Text columns: `text`, `sentences`, `sentence`, `sample`
  - NER columns: `label`, `labels`, `class`, `classes`, `ner_tag`, `ner_tags`, `ner`, `entity`
  - POS columns: `pos_tags`, `pos_tag`, `pos`, `part_of_speech`
  - Chunk columns: `chunk_tags`, `chunk_tag`
- **HuggingFace Datasets**: With proper column mapping

#### Example Configuration
```python
harness = Harness(
    task='ner',
    model={'model': 'en_core_web_sm', 'hub':'spacy'},
    data={"data_source":'test.conll'}
)
```

---

### 2. [Text Classification](https://langtest.org/docs/pages/task/text-classification)

#### Supported Test Categories
- ✅ **[Accuracy](https://langtest.org/docs/pages/docs/test_categories#accuracy-tests)**: Precision, Recall, F1 Score, Micro/Macro/Weighted F1
- ✅ **[Bias](https://langtest.org/docs/pages/docs/test_categories#bias-tests)**: Gender, ethnicity, religion, country replacement tests
- ✅ **[Fairness](https://langtest.org/docs/pages/docs/test_categories#fairness-tests)**: Male, female, unknown group evaluations
- ✅ **[Robustness](https://langtest.org/docs/pages/docs/test_categories#robustness-tests)**: Typos, case changes, perturbations
- ✅ **[Representation](https://langtest.org/docs/pages/docs/test_categories#representation-tests)**: Population representation analysis
- ✅ **[Grammar](https://langtest.org/docs/pages/docs/test_categories#grammar-test)**: Paraphrasing tests

#### Compatible Hubs
- `johnsnowlabs`
- `huggingface`
- `spacy`
- Custom frameworks (Keras, PyTorch, TensorFlow, scikit-learn)

#### Data Formats
- **CSV Format**:
  - Text columns: `text`, `sentences`, `sentence`, `sample`
  - Label columns: `label`, `labels`, `class`, `classes`
- **HuggingFace Datasets**: With proper column mapping

#### Example Configuration
```python
harness = Harness(
    task='text-classification',
    model={'model': 'mrm8488/distilroberta-finetuned-tweets-hate-speech', 'hub':'huggingface'},
    data={"data_source":'sample.csv'}
)
```

---

### 3. [Question Answering](https://langtest.org/docs/pages/task/question-answering)

#### Supported Test Categories

**Standard Tests** (with benchmark datasets):
- ✅ **[Accuracy](https://langtest.org/docs/pages/docs/test_categories#accuracy-tests)**: Performance metrics
- ✅ **[Bias](https://langtest.org/docs/pages/docs/test_categories#bias-tests)**: BoolQ dataset with bias split
- ✅ **[Fairness](https://langtest.org/docs/pages/docs/test_categories#fairness-tests)**: Male, female, unknown group evaluations
- ✅ **[Robustness](https://langtest.org/docs/pages/docs/test_categories#robustness-tests)**: Input perturbations
- ✅ **[Representation](https://langtest.org/docs/pages/docs/test_categories#representation-tests)**: Population representation analysis
- ✅ **[Grammar](https://langtest.org/docs/pages/docs/test_categories#grammar-test)**: Paraphrasing questions

**Specialized Tests** (category-specific):
- ✅ **[Factuality](https://langtest.org/docs/pages/docs/test_categories#factuality-test)**: Factual-Summary-Pairs dataset
- ✅ **[Ideology](https://langtest.org/docs/pages/docs/test_categories#ideology-tests)**: Political compass questions (auto-loaded)
- ✅ **[Legal](https://langtest.org/docs/pages/docs/test_categories#legal-tests)**: legal-support dataset
- ✅ **[Sensitivity](https://langtest.org/docs/pages/docs/test_categories#sensitivity-tests)**: 
  - Add Negation: NQ-open, OpenBookQA
  - Add Toxic Words: wikiDataset
- ✅ **[Stereoset](https://langtest.org/docs/pages/docs/test_categories#stereoset-tests)**: StereoSet dataset
- ✅ **[Sycophancy](https://langtest.org/docs/pages/docs/test_categories#sycophancy-tests)**: sycophancy-math-data, synthetic-math-data

#### Compatible Hubs
- `huggingface`
- `huggingface-inference-api`
- `openai`
- `cohere`
- `ai21`
- `azure-openai`
- `lmstudio`

#### Data Formats
- Benchmark datasets (BBQ, BoolQ, etc.)
- CSV files
- HuggingFace datasets
- Task-specific datasets for specialized categories

#### Task Specification
```python
# Standard QA
task = "question-answering"

# Category-specific
task = {"task": "question-answering", "category": "ideology"}
task = {"task": "question-answering", "category": "factuality"}
task = {"task": "question-answering", "category": "legal"}
task = {"task": "question-answering", "category": "sensitivity"}
task = {"task": "question-answering", "category": "stereoset"}
task = {"task": "question-answering", "category": "sycophancy"}
```

---

### 4. [Summarization](https://langtest.org/docs/pages/task/summarization)

#### Supported Test Categories
- ✅ **[Accuracy](https://langtest.org/docs/pages/docs/test_categories#accuracy-tests)**: ROUGE, BERTScore metrics
- ✅ **[Bias](https://langtest.org/docs/pages/docs/test_categories#bias-tests)**: XSum dataset with bias split
- ✅ **[Fairness](https://langtest.org/docs/pages/docs/test_categories#fairness-tests)**: Male, female, unknown group evaluations
- ✅ **[Robustness](https://langtest.org/docs/pages/docs/test_categories#robustness-tests)**: Input perturbations
- ✅ **[Representation](https://langtest.org/docs/pages/docs/test_categories#representation-tests)**: Population representation analysis

#### Compatible Hubs
- `huggingface`
- `huggingface-inference-api`
- `openai`
- `cohere`
- `ai21`
- `azure-openai`
- `lmstudio`

#### Data Formats
- **Benchmark Datasets**: XSum, MultiLexSum
- **HuggingFace Datasets**: With `dialogue` and `summary` columns

#### Example Configuration
```python
harness = Harness(
    task="summarization",
    model={"model": "gpt-3.5-turbo-instruct","hub":"openai"},
    data={"data_source": "XSum", "split":"test-tiny"}
)
```

---

### 5. [Text Generation](https://langtest.org/docs/pages/task/text-generation)

#### Supported Test Categories by Sub-task

| Category | Dataset | Description |
|----------|---------|-------------|
| **[Clinical](https://langtest.org/docs/pages/docs/test_categories#clinical-test)** | Clinical (Medical-files split) | Demographic bias in treatment plans |
| **[Disinformation](https://langtest.org/docs/pages/docs/test_categories#disinformation-test)** | Narrative-Wedging | Disinformation generation capability |
| **[Security](https://langtest.org/docs/pages/docs/test_categories#security-test)** | Prompt-Injection-Attack | Prompt injection vulnerabilities |
| **[Toxicity](https://langtest.org/docs/pages/docs/test_categories#toxicity-tests)** | Toxicity dataset | Ideology, LGBTQphobia, Offensive, Racism, Sexism, Xenophobia |

#### Compatible Hubs
- `huggingface`
- `huggingface-inference-api`
- `openai`
- `cohere`
- `ai21`
- `azure-openai`
- `lmstudio`

#### Task Specification (Always Dictionary)
```python
task = {"task": "text-generation", "category": "clinical"}
task = {"task": "text-generation", "category": "disinformation"}
task = {"task": "text-generation", "category": "security"}
task = {"task": "text-generation", "category": "toxicity"}
```

#### Example Configuration
```python
harness = Harness(
    task={"task":"text-generation", "category":"toxicity"},
    model={"model": "gpt-3.5-turbo-instruct","hub":"openai"},
    data={"data_source": "Toxicity", "split":"test"}
)
```

---

### 6. [Fill Mask](https://langtest.org/docs/pages/task/fill-mask)

#### Supported Test Categories
- ✅ **[Stereotype](https://langtest.org/docs/pages/docs/test_categories#stereotype-tests)**: Wino-Bias and CrowS-Pairs tests

#### Compatible Hubs
- `huggingface` (only masked language models)

#### Data Formats
- **Wino-test**: Wino-Bias dataset
- **CrowS-Pairs**: CrowS-Pairs dataset

#### Task Specification (Always Dictionary)
```python
task = {"task": "fill-mask", "category": "wino-bias"}
task = {"task": "fill-mask", "category": "crows-pairs"}
```

#### Example Configuration
```python
harness = Harness(
    task={"task": "fill-mask", "category": "wino-bias"},
    model={"model": "bert-base-uncased", "hub":"huggingface"},
    data={"data_source":"Wino-test"}
)
```

---

### 7. [Translation](https://langtest.org/docs/pages/task/translation)

#### Supported Test Categories
- ✅ **[Robustness](https://langtest.org/docs/pages/docs/test_categories#robustness-tests)**: Lowercase, uppercase, typos, and other perturbations

#### Compatible Hubs
- `johnsnowlabs`
- `huggingface`
- `spacy`

#### Data Formats
- Translation benchmark dataset

#### Example Configuration
```python
harness = Harness(
    task="translation",
    model={"model":'t5-base', "hub": "huggingface"},
    data={"data_source": "Translation"}
)
```

---

## Hub Compatibility Summary

### Hub Support by Task

| Hub | Supported Tasks |
|-----|----------------|
| **John Snow Labs** | [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Translation](https://langtest.org/docs/pages/task/translation) |
| **HuggingFace** | [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Translation](https://langtest.org/docs/pages/task/translation), [Fill Mask](https://langtest.org/docs/pages/task/fill-mask), [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Text Generation](https://langtest.org/docs/pages/task/text-generation) |
| **HuggingFace API** | [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Fill Mask](https://langtest.org/docs/pages/task/fill-mask), [Text Generation](https://langtest.org/docs/pages/task/text-generation) |
| **SpaCy** | [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Translation](https://langtest.org/docs/pages/task/translation) |
| **OpenAI** | [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Fill Mask](https://langtest.org/docs/pages/task/fill-mask), [Text Generation](https://langtest.org/docs/pages/task/text-generation) |
| **Cohere** | [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Fill Mask](https://langtest.org/docs/pages/task/fill-mask), [Text Generation](https://langtest.org/docs/pages/task/text-generation) |
| **AI21** | [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Fill Mask](https://langtest.org/docs/pages/task/fill-mask), [Text Generation](https://langtest.org/docs/pages/task/text-generation) |
| **Azure OpenAI** | [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Fill Mask](https://langtest.org/docs/pages/task/fill-mask), [Text Generation](https://langtest.org/docs/pages/task/text-generation) |
| **LM Studio** | [Fill Mask](https://langtest.org/docs/pages/task/fill-mask), [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Text Generation](https://langtest.org/docs/pages/task/text-generation) |
| **Custom Frameworks** | [Text Classification](https://langtest.org/docs/pages/task/text-classification) (Keras, PyTorch, TensorFlow, scikit-learn) |

---

## Test Category Descriptions

### [Accuracy](https://langtest.org/docs/pages/docs/test_categories#accuracy-tests)
Evaluates model performance using metrics like precision, recall, F1 score, micro F1, macro F1, and weighted F1.

**Compatible Tasks**: [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization)

---

### [Bias](https://langtest.org/docs/pages/docs/test_categories#bias-tests)
Tests how replacing names (gender, ethnicity, religion, country) affects model predictions.

**Compatible Tasks**: [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization)

**Note**: For Question Answering and Summarization, only BoolQ dataset with `bias` split is supported.

---

### [Fairness](https://langtest.org/docs/pages/docs/test_categories#fairness-tests)
Evaluates if the model treats different demographic groups (male, female, unknown) equally.

**Compatible Tasks**: [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization)

---

### [Robustness](https://langtest.org/docs/pages/docs/test_categories#robustness-tests)
Tests model consistency when input is perturbed (typos, case changes, abbreviations).

**Compatible Tasks**: [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Translation](https://langtest.org/docs/pages/task/translation)

---

### [Representation](https://langtest.org/docs/pages/docs/test_categories#representation-tests)
Assesses if the dataset accurately represents specific populations.

**Compatible Tasks**: [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization)

---

### [Grammar](https://langtest.org/docs/pages/docs/test_categories#grammar-test)
Tests model's ability to handle paraphrased or grammatically varied inputs.

**Compatible Tasks**: [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Question Answering](https://langtest.org/docs/pages/task/question-answering)

---

### [Factuality](https://langtest.org/docs/pages/docs/test_categories#factuality-test)
Evaluates ability to identify factual accuracy in summaries.

**Compatible Tasks**: [Question Answering](https://langtest.org/docs/pages/task/question-answering) only
**Dataset**: Factual-Summary-Pairs

---

### [Ideology](https://langtest.org/docs/pages/docs/test_categories#ideology-tests)
Measures political orientation on economic and social dimensions.

**Compatible Tasks**: [Question Answering](https://langtest.org/docs/pages/task/question-answering) only
**Dataset**: Auto-loaded political compass questions

---

### [Legal](https://langtest.org/docs/pages/docs/test_categories#legal-tests)
Assesses legal reasoning and comprehension capabilities.

**Compatible Tasks**: [Question Answering](https://langtest.org/docs/pages/task/question-answering) only
**Dataset**: legal-support

---

### [Sensitivity](https://langtest.org/docs/pages/docs/test_categories#sensitivity-tests)
Tests model responsiveness to negations and toxic words.

**Compatible Tasks**: [Question Answering](https://langtest.org/docs/pages/task/question-answering) only
**Datasets**: wikiDataset, NQ-open, OpenBookQA

---

### [Stereotype](https://langtest.org/docs/pages/docs/test_categories#stereotype-tests)
Evaluates gender and occupational stereotypes in model predictions.

**Compatible Tasks**: [Fill Mask](https://langtest.org/docs/pages/task/fill-mask) only
**Datasets**: Wino-test, CrowS-Pairs

---

### [Stereoset](https://langtest.org/docs/pages/docs/test_categories#stereoset-tests)
Comprehensive bias assessment using stereotypic vs anti-stereotypic sentence pairs.

**Compatible Tasks**: [Question Answering](https://langtest.org/docs/pages/task/question-answering) only
**Dataset**: StereoSet

---

### [Sycophancy](https://langtest.org/docs/pages/docs/test_categories#sycophancy-tests)
Tests if model tailors responses to align with user views incorrectly.

**Compatible Tasks**: [Question Answering](https://langtest.org/docs/pages/task/question-answering) only
**Datasets**: sycophancy-math-data, synthetic-math-data

---

### [Clinical](https://langtest.org/docs/pages/docs/test_categories#clinical-test)
Evaluates demographic bias in medical treatment recommendations.

**Compatible Tasks**: [Text Generation](https://langtest.org/docs/pages/task/text-generation) only
**Dataset**: Clinical (Medical-files split)

---

### [Disinformation](https://langtest.org/docs/pages/docs/test_categories#disinformation-test)
Tests model's capacity to generate disinformation content.

**Compatible Tasks**: [Text Generation](https://langtest.org/docs/pages/task/text-generation) only
**Dataset**: Narrative-Wedging

---

### [Security](https://langtest.org/docs/pages/docs/test_categories#security-test)
Assesses resilience against prompt injection attacks.

**Compatible Tasks**: [Text Generation](https://langtest.org/docs/pages/task/text-generation) only
**Dataset**: Prompt-Injection-Attack

---

### [Toxicity](https://langtest.org/docs/pages/docs/test_categories#toxicity-tests)
Evaluates toxic content across multiple dimensions (ideology, racism, sexism, etc.).

**Compatible Tasks**: [Text Generation](https://langtest.org/docs/pages/task/text-generation) only
**Dataset**: Toxicity

---

## Important Incompatibilities

### ❌ Common Incompatible Combinations

1. **[Grammar Test](https://langtest.org/docs/pages/docs/test_categories#grammar-test)** 
   - ❌ NOT available for: [NER](https://langtest.org/docs/pages/task/ner), [Summarization](https://langtest.org/docs/pages/task/summarization), [Fill Mask](https://langtest.org/docs/pages/task/fill-mask), [Translation](https://langtest.org/docs/pages/task/translation), [Text Generation](https://langtest.org/docs/pages/task/text-generation)
   - ✅ Only for: [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Question Answering](https://langtest.org/docs/pages/task/question-answering)

2. **Specialized QA Categories**
   - [Factuality](https://langtest.org/docs/pages/docs/test_categories#factuality-test), [Ideology](https://langtest.org/docs/pages/docs/test_categories#ideology-tests), [Legal](https://langtest.org/docs/pages/docs/test_categories#legal-tests), [Sensitivity](https://langtest.org/docs/pages/docs/test_categories#sensitivity-tests), [Stereoset](https://langtest.org/docs/pages/docs/test_categories#stereoset-tests), [Sycophancy](https://langtest.org/docs/pages/docs/test_categories#sycophancy-tests)
   - ❌ NOT available for any task except [Question Answering](https://langtest.org/docs/pages/task/question-answering)

3. **[Stereotype Test](https://langtest.org/docs/pages/docs/test_categories#stereotype-tests)**
   - ❌ NOT available for: [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Translation](https://langtest.org/docs/pages/task/translation), [Text Generation](https://langtest.org/docs/pages/task/text-generation)
   - ✅ Only for: [Fill Mask](https://langtest.org/docs/pages/task/fill-mask)

4. **Text Generation Categories**
   - [Clinical](https://langtest.org/docs/pages/docs/test_categories#clinical-test), [Disinformation](https://langtest.org/docs/pages/docs/test_categories#disinformation-test), [Security](https://langtest.org/docs/pages/docs/test_categories#security-test), [Toxicity](https://langtest.org/docs/pages/docs/test_categories#toxicity-tests)
   - ❌ NOT available for any task except [Text Generation](https://langtest.org/docs/pages/task/text-generation)

5. **Hub Restrictions**
   - John Snow Labs, SpaCy: ❌ NOT for [Question Answering](https://langtest.org/docs/pages/task/question-answering), [Summarization](https://langtest.org/docs/pages/task/summarization), [Fill Mask](https://langtest.org/docs/pages/task/fill-mask), [Text Generation](https://langtest.org/docs/pages/task/text-generation)
   - OpenAI, Cohere, AI21, Azure OpenAI, LM Studio: ❌ NOT for [NER](https://langtest.org/docs/pages/task/ner), [Text Classification](https://langtest.org/docs/pages/task/text-classification), [Translation](https://langtest.org/docs/pages/task/translation)

6. **Dataset Restrictions**
   - CoNLL format: ✅ Only for [NER](https://langtest.org/docs/pages/task/ner)
   - Wino-test, CrowS-Pairs: ✅ Only for [Fill Mask](https://langtest.org/docs/pages/task/fill-mask)
   - Clinical, Prompt-Injection-Attack, Toxicity: ✅ Only for [Text Generation](https://langtest.org/docs/pages/task/text-generation)
   - BoolQ with bias split: ✅ Only for [Question Answering](https://langtest.org/docs/pages/task/question-answering) and [Summarization](https://langtest.org/docs/pages/task/summarization) bias tests

---