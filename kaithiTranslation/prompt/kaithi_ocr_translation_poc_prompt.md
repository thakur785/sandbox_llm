# Kaithi Legal Document OCR & Translation – Copilot Prompt

## Objective
Build a **quick proof-of-concept (POC)** application that reads **historical Indian legal documents written in Kaithi script**, extracts text, transliterates it into Hindi (Devanagari), and translates it into English.

This POC is meant to validate **what works technically** rather than achieve production-grade accuracy.

---

## Background Context (Critical)

- **Kaithi is a historical script**, not a spoken language.
- It was historically used to write **Bhojpuri, Awadhi, and Maithili**, especially in land records and legal documents.
- Unicode block for Kaithi: **U+11080–U+110CF**
- Documents are typically:
  - Old
  - Noisy scans
  - Handwritten or degraded prints
- Legal use cases require **traceability and accuracy**.

---

## Task for Copilot

>You are a senior AI engineer tasked with building a **minimal, local, end-to-end POC** for digitizing Kaithi legal documents.

---

## Functional Requirements

### 1. Input
- Accept a scanned image (JPEG/PNG) of a Kaithi document.

### 2. Step 1 – OCR (Kaithi Script Recognition)
- Use open-source OCR tools only.
- Prefer one of the following:
  - Tesseract OCR (Indic or custom-trained models)
  - PaddleOCR
  - EasyOCR
- The output **must preserve Kaithi Unicode characters** if recognized.
- Even low OCR accuracy is acceptable for this POC.

### 3. Step 2 – Script Transliteration (Kaithi → Devanagari)
- Implement a **rule-based transliteration layer**.
- Use an explicit dictionary mapping Kaithi characters to Devanagari equivalents.
- Keep the design extensible for adding more characters later.
- Output valid **Hindi text in Devanagari script**.

### 4. Step 3 – Translation (Hindi → English)
- Translate Devanagari Hindi into English.
- Use open-source NLP models (e.g., HuggingFace Transformers).
- No paid or cloud-based APIs.

### 5. Output
Print or log the following clearly:
1. Raw OCR output (Kaithi script)
2. Transliterated Hindi (Devanagari)
3. English translation

---

## Technical Constraints

- Programming Language: **Python**
- Must run locally
- No UI required (CLI is sufficient)
- Modular pipeline:
  - OCR Module
  - Transliteration Module
  - Translation Module

---

## Deliverables

1. A **single runnable Python script** or small module set
2. Clear inline comments explaining:
   - Kaithi OCR limitations
   - Transliteration logic
3. Simple instructions to run the POC locally

---

## Bonus (Optional, If Time Permits)

- Suggestions for improving accuracy using:
  - Custom Kaithi OCR model training
  - Layout analysis for legal documents
  - Human-in-the-loop validation workflow
  - Confidence scoring for legal usage

---

## Key Success Criteria

- End-to-end flow works on a real Kaithi document image
- Kaithi is treated as a **script**, not a language
- Transliteration is explicit and auditable
- Code is understandable and extensible

---

**Start with a simple working pipeline, even if OCR quality is poor. Accuracy can be improved iteratively.**

