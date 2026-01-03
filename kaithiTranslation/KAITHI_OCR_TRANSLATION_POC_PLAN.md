# Kaithi Legal Document OCR & Translation — Quick POC Plan

This plan is derived from `kaithiTranslation/prompt/kaithi_ocr_translation_poc_prompt.md`.

## Objective
Build a minimal local proof-of-concept (POC) that:
- Reads a scanned image of a Kaithi-script document
- Extracts text via OCR
- Transliterates Kaithi → Hindi (Devanagari) using explicit rules
- Translates Hindi → English using an open-source model

This POC validates technical feasibility; accuracy is not expected to be production-grade.

## Inputs
- Supported formats: JPEG/PNG
- Sample documents location:
  - `kaithiTranslation/kaithidocs/`

## Constraints
- Python
- Runs locally
- No paid/cloud APIs
- Modular pipeline:
  - OCR module
  - Transliteration module
  - Translation module
- OCR must preserve Kaithi Unicode when recognized (Kaithi block: U+11080–U+110CF)

---

## Phase 0 — Align scope + confirm samples (0.5 day)
### Deliverables
- 3–10 representative Kaithi document images in `kaithiTranslation/kaithidocs/`
- Agreed POC success definition:
  - End-to-end flow runs locally
  - Low OCR accuracy acceptable
  - Unicode preservation is mandatory

### Acceptance checks
- Samples open and are visually Kaithi documents
- At least 1 sample has clearly visible Kaithi characters

---

## Phase 1 — Local setup + project skeleton (0.5 day)
### Deliverables
- Project skeleton (single script or small module set)
- Windows-friendly dependency/setup instructions
- Minimal CLI:
  - Input image path
  - Output directory
  - OCR engine selection (optional)

### Acceptance checks
- `python ... --help` works
- Running with a sample image reaches the OCR call path

---

## Phase 2 — OCR module (Kaithi recognition) (1–2 days)
### Requirements
- Open-source OCR only. Preferred:
  - Tesseract OCR (Indic or custom-trained)
  - PaddleOCR
  - EasyOCR

### Deliverables
- `ocr` module that:
  - Loads image
  - Optional preprocessing (grayscale / threshold / denoise / deskew)
  - Runs OCR
  - Writes raw OCR output text
- Kaithi Unicode preservation and detection check (U+11080–U+110CF)
- Debug artifacts (recommended):
  - Raw OCR output file
  - Preprocessed image output for comparison

### Acceptance checks
- Runs on at least 3 sample images from `kaithiTranslation/kaithidocs/`
- Output is valid UTF-8 and does not crash/garble
- If Kaithi codepoints appear, they remain present in output

---

## Phase 3 — Transliteration module (Kaithi → Devanagari) (1 day)
### Deliverables
- `transliteration` module:
  - Explicit dictionary mapping Kaithi chars → Devanagari
  - Extensible mapping design
  - Deterministic handling for unmapped characters
- Output: valid Hindi (Devanagari) text

### Acceptance checks
- Known Kaithi test string produces expected Devanagari output
- Unmapped characters do not crash the pipeline

---

## Phase 4 — Translation module (Hindi → English) (1–2 days)
### Deliverables
- `translation` module:
  - Open-source HuggingFace model hi→en
  - Local inference (no paid/cloud APIs)
  - Basic length handling
- Output: English translation text

### Acceptance checks
- After model download, translation runs locally
- Produces an English translation string for sample Hindi input

---

## Phase 5 — End-to-end integration + outputs (0.5–1 day)
### Deliverables
- One command runs full pipeline and prints/logs:
  1. Raw OCR output (Kaithi)
  2. Transliterated Hindi (Devanagari)
  3. English translation
- Consistent output folder structure per input file

### Acceptance checks
- End-to-end works on at least 1 real Kaithi image
- Outputs are clearly separated and traceable to input file

---

## Phase 6 (Optional) — Traceability & future improvements (1–2 days)
### Deliverables
- OCR bounding boxes + confidence (if supported)
- Simple layout hints for legal docs (optional)
- Notes for future improvements:
  - Custom Kaithi OCR training
  - Layout analysis
  - Human-in-the-loop validation

### Acceptance checks
- Basic traceability: can associate text segments back to image regions (where available)

---

## Decisions to make early
- OCR engine preference for Phase 2: Tesseract vs PaddleOCR vs EasyOCR
- Packaging preference: single script vs small module set
