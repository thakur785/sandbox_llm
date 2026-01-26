## Phase 0 – OCR Feasibility Using Devanagari as a Proxy for Kaithi

### Objective

The goal of Phase 0 was to validate whether historical legal documents written in the **Kaithi script** can be digitized in a practical and scalable way, despite the absence of an official Kaithi OCR model.

Rather than targeting perfect character accuracy, this phase focused on answering a narrower but critical question:

> *Can we reliably extract structured, line-preserving text from Kaithi documents that is suitable for downstream AI-based recovery and translation?*

---

### Constraints and Technical Reality

* Kaithi is a **historical Brahmi-derived script** with **no official OCR language pack** in Tesseract or other mainstream OCR engines.
* Training a custom Kaithi OCR model would require significant time, curated datasets, and manual ground-truthing, which is unsuitable for a rapid POC.
* Therefore, a **proxy OCR strategy** was required.

---

### Approach

* **OCR Engine** : Tesseract OCR
* **Strategy** : Use the **Devanagari script model** as a structural proxy for Kaithi
* **Rationale** :
* Kaithi and Devanagari share historical and visual similarities
* Devanagari OCR can often capture  **layout, line structure, word boundaries, and approximate phonetics** , even when glyphs are incorrect
* **Preprocessing** : Minimal grayscale normalization to preserve original glyph shapes
* **Environment** : Google Colab, with code and data sourced directly from GitHub

---

### Outcome

* OCR execution completed successfully on scanned Kaithi legal documents
* The extracted text is  **noisy at the character level** , but:
  * Preserves **paragraph and line structure**
  * Retains **numbers, punctuation, and legal formatting patterns**
  * Produces consistent, interpretable output rather than random noise

This result confirms that  **structural text extraction from Kaithi documents is feasible** , which is sufficient for subsequent AI-based normalization and translation.

---

### Key Takeaway

Phase 0 demonstrates that even without a native Kaithi OCR model, it is possible to extract usable textual structure from historical Kaithi documents using a proxy OCR approach. This validates the core feasibility of the system and enables the next phase, where **LLM-based recovery and translation** will be applied to convert noisy OCR output into clean Hindi and English text.
