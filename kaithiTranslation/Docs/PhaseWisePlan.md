## POC Phase Overview

### **Phase 0 – OCR Feasibility (Completed)**

* Use Devanagari OCR as a proxy for Kaithi
* Extract text while preserving layout and structure
* Validate that noisy but interpretable OCR output is achievable
* Save OCR output as input for downstream processing

---

### **Phase 1 – LLM-Based Text Recovery**

* Ingest noisy OCR output from Phase 0
* Provide Kaithi and historical legal context to the LLM
* Normalize text into modern Hindi (Devanagari)
* Translate normalized text into English
* Preserve legal meaning, numbers, names, and references

---

### **Phase 2 – OCR Quality Improvement (Optional/Future)**

* Replace proxy OCR with a more robust engine (e.g., PaddleOCR)
* Evaluate Indic OCR or community-trained models
* Assess feasibility of training a custom Kaithi OCR model
* Improve character-level accuracy for production readiness

---

### **Phase 3 – End-to-End Pipeline & Evaluation (Future)**

* Integrate OCR, recovery, and translation into a single pipeline
* Add confidence scoring and error annotations
* Validate output with domain/legal experts
* Prepare for archival, search, or legal reuse use cases
