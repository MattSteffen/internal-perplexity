# internal-perplexity

A great chat interface over a variety of information sources. Designed to be fast, accurate, and comprehensive.

Basic Components:
- [ ] Chat Interface (through open-webui)
- [ ] RAG pipeline
  - personal data can be shared on open-webui
  - general data can be indexed by the crawler, and accessed through an open-webui pipeline
  - documents can be uploaded through a distinct interface
  - RBAC enforced on general data, linked through ldap groups
  - should be able to:
    - document level questions
      - find the correct document you're looking for
      - explain concepts from the document you're looking for
      - give citations for the information you're using
    - answer database level questions
      - like "what are the most common topics in the database?"
      - how many documents discuss a particular topic?
      - how many or what documents are written by a particular author?
- [ ] Crawler
  - processes pdfs and uploads to the vector database
  - extracts metadata following a json schema
  - [ ] See https://github.com/huggingface/finepdfs for pdf processing
  - [ ] try out deepseek-ocr
    ```python
    import ollama
    import fitz  # PyMuPDF for PDF handling
    
    def pdf_to_md(pdf_path):
        doc = fitz.open(pdf_path)
        md_pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img_path = f"page_{page.number}.png"
            pix.save(img_path)
            response = ollama.generate(model="deepseek-ocr", prompt=f"<|grounding|>Convert the document to markdown.", images=[img_path])
    
            md_pages.append(response['response'])
        return '\n\n---\n\n'.join(md_pages)  # Concatenate pages
    pdf_to_md("path/to/pdf.pdf")
    ```
- context generator
  - wiki: given a set of documents, generate a wiki-like structure
  - fine-tune: given a set of documents, generate a dataset to fine-tune a model

