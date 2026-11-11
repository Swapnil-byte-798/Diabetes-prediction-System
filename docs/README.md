# Diabetes Prediction System study material

Updated 11 November 2026. The guides cover the repaired Python CLI and notebook, the measured results and how to explain them in interviews.

| Purpose | Markdown | PDF |
|---|---|---|
| Architecture, data and execution | [How it works](how-it-works.md) | [Architecture guide](Diabetes-How-It-Works.pdf) |
| Conversational interview preparation | [Interview questions](interview-questions.md) | [Interview guide](Diabetes-Interview-Questions.pdf) |
| Tools, worked metrics and practice | [Tools primer](tools-primer.md) | [Tools guide](Diabetes-Tools-Primer.pdf) |

Start with the tools primer, then follow the architecture and practise the interview questions. The [execution audit](verification.md) records the original failures and the repaired results.

## Rebuild the PDFs

From the repository root, install `requirements-docs.txt` and run `python scripts/render_study_docs.py`. The renderer uses DejaVu Sans fonts. Set `STUDY_FONT_DIR` if the fonts are outside the default Linux directory.
