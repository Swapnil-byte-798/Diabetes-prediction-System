# Diabetes Prediction System study material

Updated 29 October 2026. These notes describe the notebook code reviewed at `f2786e3`, its reproduced metrics and the unresolved problems. This documentation update does not repair or replace the notebook.

| Read this for | Markdown | PDF |
|---|---|---|
| Actual workflow, setup issues and measured results | [How it works](how-it-works.md) | [Architecture guide](Diabetes-How-It-Works.pdf) |
| Interview explanations, metrics and limitations | [Interview questions](interview-questions.md) | [Interview Q&A](Diabetes-Interview-Questions.pdf) |
| Python/ML fundamentals, analogies and study order | [Tool primer](tools-primer.md) | [Tool primer PDF](Diabetes-Tools-Primer.pdf) |

Start with the primer, follow the actual notebook in the architecture guide, then practise the questions. The compatible audit reproduced 77.27% held-out accuracy and 51.85% positive recall. It found correct training-only scaling but unresolved packaging, missing-measurement and input-validation issues.

The notebook's CSV filename does not match the committed dataset, and current pandas rejects its redundant drop arguments. Audit-only cross-validation, split replays and proposed repairs are explicitly labelled. The unrelated plant-image notebook is not part of the diabetes baseline and its stored training plots are not held-out diabetes evidence.

## Rebuild the PDFs

Install the separate documentation dependencies with `python -m pip install -r requirements-docs.txt`, then run `python scripts/render_study_docs.py` from the repository. The renderer reads these Markdown files, builds all three PDFs and checks horizontal text bounds. It uses DejaVu Sans fonts; set `STUDY_FONT_DIR` to the directory containing those fonts if they are not in the default Linux font location. App/model dependencies are unchanged.
