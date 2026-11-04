# Sources

Six PDF files in `data/raw/` (gitignored). `python -m rag_ai_act fetch` checks each file against the SHA-256 below,
downloads the Commission files that are missing, and prints a manual download line for the EUR-Lex files
(EUR-Lex answers scripted requests with a bot check). All files were retrieved on 2026-11-03.

| File | Role | URL | Retrieved | SHA-256 | Licence |
|---|---|---|---|---|---|
| `CELEX_02024R1689-20260727_EN_TXT.pdf` | indexed: AI Act, consolidated text of 2026-07-27 | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:02024R1689-20260727 | 2026-11-03 | `1ccd38d1c78482cf2053b70110115adcc5080700acc8bd7bead8cb3579143ccf` | EUR-Lex reuse, Commission Decision 2011/833/EU |
| `OJ_L_202401689_EN_TXT.pdf` | not indexed, used only to write changed-facts questions: AI Act as originally published in the Official Journal (2024) | https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689 | 2026-11-03 | `bba630444b3278e881066774002a1d7824308934f49ccfa203e65be43692f55e` | EUR-Lex reuse, Commission Decision 2011/833/EU |
| `Draft_Guidelines_on_the_classification_on_highrisk_AI_general_principles_UQmTa2BdJ3IH8bIsWJyDqaWOY_128559.pdf` | indexed: draft high-risk guidelines, general principles | https://ec.europa.eu/newsroom/dae/redirection/document/128559 | 2026-11-03 | `b127bbdc50b1741bb2d97e8aff5839cccd1a4484445be4e8cca246c12541fc42` | CC BY 4.0 (European Commission) |
| `Draft_Guidelines_on_the_classification_of_high_risk_AI_Annex_I_E9XHEqRLx555OO6iijKwJF32Cvc_128560.pdf` | indexed: draft high-risk guidelines, Annex I part | https://ec.europa.eu/newsroom/dae/redirection/document/128560 | 2026-11-03 | `10f1302c9090d2bcfdb8eacd2b36ff7a09860fc7909f7c1c4b42cc3b9bed3b50` | CC BY 4.0 (European Commission) |
| `Draft_Guidelines_on_the_classification_of_high_risk_AI_Annex_III_7MXR3YIz2GW3uPpJPWvvNDd8IOI_128561.pdf` | indexed: draft high-risk guidelines, Annex III part | https://ec.europa.eu/newsroom/dae/redirection/document/128561 | 2026-11-03 | `b1df0ffb30310e126c7e060e03c9b5aab97c0a2ab61a2f3e3e00ede3655e2792` | CC BY 4.0 (European Commission) |
| `Commission_Guidelines_on_the_definition_of_an_artificial_intelligence_system_established_by_Regulation_EU_20241689_AI_ActEnglish_nf2skCqfRtJdFgGJAvCODoPCWZ4_112455.PDF` | indexed: Commission guidelines on the definition of an AI system | https://ec.europa.eu/newsroom/dae/redirection/document/112455 | 2026-11-03 | `fe39f41d061184a913c32f1f92aaaa30a096858fa168c6053e22a19eef58910e` | CC BY 4.0 (European Commission) |

The guidelines on high-risk classification are drafts: they are indexed as such and every chunk from them is flagged as a draft.
