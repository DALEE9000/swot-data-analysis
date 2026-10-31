# SSV preprint: style and formatting checklist

Compiled 2026-10-16 for the SWOT + HFR machine-learned sea surface velocity paper (SWOTxAI).
Sources: AGU author resources (text requirements, Open Research guidance, agutex template),
AMS formatting and manuscript components page, four exemplar papers in `papers/`
(Kachelein 2024 JGR Oceans, Jensen 2025 GRL, Martin 2024 GRL, Martin 2025 JAMES, Tchonang 2026 ESS),
the ml-ssv literature review (`../ml-ssv-literature-review/`), and an oceanscout `ask` on
community conventions (section 15). Official AGU pages block scrapers; numbers marked (verify)
came from third-party restatements of the Feb 2025 AGU manuscript document. Re-check on agu.org
before submission.

Use as a tick list. Every item is a thing to do, not an idea.

---

## 0. Pick the journal first (it fixes the length, abstract, and front matter)

| Journal | Publisher | Length limit | Abstract | Front matter | Fit for this paper |
|---|---|---|---|---|---|
| **JGR: Oceans** | AGU | 25 publication units (PU) (verify); 1 PU = 500 words or 1 figure or 1 table | 250 words | Key Points (1 to 3, 140 chars each), Plain Language Summary (PLS) | Default home. Full method + regional validation + discussion fits. |
| **GRL** | AGU | 12 PU, hard; returned unread if over | 150 words | Key Points + PLS required | Only if there is one sharp headline result (for example "SWOT ML velocities match HFR to X cm/s out to Y km from coast"). Everything else goes to Supporting Information (SI). |
| **JAMES** | AGU | no strict cap; long methods papers common (Martin 2025: 33 pp, 16 figs) | 250 words | Key Points + PLS | If the contribution is mainly the ML/data-assimilation method and the sim-to-real / evolve story. |
| **ESS** (Earth and Space Science) | AGU | as JGR | 250 | Key Points + PLS | Where Tchonang 2026 put the SWOT geostrophic validation. Good for a validation-first framing. |
| **JPO** | AMS | 7,500 words (body + acknowledgments + appendices; excludes abstract, captions, refs, tables, figures) | 250 words, no citations/URLs | Significance Statement optional, 120 words max | Only if the paper is about dynamics (where and why geostrophy breaks), not about a product. |
| **JTECH** | AMS | 7,500 words | 250 | Significance Statement optional | Validation methodology / observing-system framing. |
| **AIES** | AMS | 7,500 | 250 | Significance Statement | ML-forward framing; smaller audience. |

- [ ] Decide JGR Oceans vs GRL before drafting. The two lead to different papers, not different trims.
- [ ] If GRL: count PU = words/500 + figures + tables. Words include abstract, main text, in-text citations, captions, appendices. Excluded: title, authors, PLS, table text, Open Research, references, SI.
- [ ] If AMS: 12 pt black font, 1.5 line spacing everywhere including references, 1 inch margins, line numbers, page numbers, paragraph breaks visible (indent or blank line).
- [ ] Use the publisher template from day one (agutex on Overleaf for AGU; AMS LaTeX template for AMS). Converting later is where formatting errors creep in.

## 1. Title

- [ ] Specific, informative, brief. State the object and the result, not the method alone. Pattern used by the exemplars: "[Instrument/product] [verb: unveils / improves / resolves] [phenomenon] [region]". Example shape: "Machine-learned total surface currents from SWOT altimetry validated against HF radar on the US West Coast".
- [ ] No acronyms in the title unless defined in the abstract (SWOT is fine; HFR, SSV, MLP are not).
- [ ] Title case for AGU (capitalize major words; articles/prepositions of three letters or fewer lowercase). AMS: sentence case in the template.
- [ ] Avoid "A machine learning approach to ...", "Towards ...", "Novel". Reviewers read these as weak.

## 2. Key Points (AGU only)

- [ ] Exactly 1 to 3. Each is a complete sentence, 140 characters or fewer, no special characters, no punctuation inside, no citations.
- [ ] Each one carries a number or a concrete claim. Exemplar: "SWOT reveals a 20-km anticyclonic eddy in agreement with sea surface temperature and optical observations".
- [ ] Point 1: what you did. Point 2: the headline quantitative result. Point 3: why it matters or where it fails.
- [ ] Editors at GRL screen Key Points and abstract for cross-discipline reach, immediacy, and quantitative significance. Write them for that screen.

## 3. Abstract

- [ ] 250 words (JGR, JAMES, ESS, AMS) or 150 words (GRL). One paragraph.
- [ ] Order: problem, what is new (data or method), main quantitative result with numbers and units, independent-validation statement, uncertainties and limits.
- [ ] No citations, footnotes, URLs, or undefined acronyms. Define SWOT, HFR, and anything else at first use inside the abstract.
- [ ] Give at least one error number with units (cm s^-1, RMSE, correlation, skill vs baseline) and one scale number (km, days, distance from coast).
- [ ] AMS: the abstract in the manuscript must match the abstract typed into the submission system character for character.

## 4. Plain Language Summary (AGU) / Significance Statement (AMS)

- [ ] AGU PLS: required for GRL and JGR at submission. Non-specialist audience (journalists, science-interested public). Exemplar length: 120 to 200 words. No jargon, no acronyms, no numbers that need context.
- [ ] Shape used by Martin 2024 and Jensen 2025: one sentence on why ocean currents matter, one on what the satellite / radar can and cannot see, one on what you did, one on what it enables.
- [ ] AMS Significance Statement: 120 words max, single paragraph, answers three questions in order: purpose, key findings and why they matter, suggested follow-up (optional). Not a plain-language copy of the abstract.

## 5. Section skeleton (matches the community exemplars)

AGU template order: Abstract, PLS, body, Appendices (optional), Open Research, Inclusion in Global Research Statement (if applicable), Conflict of Interest, Acknowledgments, References.
AMS order: Title page, Abstract, Significance Statement, Body, Tables, Figures, Acknowledgments, Availability Statement, Appendix, References.

Body skeleton that every exemplar in `papers/` follows:

```
1. Introduction
2. Data                       (one subsection per source: SWOT L3, HFR, drifters, ERA5, GOES SST)
3. Methods
   3.1 Colocation and target construction (25 h HFR mean, regridding, depth matching, detiding)
   3.2 Baselines (geostrophic, cyclogeostrophic, Ekman-corrected, DUACS/OI)
   3.3 Model (features, architecture, loss, training, hyperparameters)
   3.4 Evaluation protocol (temporal holdout, held-out regions, metrics)
4. Results
   4.1 Skill vs baselines
   4.2 Skill as a function of scale / distance from coast / season
   4.3 Where it fails (internal tides, low-energy regions, data gaps)
5. Discussion
6. Conclusions
Appendix A  (hyperparameters, extra tables)
Open Research / Availability Statement
Conflict of Interest
Acknowledgments
References
Supporting Information (separate file)
```

- [ ] Headings are sentence fragments, numbered, balanced (no 3.1 without 3.2). AGU: Title Case. AMS: `1. Introduction`, `a. Data`, `1) RADIOSONDE`, `(i) Experiment 1`.
- [ ] No bulleted lists in AGU body text. Enumerated lists are allowed.
- [ ] Every figure and table cited in the text, in numerical order, placed near first mention.
- [ ] Methods must be reproducible from the text plus the archived code. A reader should be able to rebuild the training matrix from the Data and Methods sections alone.

## 6. Introduction conventions

- [ ] Paragraph 1: the physical question (total surface velocity at km scale in coastal zones; SSH gives only the geostrophic part; geostrophy weakest on the shelf).
- [ ] Paragraph 2: what SWOT changed (120 km swath, 20 km nadir gap, 15 to 30 km target resolution). Cite Morrow 2019, Fu 2024.
- [ ] Paragraph 3: state of velocity validation (drifters and moorings: Tchonang 2026, Tranchant 2025, Zhou 2025) and the gap: no peer-reviewed SWOT-vs-HFR velocity assessment, no ML fusion using HFR as truth. Tchonang et al. call for HFR validation explicitly. Cite that sentence.
- [ ] Paragraph 4: ML mapping lineage (Manucharyan 2021, Martin 2023/2024/2025, 4DVarNet, Sinha 2024, Ubelmann 2025) and the sim-to-real problem.
- [ ] Final paragraph: "In this paper we ..." with the three concrete contributions, then a one-sentence roadmap of sections. Martin 2025 uses a "1.2 Our Contributions" subsection; JGR papers do it in prose.
- [ ] Every claim about prior work cites the paper, not a review. Do not cite your own literature-review notes.

## 7. Data section conventions

- [ ] SWOT: product name and version exactly as on PO.DAAC (for example "SWOT Level-3 KaRIn Low Rate SSH Expert, v2.0.1", or the reproc v3 name used in `model-data.md`), orbit phase (1-day cal/val cycles 474 to 578; 21-day science cycles 1 to 16), variables used (`ssha_filtered`, `mdt`, `ugos`, `vgos`), the filter applied and its scale, and the DOI. Cite as a [Dataset] reference.
- [ ] HFR: network (IOOS national HFR network via NDBC THREDDS or HFRNet at Scripps), resolution (2 km / 6 km), the 25 h averaging and why (removes diurnal and semidiurnal tides and inertial motions at these latitudes), effective measurement depth (frequency-dependent, roughly 0.5 to 2 m), quality flags, coverage gaps. Cite the data DOI and the method papers (Kirincich, Roarty).
- [ ] Drifters, if used: GDP drogued vs undrogued, 15 m drogue depth, slip corrections (Lumpkin 2013).
- [ ] ERA5: 10 m winds, hourly, version/DOI (Hersbach 2020 + Copernicus CDS DOI).
- [ ] GOES SST: product, L2/L3, cloud mask; note glna winter ice/cloud contamination.
- [ ] A table listing every region (uswc, usegc, gak, glna, prvi, ushi) with lat/lon box, HFR resolution, number of SWOT passes, number of colocated samples, mission(s), and which are trainable. Reviewers ask for this first.
- [ ] A map figure of all regions with SWOT swaths overlaid on HFR coverage (Figure 1 in nearly every exemplar).
- [ ] State time ranges with explicit dates in ISO format (2023-03-30 to 2023-07-10), all times UTC.

## 8. Methods: what reviewers in this community check

- [ ] **Train/validation/test split is temporal, not random.** State it in one sentence and explain why (the research log already documents the random-split leakage). A random split on autocorrelated fields is an automatic reject from an ML-literate reviewer.
- [ ] Held-out regions: report skill on a region never trained on, separately from in-region temporal holdout.
- [ ] **Baselines, in increasing strength:** SWOT geostrophic (`ugos`/`vgos`), cyclogeostrophic, geostrophic + Ekman (Ubelmann 2025 style, or classical 2 to 3 percent of wind speed at 15 to 25 degrees), DUACS/MIOST gridded currents, multiple linear regression (Sinha 2024 did exactly this ladder). The ML number means nothing without the ladder.
- [ ] Define the target precisely: what "surface velocity" means for HFR (effective depth, 25 h mean) vs what SWOT geostrophy represents vs what a 15 m drogue measures. One paragraph on representativeness error (Gap 5 of the lit review). Cite Wei 2020, Lund 2018, Chang 2026.
- [ ] Tides and inertial motions: how they were removed from the target and whether the SWOT internal-tide correction was applied. In tide hotspots SWOT-derived velocities have errors above 15 cm s^-1 in 10 percent of cases; say what you did about it.
- [ ] Model description: features table (name, source, units, any transform), architecture (layers, widths, activation), loss (masked MSE on u and v jointly), optimizer, learning rate, epochs, early stopping, number of parameters, training time and hardware. Random forest: number of trees, max depth, min samples leaf. Hyperparameter search method. All of this can live in Appendix A, but it must exist.
- [ ] If the AlphaEvolve-derived architecture is used: describe it as a search procedure in Methods, report the final architecture as a normal model, and put the evolution history in SI. Do not make the LLM-search the headline unless the paper is going to JAMES or AIES.
- [ ] Feature ablations (no SST, no wind, SSH only) as a table.
- [ ] Define every metric with an equation: RMSE, bias, vector correlation (complex correlation, Kundu 1976), skill score relative to baseline, explained variance, wavenumber spectrum, coherence. Say whether metrics are per-component or vector.
- [ ] Uncertainty: at minimum, bootstrap or ensemble confidence intervals on every headline number. Better: per-pixel uncertainty map and a calibration plot (predicted spread vs observed error). The literature review flags UQ as almost universally absent; including it is a differentiator.

## 9. Results and validation presentation

- [ ] Headline numbers in a single summary table: rows = regions and missions, columns = baselines and model, cells = RMSE (cm s^-1) and correlation, with 95 percent intervals. Mark the best per row.
- [ ] Skill vs scale: wavenumber spectra of u, v, or kinetic energy for HFR, SWOT geostrophic, and model, on the same axes, with the noise floor marked. Report the wavelength where model coherence with HFR drops below 0.5. This is the number the SWOT community compares (18 km ACC, 25 km Mediterranean, 70 km California Current).
- [ ] Skill vs distance from coast and vs bathymetry: a line plot with confidence bands. Coastal is the selling point; show where it degrades.
- [ ] Skill vs season or month if the record allows.
- [ ] A Taylor diagram or target diagram for all models and regions on one panel. Standard in JGR Oceans validation papers.
- [ ] Scatter or 2-D histogram of predicted vs observed u and v with 1:1 line, N, slope, intercept, r. Use density shading, not points, when N is large.
- [ ] Map panels: HFR truth, SWOT geostrophic, model prediction, model minus truth, for one or two example passes, same colorbar across the first three panels and a diverging colorbar for the difference.
- [ ] Case where it fails: one figure dedicated to a failure mode (internal tide contamination, frontal zone, data gap). Reviewers trust papers that show failures.
- [ ] Statistical significance: state the test, the effective degrees of freedom (serial correlation reduces N), and the p-value or interval for every "improves on" claim.
- [ ] Never report skill only against the simulation (LLC4320) if real-data validation exists. Lead with real HFR validation; simulation results support, they do not headline.
- [ ] Tense: past for what you did ("we trained"), present for what the figures show ("Figure 4 shows") and for established facts.

## 10. Figures

Mechanical (AGU and AMS):
- [ ] One complete figure per file; multi-panel figures combined into one file.
- [ ] Raster at 300 dpi minimum for photos/maps, 600 dpi for line art; vector (PDF) preferred for line plots. AGU: PDF over EPS for pdfLaTeX.
- [ ] Width: single column about 8.5 cm (3.3 in), double column about 17.8 cm (7 in). Design at final size. Fonts in figures 7 to 9 pt at final size, same family throughout (Helvetica/Arial or the template font).
- [ ] Caption below the figure. First sentence is a title fragment stating what the figure shows; then define every panel letter, line, color, symbol, and unit. Acronyms in captions get spelled out in the published version, so define them.
- [ ] Panel labels (a), (b), (c) top-left, bold or in parentheses, consistent across all figures.
- [ ] AMS: captions single-spaced or 10.5 pt, same caption format throughout; no text wrapping; print-quality source files required at revision.
- [ ] AGU LaTeX: no `\psfrag`, no `\subfigure`; use `sidewaystable` not `landscapetable`.
- [ ] Supporting Information figures named Figure S1, S2 and cited as "Figure S1 in Supporting Information S1".

Community conventions (what reviewers in SWOT / HFR work expect):
- [ ] Perceptually uniform, colorblind-safe colormaps. cmocean: `balance` for u, v, vorticity (diverging, centered at 0); `speed` or `tempo` for speed; `thermal` for SST; `deep` for bathymetry; `amp` for RMSE/error magnitude. Never jet/rainbow. Never red-green diverging.
- [ ] Diverging colormaps centered on zero with symmetric limits. Sequential for magnitudes. Same limits across panels that are meant to be compared.
- [ ] Every colorbar labeled with quantity and unit (cm s^-1, not m/s; AMS style uses negative exponents).
- [ ] Maps: lat/lon ticks with degree symbols and N/W, coastline, bathymetry contours (200 m and 1000 m isobaths at minimum), SWOT swath edges and nadir gap drawn, HFR coverage footprint drawn, scale bar or gridlines. Equal-area or plate carree with the aspect ratio corrected for latitude.
- [ ] Velocity as quivers subsampled to be readable, or as speed shading plus streamlines. Never dense quivers that turn black.
- [ ] Spectra: log-log, wavelength axis on top in km, wavenumber on bottom in cpkm, reference slopes (k^-2, k^-3, k^-11/3 as relevant) as thin dashed lines, noise floor marked, confidence interval shown.
- [ ] Time series: UTC on the x axis, ISO dates, pass or cycle numbers annotated if relevant.
- [ ] Figure 1 is the study map. Last figure before Discussion is the failure/limits figure.
- [ ] Every figure must be readable in grayscale print (line styles differ, not just colors).
- [ ] Figures generated by script, script archived with the paper (Kachelein 2024 archived "scripts used to generate all figures"). Reviewers increasingly ask for this.
- [ ] Animations go in SI as Movie S1 with a static key frame in the main text.

## 11. Tables

- [ ] Caption above the table (AGU); AMS wants a full caption directly beneath. Follow the template.
- [ ] No shaded cells, color, underlining, or graphics (AMS rule; follow it for AGU too).
- [ ] Units in column headers, not in cells. Same number of decimals down a column. Bold the best value per row.
- [ ] Table notes in the last row with `\multicolumn` (agutex) or footnotes (AMS).
- [ ] Appendix tables numbered A1, A2; SI tables S1, S2.

## 12. Equations, symbols, units

- [ ] SI units throughout. Velocity in m s^-1 or cm s^-1 (pick one and keep it). Negative exponents, not slashes, in AMS; AGU accepts either but be consistent.
- [ ] A space between number and unit; no space before percent in AGU style ("10%"), AMS writes "10%" too.
- [ ] Variables italic, vectors bold upright or bold italic per template, units upright, functions upright (sin, exp), subscripts that are words upright (u_geo), subscripts that are variables italic.
- [ ] Every symbol defined at first use, with units. Consider a notation table in the appendix if more than about 20 symbols.
- [ ] Number all displayed equations; refer to them as Equation (3) in AGU, Eq. (3) in AMS. AGU LaTeX: wrap numbered equations in `\begin{linenomath*}`.
- [ ] Write out the geostrophic relation, the loss function, and every metric. Do not describe an equation in words when it can be written.
- [ ] Latitude and longitude as 36.5 degrees N, 122.0 degrees W (decimal degrees, hemisphere letter). Times as 2023-04-01 06:00 UTC.
- [ ] Numbers: spell out one to nine in prose, numerals from 10; always numerals with units. Significant figures matched to the uncertainty (do not report RMSE to four decimals when the interval is plus or minus 1 cm s^-1).
- [ ] Use "about" rather than the tilde symbol in prose.

## 13. Grammar and house style

- [ ] Define every acronym at first use in the abstract and again at first use in the body. Keep a list: SWOT, KaRIn, SSH, SSHA, MDT, HFR, SST, MLP, RF, RMSE, OI, DUACS, ERA5, GOES, IOOS, PO.DAAC.
- [ ] Do not define an acronym you use fewer than three times; just write the words.
- [ ] Active voice, first person plural ("we trained"). Both AGU and AMS accept this; the exemplars all use it.
- [ ] Past tense for actions and results obtained; present tense for what figures show and for established physics.
- [ ] "Data are" plural.
- [ ] Hyphenate compound modifiers: "high-frequency radar", "25-h mean", "machine-learned velocities", "sub-mesoscale" is now usually "submesoscale" (one word, AMS and JPO usage).
- [ ] AMS has a style sheet for oceanography terms (for example "sea surface height", "sea surface temperature", "mixed layer depth" unhyphenated as nouns). Check it for JPO/JTECH.
- [ ] Avoid "significant" unless statistically tested. Avoid "novel", "unprecedented", "for the first time" (editors strip these; reviewers resent them). Say what is new once, in the introduction, in plain terms.
- [ ] Avoid "validate" when you mean "compare"; in this community "validation against independent observations" is a specific claim.
- [ ] Short sentences. One idea per sentence. Lead paragraphs with the finding, then the evidence.
- [ ] Spell out "Figure" in prose (AGU); AMS uses "Fig." mid-sentence and "Figure" at sentence start.
- [ ] Use "cf.", "e.g.", "i.e." sparingly and only inside parentheses.
- [ ] Spell check against American English for both publishers.

## 14. References and citations

- [ ] AGU: APA style, author-year in text, alphabetical reference list, DOI on every reference as a URL (`https://doi.org/...`). Use only `\cite` and `\citeA` in agutex (no `\citet`, `\citep`).
- [ ] AMS: author-year, alphabetical, unnumbered; "(Rossby 1945)" without a comma; suffixes a, b for same-author same-year.
- [ ] Cite datasets and software in the reference list with bracketed type labels: [Dataset], [Software], [ComputationalNotebook]. Required items per AGU checklist: type, repository name, persistent identifier (DOI preferred), access conditions, license, version (software), development-platform link (software), author or project name (not a username).
- [ ] Datasets to cite as references, each with DOI: SWOT L3 (AVISO/CNES or PO.DAAC DOI), HFR (NDBC THREDDS / HFRNet, plus the network paper), ERA5 (Hersbach 2020 and the CDS DOI), GOES SST (NOAA), GDP drifters if used, LLC4320 if used, DUACS if used as baseline.
- [ ] Software to cite: PyTorch, scikit-learn, cuML or LightGBM if used, xarray, numpy, cmocean, cartopy, and your own SWOTxAI release.
- [ ] Preprints cited with the server and DOI and marked as preprints. Replace with the published version at revision.
- [ ] Do not cite "personal communication" for anything load-bearing.
- [ ] Check that every reference is cited and every citation has a reference. Run a script.
- [ ] 40 to 80 references is typical for JGR Oceans; GRL exemplars have 30 to 50.

## 15. Open Research / Availability Statement (this is a reject trigger if wrong)

AGU section title: "Open Research". AMS: "Data Availability Statement" in its own section.

- [ ] "Available from the authors" or "available upon request" is not acceptable. A link to a project website is not acceptable. Supporting Information alone is not an acceptable data home.
- [ ] GitHub alone is not an acceptable preservation repository. Deposit a tagged release of the SWOTxAI code on Zenodo (GitHub-to-Zenodo integration), cite the Zenodo DOI and the version, and also link the GitHub repository. Pattern from Martin 2025: "We share the code through a public GitHub repository archived on Zenodo (Martin, 2025a)".
- [ ] Deposit: trained model weights (`.pt`, `.joblib`), the colocated training/target matrices or the scripts plus exact inputs that regenerate them, the evaluation outputs behind every figure and table, and the figure scripts. Zenodo takes 50 GB per record by default; request more or split by region if needed. Large raw inputs (the 17 GB mirror) do not need depositing if the public source and exact product version, variable names, time ranges, and spatial boxes are stated.
- [ ] State data product names, variable names, time ranges, spatial boxes, and access conditions for every third-party dataset (AGU requires this level of detail).
- [ ] State the license for code (MIT, matching the repo) and data (CC-BY 4.0).
- [ ] The S3 bucket (`s3://swot-ai-ssv`) is not a citable repository. Do not reference it.
- [ ] Software must be accessible to reviewers at submission (private Zenodo link or reviewer access to the repo is fine), and public at acceptance.
- [ ] Cite every item in the statement in the reference list too.
- [ ] Draft text pattern:
  "SWOT Level-3 ... (version ...) is available from ... (DOI). HFR surface currents ... (DOI). ERA5 ... (DOI). The SWOTxAI code (version X.Y), trained model weights, colocated training data, and scripts that reproduce every figure are archived on Zenodo (DOI) under the MIT license and developed at https://github.com/DALEE9000/..."

## 16. Supporting Information

- [ ] Separate file, its own cover page with title and authors, "Contents of this file" list, then Text S1, Figure S1, Table S1, Movie S1 in order.
- [ ] Everything in SI is cited from the main text at least once.
- [ ] Put in SI: per-region figures that repeat a main-text figure, full hyperparameter sweeps, evolve history, extra ablations, all 40 example passes. Keep in main text: anything the conclusions depend on.
- [ ] AMS prefers an appendix over a supplement for text, normal tables, equations, and figures; supplements are for large tables and animations.

## 17. Other required sections

- [ ] Acknowledgments: funding sources with grant numbers (omit the word "number" in AMS style), computing resources, data providers (NASA/CNES SWOT project, IOOS HFR network and the operating institutions, ECMWF/Copernicus, NOAA), colleagues who helped but are not authors. AMS also wants conflicts of interest here.
- [ ] Conflict of Interest (AGU): "The authors declare there are no conflicts of interest for this manuscript."
- [ ] Author contributions (CRediT): optional at AGU submission but expected; write it anyway (conceptualization, data curation, methodology, software, validation, visualization, writing original draft, review and editing, supervision, funding).
- [ ] ORCID for the corresponding author (required); affiliations where the work was done; one corresponding author only in the AGU system.
- [ ] Inclusion in Global Research Statement only if fieldwork or local collaborators abroad; probably not applicable.
- [ ] Funding statement is a separate required field in the AGU system.
- [ ] Keywords: 5 to 6 (AGU index terms are chosen from a list in the system; AMS has its own keyword list).

## 18. Preprint-specific items (before posting to ESS Open Archive or arXiv)

- [ ] Server: ESS Open Archive (AGU's own; same submission can be forwarded to the journal) or arXiv physics.ao-ph (wider ML audience; oceanscout indexes it). Posting to both is allowed. AMS accepts not-for-profit preprint servers and recommends a header or footnote saying the work has not been peer reviewed.
- [ ] Add the line "This manuscript is a non-peer-reviewed preprint submitted to [journal]" on the title page, plus a version number and date.
- [ ] Line numbers on. Page numbers on. Double or 1.5 spacing per template. Figures embedded near first mention (not at the end) so the preprint reads well.
- [ ] Full author list with affiliations, ORCIDs, and the corresponding author email on page 1.
- [ ] CC-BY 4.0 license on the preprint so it can be reused, and so the journal version has no conflict.
- [ ] Open Research section already complete in the preprint with live DOIs. Reserve the Zenodo DOI before posting (Zenodo lets you reserve a DOI before publishing the record).
- [ ] Title and abstract on the preprint server exactly match the manuscript (the server metadata is what gets indexed and what oceanscout-like tools read).
- [ ] Key Points and PLS included in the preprint even though the server does not require them; they are what skimmers read.
- [ ] No "submitted to GRL" in the file if there is any chance of changing the target; use "submitted for publication".
- [ ] Check the PDF: fonts embedded, no overfull hboxes bleeding into margins, hyperlinks working, figure resolution preserved (do not let the PDF compiler downsample).
- [ ] Run the manuscript through a spell checker and a reference checker. Then read it aloud once.

## 19. Self-review against what reviewers actually write (from the exemplars and the lit review gaps)

- [ ] Can a reviewer rebuild your training matrix from Section 2 and 3 alone? If not, add detail.
- [ ] Is the test set temporally and spatially independent, and is that stated in the abstract?
- [ ] Is every "improves on" claim quantified against the strongest baseline, with an interval?
- [ ] Is there a figure showing where it fails?
- [ ] Is the representativeness mismatch between HFR effective depth and SWOT geostrophic depth discussed, with a number?
- [ ] Are internal tides addressed (both in the SWOT SSH and in the HFR target)?
- [ ] Is the filter scale on SWOT SSH stated and justified, and compared to the 18 / 25 / 70 km choices in the literature?
- [ ] Are uncertainties attached to the product, not only to the summary statistics?
- [ ] Does the Discussion say what the result means for SWOT science (coastal geostrophy breakdown, ageostrophic recovery fraction, what fraction of HFR variance is explained) rather than only for the model?
- [ ] Does the Conclusion stop at what was shown? No "future work will" padding beyond one paragraph.
- [ ] Is the length within the journal limit with the PU calculation written down?
- [ ] Are there any sentences that only an author would understand (project jargon: "calval", "science orbit pkl", "evolve champion", "flattened")? Rename to community terms (1-day orbit phase, 21-day science phase, selected architecture, training matrix).

## 20. Journal-specific delta sheet

GRL only:
- 12 PU hard cap; 150-word abstract; Key Points and PLS mandatory; "broad and immediate implications" test; 3 to 4 figures; everything else to SI.

JGR Oceans only:
- 250-word abstract; Key Points and PLS; longer Data and Methods expected; regional validation tables expected; 8 to 12 figures typical.

JAMES only:
- Method papers welcome; expects full training details, ablations, data-challenge style benchmark tables (Martin 2025 Table A1 pattern: method, RMSE score, effective resolution in km, notes), and code release.

AMS (JPO, JTECH, AIES) only:
- 7,500 words; 250-word abstract with no citations; Significance Statement 120 words; 12 pt, 1.5 spacing, 1 inch margins, line numbers; AMS heading style; negative-exponent units; Availability Statement after Acknowledgments; appendix preferred over supplement.

---

## 21. Community conventions from oceanscout (110 sources, run 2026-10-16)

Full synthesis with references: `../../oceanscout/answers/2026-10-16-what-do-the-most-cited-and-most-recent-papers-on-swot-derive.md`.
Reference numbers below are that file's.

The genre template comes from the DUACS product papers (Pujol 2016 [10], Taburet 2019 [12]), not from ML: describe the product and processing chain, compare against independent observations, report an error budget in physical units stratified by regime. SWOT and ML velocity papers still follow it.

Research question framing:
- [ ] Phrase the question comparatively and quantitatively: "product X reduces velocity error by N percent relative to DUACS against independent in situ velocity" (Verger-Miralles 2025 [57]: 35 percent vs ADCP, 31 percent vs drifters; Barabinot 2026 [90]: 22 and 34 percent vs drifters) or "resolves scales down to L km and T days" (Fablet 2024 [58]; Demol 2026 [89]). Avoid open-ended "we explore" framing; it reads as an idealized ML paper.
- [ ] The currency of the field is "percent better than DUACS". Include a DUACS (or MIOST / 4DVarNet) column in the headline table even though HFR is the truth.

Representativeness disclosures that reviewers now expect in Data and Methods (every one, explicitly):
- [ ] SWOT product version (L3 v2.0.1 vs v3.0 gains are 21 to 40 percent, as large as gains over DUACS [90]).
- [ ] Orbit phase and the caveat that the 1-day cal/val phase "is not representative of the nominal 21-day science orbit" (Tchonang 2026 [76]).
- [ ] Filter scale before differentiating SSH (70 km in [76]; 25 km Gaussian in [89]) and a sensitivity test of the conclusions to that scale.
- [ ] Wind / Ekman correction used and its source (ERA5 in [89]).
- [ ] Internal-tide correction applied, and the residual it leaves (Yadidya 2026 [75]: model-based non-phase-locked correction cuts error 59 percent).
- [ ] In situ platform depth: HFR top metre vs SVP 15 m drogue vs SWOT geostrophy with no depth (Morey 2018 [47]); HFR sub-footprint variability of 3 to 5 cm s^-1 (Ohlmann 2007 [24]). Cite both; reviewers will.
- [ ] Independence of truth from training and calibration: HFR goes into the CNES-CLS22 MDT [64], so if that MDT is a feature, HFR is not fully independent truth. Say so. Asdar 2025 [87] trained on drifters and validated on drifters; do not repeat that.
- [ ] Matchup counts and collocation windows (space, time) in a table.

Evaluation stack that has converged across [57][63][89][90][104]:
- [ ] Pointwise matchups: bias, RMSE, correlation, for speed and direction or u and v separately.
- [ ] Wavenumber spectra or a PSD-score effective resolution (Ballarotta 2019, Ocean Science).
- [ ] One case-study snapshot of a coherent feature with in situ data overlaid.
- [ ] Lagrangian trajectory skill (Liu and Weisberg 2011 normalized separation) is now expected alongside Eulerian metrics (Wu 2026 [104]; Rosquete-Estevez 2026 [103]). Seed virtual drifters in HFR and in model fields and compare.
- [ ] Variance fractions: balanced signal, noise, unresolved, by product and processing choice (Demol 2026 [89]: residual noise about one third of L3 variance). This is the closest thing to a modern velocity error budget; copying the format is the easiest way to look current.
- [ ] Component-wise skill (geostrophic, Ekman, inertial, tidal bands) following Ubelmann 2021 [30]; HFR hourly sampling allows it and nobody has done it against SWOT.

Significance (weakest-reported element in the field; a differentiator if done):
- [ ] Bootstrap confidence intervals on RMSE differences between model and baseline.
- [ ] Effective degrees of freedom from decorrelation scales of the matchups (block bootstrap). Claims of 20 to 30 percent improvement from a few dozen drifters [90] are what reviewers push back on.
- [ ] Spread-skill and interval-coverage plots if any uncertainty is reported (Beauchamp 2025 [82] is the UQ precedent; none is validated against observations yet).

Figure conventions observed in the genre:
- [ ] Figure 1: region map with SWOT swaths, HFR footprint, drifter tracks, over SST or SSH.
- [ ] Snapshot panel row comparing products (DUACS, SWOT geostrophic, ML) with quivers and vorticity over f.
- [ ] Scatter or 2-D histogram of product vs in situ u and v with N, RMSE, bias, r annotated on the panel.
- [ ] Wavenumber KE or SSH spectra with noise floor and slope guides, or PSD-ratio resolution curves.
- [ ] Time series at moorings or along HFR grid points.
- [ ] Binned RMSE maps, or error vs distance to coast and vs cross-track position in the swath.
- [ ] Trajectory overlays for Lagrangian skill.
- [ ] Taylor diagrams are common in HFR and multi-product evaluations, less so in SWOT GRL papers; use one if comparing more than three products.

Venue signals from the literature:
- [ ] GRL exemplars ([57][63][89]) are one feature, one to three independent truths, one headline percentage, about four figures.
- [ ] JAMES exemplars ([58][61][78]) carry long OSSE sections, ablations, and a citable code release.
- [ ] Ocean Science has open signed review; its discussion pages are the best place to read what reviewers of SWOT velocity papers actually ask. Read the review threads for Wu 2026 [104] and de Marez 2026 [85] before drafting the Discussion.
- [ ] Report the OSSE (LLC4320) result as supporting, and the real HFR result as the headline; the ML sub-genre is criticized for the reverse.

Reviewer expectation list (synthesized from [76][87][89][90]):
1. Operational baseline plus current state of the art (DUACS, then MIOST and/or 4DVarNet).
2. Truth independent of training and calibration.
3. Explicit depth mismatch and ageostrophic treatment.
4. Sensitivity to filter scale.
5. Matchup counts and collocation windows.
6. For ML: out-of-region and out-of-period testing, plus a sim-to-real check.
7. A limitations paragraph naming failure regimes: coast, internal tides, low eddy energy.

Metrics methods to cite when defining them: Kundu 1976 (complex correlation), Taylor 2001 (Taylor diagram), Ballarotta 2019 (effective resolution), Liu and Weisberg 2011 (Lagrangian skill), Chelton 2019 Prog. Oceanogr. (SSH noise propagation into velocity, the k^2 amplification that justifies the filter scale), Elipot 2016 (hourly GDP drifters), Roarty 2019 (HFR network spec), Paduan and Washburn 2013 (HFR review).
