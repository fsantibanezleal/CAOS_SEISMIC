// Canonical, public-safe citations rendered across the content pages
// (Problem, Methodology, Implementation, Back-analysis).
//
// These mirror the reference list in the build spec (web-app-spec.md §10) and the
// synthesis docs. They are language-neutral bibliographic facts, so they live here as data
// (not in the i18n bundles) — the surrounding prose is what gets translated, the DOI/author
// string is the same in every language. Each entry carries a stable `id` so prose can refer
// to it (e.g. "(Geller et al. 1997)") and an optional `doi`/`url` for a real outbound link.

export interface Citation {
  /** Stable key used to reference the citation from prose and the reference list. */
  id: string;
  /** Short inline label, e.g. "Geller et al. 1997". */
  label: string;
  /** Full bibliographic string (authors, year, title, venue). */
  full: string;
  /** DOI without the leading scheme, e.g. "10.1126/science.275.5306.1616". */
  doi?: string;
  /** A direct URL when there is no DOI (docs, software pages). */
  url?: string;
}

/** Resolve a citation to its canonical outbound href (DOI preferred), or undefined. */
export function citationHref(c: Citation): string | undefined {
  if (c.doi) return `https://doi.org/${c.doi}`;
  return c.url;
}

export const CITATIONS = {
  geller1997: {
    id: "geller1997",
    label: "Geller, Jackson, Kagan & Mulargia 1997",
    full: "Geller, R. J., Jackson, D. D., Kagan, Y. Y., & Mulargia, F. (1997). Earthquakes Cannot Be Predicted. Science 275(5306), 1616–1617.",
    doi: "10.1126/science.275.5306.1616",
  },
  bakTang1989: {
    id: "bakTang1989",
    label: "Bak & Tang 1989",
    full: "Bak, P., & Tang, C. (1989). Earthquakes as a self-organized critical phenomenon. JGR 94(B11), 15635–15637.",
    doi: "10.1029/JB094iB11p15635",
  },
  icef2011: {
    id: "icef2011",
    label: "Jordan et al. 2011 (ICEF)",
    full: "Jordan, T. H., et al. (2011). Operational Earthquake Forecasting: State of Knowledge and Guidelines for Utilization (ICEF Report). Annals of Geophysics 54(4), 315–391.",
    doi: "10.4401/ag-5350",
  },
  bakunLindh1985: {
    id: "bakunLindh1985",
    label: "Bakun & Lindh 1985",
    full: "Bakun, W. H., & Lindh, A. G. (1985). The Parkfield, California, earthquake prediction experiment. Science 229(4714), 619–624.",
    doi: "10.1126/science.229.4714.619",
  },
  bakun2005: {
    id: "bakun2005",
    label: "Bakun et al. 2005",
    full: "Bakun, W. H., et al. (2005). Implications for prediction and hazard assessment from the 2004 Parkfield earthquake. Nature 437, 969–974.",
    doi: "10.1038/nature04067",
  },
  savran2020: {
    id: "savran2020",
    label: "Savran et al. 2020",
    full: "Savran, W. H., et al. (2020). Pseudoprospective Evaluation of UCERF3-ETAS Forecasts during the 2019 Ridgecrest Sequence. BSSA 110(4), 1799–1817.",
    doi: "10.1785/0120200026",
  },
  ogata1988: {
    id: "ogata1988",
    label: "Ogata 1988",
    full: "Ogata, Y. (1988). Statistical Models for Earthquake Occurrences and Residual Analysis for Point Processes. JASA 83(401), 9–27.",
    doi: "10.1080/01621459.1988.10478560",
  },
  ogata1998: {
    id: "ogata1998",
    label: "Ogata 1998",
    full: "Ogata, Y. (1998). Space-time point-process models for earthquake occurrences. Annals of the Institute of Statistical Mathematics 50(2), 379–402.",
    doi: "10.1023/A:1003403601725",
  },
  reasenbergJones1989: {
    id: "reasenbergJones1989",
    label: "Reasenberg & Jones 1989",
    full: "Reasenberg, P. A., & Jones, L. M. (1989). Earthquake Hazard After a Mainshock in California. Science 243(4895), 1173–1176.",
    doi: "10.1126/science.243.4895.1173",
  },
  gerstenberger2005: {
    id: "gerstenberger2005",
    label: "Gerstenberger et al. 2005",
    full: "Gerstenberger, M. C., Wiemer, S., Jones, L. M., & Reasenberg, P. A. (2005). Real-time forecasts of tomorrow's earthquakes in California. Nature 435, 328–331.",
    doi: "10.1038/nature03622",
  },
  page2016: {
    id: "page2016",
    label: "Page et al. 2016",
    full: "Page, M. T., et al. (2016). Three Ingredients for Improved Global Aftershock Forecasts. BSSA 106(5), 2290–2301.",
    doi: "10.1785/0120160073",
  },
  utsu1995: {
    id: "utsu1995",
    label: "Utsu, Ogata & Matsu'ura 1995",
    full: "Utsu, T., Ogata, Y., & Matsu'ura, R. S. (1995). The Centenary of the Omori Formula for a Decay Law of Aftershock Activity. Journal of Physics of the Earth 43(1), 1–33.",
    doi: "10.4294/jpe1952.43.1",
  },
  aki1965: {
    id: "aki1965",
    label: "Aki 1965",
    full: "Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a − bM and its confidence limits. Bull. Earthq. Res. Inst. 43, 237–239.",
  },
  tintiMulargia1987: {
    id: "tintiMulargia1987",
    label: "Tinti & Mulargia 1987",
    full: "Tinti, S., & Mulargia, F. (1987). Confidence intervals of b values for grouped magnitudes. BSSA 77(6), 2125–2134.",
  },
  wiemerWyss2000: {
    id: "wiemerWyss2000",
    label: "Wiemer & Wyss 2000",
    full: "Wiemer, S., & Wyss, M. (2000). Minimum Magnitude of Completeness in Earthquake Catalogs. BSSA 90(4), 859–869.",
    doi: "10.1785/0119990114",
  },
  woessnerWiemer2005: {
    id: "woessnerWiemer2005",
    label: "Woessner & Wiemer 2005",
    full: "Woessner, J., & Wiemer, S. (2005). Assessing the Quality of Earthquake Catalogues. BSSA 95(2), 684–698.",
    doi: "10.1785/0120040007",
  },
  helmstetter2007: {
    id: "helmstetter2007",
    label: "Helmstetter, Kagan & Jackson 2007",
    full: "Helmstetter, A., Kagan, Y. Y., & Jackson, D. D. (2007). High-resolution time-independent grid-based forecast for M≥5 earthquakes in California. SRL 78(1), 78–86.",
    doi: "10.1785/gssrl.78.1.78",
  },
  schorlemmer2007: {
    id: "schorlemmer2007",
    label: "Schorlemmer et al. 2007",
    full: "Schorlemmer, D., Gerstenberger, M. C., Wiemer, S., Jackson, D. D., & Rhoades, D. A. (2007). Earthquake Likelihood Model Testing. SRL 78(1), 17–29.",
    doi: "10.1785/gssrl.78.1.17",
  },
  zechar2010: {
    id: "zechar2010",
    label: "Zechar, Gerstenberger & Rhoades 2010",
    full: "Zechar, J. D., Gerstenberger, M. C., & Rhoades, D. A. (2010). Likelihood-Based Tests for Evaluating Space-Rate-Magnitude Earthquake Forecasts. BSSA 100(3), 1184–1195.",
    doi: "10.1785/0120090192",
  },
  rhoades2011: {
    id: "rhoades2011",
    label: "Rhoades et al. 2011",
    full: "Rhoades, D. A., et al. (2011). Efficient testing of earthquake forecasting models. Acta Geophysica 59(4), 728–747.",
    doi: "10.2478/s11600-011-0013-5",
  },
  savran2022: {
    id: "savran2022",
    label: "Savran et al. 2022 (pyCSEP)",
    full: "Savran, W. H., et al. (2022). pyCSEP: A Python Toolkit for Earthquake Forecast Developers. SRL 93(5), 2858–2870.",
    doi: "10.1785/0220220033",
  },
  serafini2025: {
    id: "serafini2025",
    label: "Serafini et al. 2025",
    full: "Serafini, F., et al. (2025). A benchmark database of ten years of prospective next-day earthquake forecasts in California from CSEP. Scientific Data 12, 1501.",
    doi: "10.1038/s41597-025-05766-3",
  },
  spassiani2023: {
    id: "spassiani2023",
    label: "Spassiani et al. 2023",
    full: "Spassiani, I., Falcone, G., Murru, M., & Marzocchi, W. (2023). Operational Earthquake Forecasting in Italy: validation after 10 yr of operativity. GJI 234(3), 2501–2518.",
    doi: "10.1093/gji/ggad256",
  },
  stockman2026: {
    id: "stockman2026",
    label: "Stockman, Lawson & Werner 2026 (EarthquakeNPP)",
    full: "Stockman, S., Lawson, D., & Werner, M. J. (2026). EarthquakeNPP: A Benchmark for Earthquake Forecasting with Neural Point Processes. TMLR (arXiv:2410.08226).",
    url: "https://arxiv.org/abs/2410.08226",
  },
  dascher2023: {
    id: "dascher2023",
    label: "Dascher-Cousineau et al. 2023 (RECAST)",
    full: "Dascher-Cousineau, K., et al. (2023). Using Deep Learning for Flexible and Scalable Earthquake Forecasting. GRL 50, e2023GL103909.",
    doi: "10.1029/2023GL103909",
  },
  schneider2022: {
    id: "schneider2022",
    label: "Schneider et al. 2022",
    full: "Schneider, M., McDowell, M., Guttorp, P., Steel, E. A., & Fleischhut, N. (2022). Effective uncertainty visualization for aftershock forecast maps. NHESS 22(4), 1499–1518.",
    doi: "10.5194/nhess-22-1499-2022",
  },
  kagan2017: {
    id: "kagan2017",
    label: "Kagan 2017",
    full: "Kagan, Y. Y. (2017). Earthquake number forecasts testing. GJI 211(1), 335–345.",
    doi: "10.1093/gji/ggx300",
  },
  zecharJordan2008: {
    id: "zecharJordan2008",
    label: "Zechar & Jordan 2008",
    full: "Zechar, J. D., & Jordan, T. H. (2008). Testing alarm-based earthquake predictions. GJI 172(2), 715–724.",
    doi: "10.1111/j.1365-246X.2007.03676.x",
  },
  gneitingRaftery2007: {
    id: "gneitingRaftery2007",
    label: "Gneiting & Raftery 2007",
    full: "Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. JASA 102(477), 359–378.",
    doi: "10.1198/016214506000001437",
  },
  cabuk2025: {
    id: "cabuk2025",
    label: "Çabuk et al. 2025",
    full: "Çabuk, Ç., et al. (2025). Vector Data Rendering Performance Analysis of Open-Source Web Mapping Libraries. ISPRS Int. J. Geo-Information 14(9), 336.",
    doi: "10.3390/ijgi14090336",
  },
  mizrahi2024: {
    id: "mizrahi2024",
    label: "Mizrahi et al. 2024",
    full: "Mizrahi, L., Dallo, I., van der Elst, N. J., et al. (2024). Developing, Testing, and Communicating Earthquake Forecasts: Current Practices and Future Directions. Rev. Geophys. 62, e2023RG000823.",
    doi: "10.1029/2023RG000823",
  },
} as const satisfies Record<string, Citation>;

export type CitationId = keyof typeof CITATIONS;

/** Ordered list of all citations (for a full reference block). */
export const ALL_CITATIONS: Citation[] = Object.values(CITATIONS);
