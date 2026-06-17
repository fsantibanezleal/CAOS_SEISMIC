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
  du2016: {
    id: "du2016",
    label: "Du et al. 2016 (RMTPP)",
    full: "Du, N., Dai, H., Trivedi, R., Upadhyay, U., Gomez-Rodriguez, M., & Song, L. (2016). Recurrent Marked Temporal Point Processes: Embedding Event History to Vector. KDD 2016, 1555–1564.",
    doi: "10.1145/2939672.2939875",
  },
  meiEisner2017: {
    id: "meiEisner2017",
    label: "Mei & Eisner 2017 (Neural Hawkes)",
    full: "Mei, H., & Eisner, J. (2017). The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process. NeurIPS 2017 (arXiv:1612.09328).",
    url: "https://arxiv.org/abs/1612.09328",
  },
  zhang2020sahp: {
    id: "zhang2020sahp",
    label: "Zhang et al. 2020 (Self-Attentive Hawkes)",
    full: "Zhang, Q., Lipani, A., Kirnap, O., & Yilmaz, E. (2020). Self-Attentive Hawkes Process. ICML 2020 (arXiv:1907.07561).",
    url: "https://arxiv.org/abs/1907.07561",
  },
  zuo2020thp: {
    id: "zuo2020thp",
    label: "Zuo et al. 2020 (Transformer Hawkes)",
    full: "Zuo, S., Jiang, H., Li, Z., Zhao, T., & Zha, H. (2020). Transformer Hawkes Process. ICML 2020, PMLR 119, 11692–11702.",
    url: "https://proceedings.mlr.press/v119/zuo20a.html",
  },
  devries2018: {
    id: "devries2018",
    label: "DeVries et al. 2018",
    full: "DeVries, P. M. R., Viégas, F., Wattenberg, M., & Meade, B. J. (2018). Deep learning of aftershock patterns following large earthquakes. Nature 560, 632–634.",
    doi: "10.1038/s41586-018-0438-y",
  },
  mignanBroccardo2019: {
    id: "mignanBroccardo2019",
    label: "Mignan & Broccardo 2019",
    full: "Mignan, A., & Broccardo, M. (2019). One neuron versus deep learning in aftershock prediction. Nature 575, E1–E3.",
    doi: "10.1038/s41586-019-1582-8",
  },
  devriesReply2019: {
    id: "devriesReply2019",
    label: "DeVries et al. 2019 (Reply)",
    full: "DeVries, P. M. R., Viégas, F., Wattenberg, M., & Meade, B. J. (2019). Reply to: One neuron versus deep learning in aftershock prediction. Nature 575, E4–E5.",
    doi: "10.1038/s41586-019-1583-7",
  },
  zlydenko2023: {
    id: "zlydenko2023",
    label: "Zlydenko et al. 2023 (FERN)",
    full: "Zlydenko, O., Elidan, G., Hassidim, A., Kukliansky, D., Matias, Y., Meade, B., Molchanov, A., Nevo, A., & Bar-Sinai, Y. (2023). A neural encoder for earthquake rate forecasting. Scientific Reports 13, 12350.",
    doi: "10.1038/s41598-023-38033-9",
  },
  schultz2026: {
    id: "schultz2026",
    label: "Schultz 2026",
    full: "Schultz, R. (2026). Forecasting the Rate of Induced Seismicity as a Neural Temporal Point Process. JGR: Machine Learning and Computation.",
    doi: "10.1029/2025JH001052",
  },
  zhuBeroza2019: {
    id: "zhuBeroza2019",
    label: "Zhu & Beroza 2019 (PhaseNet)",
    full: "Zhu, W., & Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. GJI 216(1), 261–273.",
    doi: "10.1093/gji/ggy423",
  },
  mousavi2020: {
    id: "mousavi2020",
    label: "Mousavi et al. 2020 (EQTransformer)",
    full: "Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature Communications 11, 3952.",
    doi: "10.1038/s41467-020-17591-w",
  },
  woollam2022: {
    id: "woollam2022",
    label: "Woollam et al. 2022 (SeisBench)",
    full: "Woollam, J., et al. (2022). SeisBench—A Toolbox for Machine Learning in Seismology. SRL 93(3), 1695–1709.",
    doi: "10.1785/0220210324",
  },
  liu2024seislm: {
    id: "liu2024seislm",
    label: "Liu et al. 2024 (SeisLM)",
    full: "Liu, T., Münchmeyer, J., Laurenti, L., Marone, C., de Hoop, M. V., & Dokmanić, I. (2024). SeisLM: a Foundation Model for Seismic Waveforms. NeurIPS 2024 (arXiv:2410.15765).",
    url: "https://arxiv.org/abs/2410.15765",
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
  gutenbergRichter1944: {
    id: "gutenbergRichter1944",
    label: "Gutenberg & Richter 1944",
    full: "Gutenberg, B., & Richter, C. F. (1944). Frequency of earthquakes in California. BSSA 34(4), 185–188.",
    doi: "10.1785/BSSA0340040185",
  },
  ogata1983: {
    id: "ogata1983",
    label: "Ogata 1983",
    full: "Ogata, Y. (1983). Estimation of the parameters in the modified Omori formula for aftershock frequency by the maximum likelihood procedure. Journal of Physics of the Earth 31(2), 115–124.",
    doi: "10.4294/jpe1952.31.115",
  },
  zhuang2002: {
    id: "zhuang2002",
    label: "Zhuang, Ogata & Vere-Jones 2002",
    full: "Zhuang, J., Ogata, Y., & Vere-Jones, D. (2002). Stochastic declustering of space-time earthquake occurrences. JASA 97(458), 369–380.",
    doi: "10.1198/016214502760046925",
  },
  reasenbergJones1994: {
    id: "reasenbergJones1994",
    label: "Reasenberg & Jones 1994",
    full: "Reasenberg, P. A., & Jones, L. M. (1994). Earthquake Aftershocks: Update. Science 265(5176), 1251–1252.",
    doi: "10.1126/science.265.5176.1251",
  },
  rhoadesEvison2004: {
    id: "rhoadesEvison2004",
    label: "Rhoades & Evison 2004",
    full: "Rhoades, D. A., & Evison, F. F. (2004). Long-range Earthquake Forecasting with Every Earthquake a Precursor According to Scale. Pure and Applied Geophysics 161, 47–72.",
    doi: "10.1007/s00024-003-2434-9",
  },
  matthews2002: {
    id: "matthews2002",
    label: "Matthews, Ellsworth & Reasenberg 2002",
    full: "Matthews, M. V., Ellsworth, W. L., & Reasenberg, P. A. (2002). A Brownian Model for Recurrent Earthquakes. BSSA 92(6), 2233–2250.",
    doi: "10.1785/0120010267",
  },
  schwartzCoppersmith1984: {
    id: "schwartzCoppersmith1984",
    label: "Schwartz & Coppersmith 1984",
    full: "Schwartz, D. P., & Coppersmith, K. J. (1984). Fault behavior and characteristic earthquakes: Examples from the Wasatch and San Andreas fault zones. JGR 89(B7), 5681–5698.",
    doi: "10.1029/JB089iB07p05681",
  },
  kingSteinLin1994: {
    id: "kingSteinLin1994",
    label: "King, Stein & Lin 1994",
    full: "King, G. C. P., Stein, R. S., & Lin, J. (1994). Static stress changes and the triggering of earthquakes. BSSA 84(3), 935–953.",
    doi: "10.1785/BSSA0840030935",
  },
  stein1999: {
    id: "stein1999",
    label: "Stein 1999",
    full: "Stein, R. S. (1999). The role of stress transfer in earthquake occurrence. Nature 402, 605–609.",
    doi: "10.1038/45144",
  },
  dieterich1994: {
    id: "dieterich1994",
    label: "Dieterich 1994",
    full: "Dieterich, J. (1994). A constitutive law for rate of earthquake production and its application to earthquake clustering. JGR 99(B2), 2601–2618.",
    doi: "10.1029/93JB02581",
  },
  heimissonSegall2018: {
    id: "heimissonSegall2018",
    label: "Heimisson & Segall 2018",
    full: "Heimisson, E. R., & Segall, P. (2018). Constitutive Law for Earthquake Production Based on Rate-and-State Friction: Dieterich 1994 Revisited. JGR Solid Earth 123(5), 4141–4156.",
    doi: "10.1029/2018JB015656",
  },
} as const satisfies Record<string, Citation>;

export type CitationId = keyof typeof CITATIONS;

/** Ordered list of all citations (for a full reference block). */
export const ALL_CITATIONS: Citation[] = Object.values(CITATIONS);
