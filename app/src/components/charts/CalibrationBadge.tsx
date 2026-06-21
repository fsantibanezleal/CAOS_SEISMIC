import { useTranslation } from "react-i18next";

import type { CsepScores } from "@/data/types";

/**
 * Calibration badge — the ONLY place the green/amber/red traffic-light triad appears, and it
 * describes MODEL QUALITY, never earthquake danger (web-app-spec.md §7.3, evaluation-plan §9).
 *
 *   green    = within CSEP consistency (all tracked tests pass)
 *   amber    = borderline / partial (some tests pass, some borderline)
 *   red      = REJECTED — a tracked test actually failed (do not trust the numbers)
 *   untested = no per-forecast CSEP tests on this artifact (NEUTRAL, not a failure)
 *
 * The "untested" state is distinct from "red" on purpose: a single 1–7 day global forecast has
 * too few M≥5 events to power the N/M/S/L tests, so the per-forecast pass flags are empty — that
 * is NOT a rejection. The model's consistency + skill are established over many windows in the
 * Back-analysis. Conflating the two (the old behaviour) made a healthy daily forecast read as
 * "rejected", which is misleading. red here still means "do not trust", never "danger".
 */
export type CalibrationLevel = "green" | "amber" | "red" | "untested";

/** Derive the badge level from the CSEP pass flags (model-quality only). */
export function calibrationLevel(csep: CsepScores | undefined): CalibrationLevel {
  const pass = csep?.pass;
  // No per-forecast tests recorded ⇒ UNTESTED (neutral), not rejected — see Back-analysis.
  if (!pass || Object.keys(pass).length === 0) return "untested";
  const flags = Object.values(pass);
  const passed = flags.filter(Boolean).length;
  if (passed === flags.length) return "green";
  if (passed >= Math.ceil(flags.length / 2)) return "amber";
  return "red"; // a tracked test genuinely failed
}

export interface CalibrationBadgeProps {
  csep: CsepScores | undefined;
  /** Render the longer label text alongside the dot. */
  showLabel?: boolean;
}

export function CalibrationBadge({ csep, showLabel = true }: CalibrationBadgeProps) {
  const { t } = useTranslation();
  const level = calibrationLevel(csep);
  const label = t(`calibration.level.${level}`);
  return (
    <span className={`calibration-badge level-${level}`} title={t("calibration.tooltip")}>
      <span className="cal-dot" aria-hidden="true" />
      <span className="cal-text">
        {t("calibration.modelQuality")}
        {showLabel ? <>: {label}</> : null}
      </span>
    </span>
  );
}
