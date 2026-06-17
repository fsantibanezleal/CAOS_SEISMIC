import { useTranslation } from "react-i18next";

import type { CsepScores } from "@/data/types";

/**
 * Calibration badge — the ONLY place the green/amber/red traffic-light triad appears, and it
 * describes MODEL QUALITY, never earthquake danger (web-app-spec.md §7.3, evaluation-plan §9).
 *
 *   green  = within CSEP consistency (all tracked tests pass)
 *   amber  = borderline / partial (some tests pass, some borderline)
 *   red    = rejected / under-tested
 *
 * This deliberately inverts the "red = run" instinct: red here means "do not trust this
 * model's numbers", not "danger". The badge is compact and always-present; the CSEP panel
 * (separate component) expands the per-test detail.
 */
export type CalibrationLevel = "green" | "amber" | "red";

/** Derive the badge level from the CSEP pass flags (model-quality only). */
export function calibrationLevel(csep: CsepScores | undefined): CalibrationLevel {
  const pass = csep?.pass;
  if (!pass || Object.keys(pass).length === 0) return "red"; // under-tested
  const flags = Object.values(pass);
  const passed = flags.filter(Boolean).length;
  if (passed === flags.length) return "green";
  if (passed >= Math.ceil(flags.length / 2)) return "amber";
  return "red";
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
