import { useTranslation } from "react-i18next";

import type { CalibrationSummary, CsepScores } from "@/data/types";

/**
 * CSEP consistency-test panel (web-app-spec.md §7.3, evaluation-plan.md §6/§9). Renders the
 * N / M / S / L / CL quantile scores with their pass flags, plus the information-gain figures
 * (in NATS, never bits) vs the Poisson null and the ETAS reference.
 *
 * Honest framing carried in copy: passing consistency tests is necessary but NOT sufficient —
 * skill is established only by winning the comparison (T/W) tests against real baselines. The
 * info-gain-vs-ETAS line states this. The per-test pass uses the same model-quality semantics
 * as the calibration badge; it never colours earthquake danger.
 */

const TEST_KEYS: (keyof Pick<CsepScores, "N" | "M" | "S" | "L" | "CL">)[] = ["N", "M", "S", "L", "CL"];

export interface CsepPanelProps {
  calibration: CalibrationSummary;
}

export function CsepPanel({ calibration }: CsepPanelProps) {
  const { t } = useTranslation();
  const csep = calibration.csep ?? {};
  const pass = csep.pass ?? {};
  const allUntested = Object.keys(pass).length === 0;

  const fmtNats = (v: number | null | undefined): string => {
    if (v === null || v === undefined) return "—";
    const sign = v > 0 ? "+" : "";
    return `${sign}${v.toFixed(2)}`;
  };

  return (
    <div className="csep-panel">
      <table className="csep-table">
        <thead>
          <tr>
            <th scope="col">{t("csep.test")}</th>
            <th scope="col">{t("csep.quantile")}</th>
            <th scope="col">{t("csep.status")}</th>
          </tr>
        </thead>
        <tbody>
          {TEST_KEYS.map((k) => {
            const q = csep[k] as number | undefined;
            const p = pass[k];
            const status = p === undefined ? "untested" : p ? "pass" : "fail";
            return (
              <tr key={k}>
                <th scope="row" title={t(`csep.name.${k}`)}>
                  {t(`csep.short.${k}`)}
                </th>
                <td className="mono">{q === undefined ? "—" : q.toFixed(2)}</td>
                <td>
                  <span className={`csep-status status-${status}`}>{t(`csep.statusLabel.${status}`)}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <dl className="infogain">
        <div>
          <dt>{t("csep.igPoisson")}</dt>
          <dd className="mono">{fmtNats(calibration.info_gain_vs_poisson_nats)} nats</dd>
        </div>
        <div>
          <dt>{t("csep.igEtas")}</dt>
          <dd className="mono">{fmtNats(calibration.info_gain_vs_etas_nats)} nats</dd>
        </div>
      </dl>

      {allUntested ? <p className="csep-note untested-note">{t("csep.untestedNote")}</p> : null}
      <p className="csep-note muted">{t("csep.necessaryNotSufficient")}</p>
    </div>
  );
}
