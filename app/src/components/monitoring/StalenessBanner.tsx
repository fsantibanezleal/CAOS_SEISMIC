import { useTranslation } from "react-i18next";

import type { Staleness } from "@/data/types";

/**
 * Staleness / last-run indicator (web-app-spec.md §7.3, §8.2). One inference per day — the
 * reader must know the data's age. Renders "Forecast generated: {UTC} · next run: {UTC}".
 *
 * If the daily job failed (`ok === false`) the UI MUST degrade visibly — a warning banner
 * here, plus the parent desaturates the field. A stale or corrupted artifact is worse than
 * honestly saying "unavailable", so the failed state is loud, not silent.
 */
export interface StalenessBannerProps {
  staleness: Staleness;
  /** Whether the artifact is the bundled illustrative SAMPLE (adds a separate banner). */
  sample?: boolean;
}

function fmtUtc(iso: string): string {
  // Render the ISO string compactly in UTC; keep it explicit ("UTC") so there is no
  // local-time ambiguity for a global product.
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toISOString().replace("T", " ").replace(".000Z", "").replace("Z", "") + " UTC";
  } catch {
    return iso;
  }
}

export function StalenessBanner({ staleness, sample }: StalenessBannerProps) {
  const { t } = useTranslation();
  const failed = staleness.ok === false;

  return (
    <div className="staleness-stack">
      {sample ? <p className="sample-banner">{t("monitoring.sampleBanner")}</p> : null}

      <div className={`staleness-banner ${failed ? "failed" : "ok"}`} role="status">
        <span className="staleness-dot" aria-hidden="true" />
        <span>
          {t("monitoring.staleness.generated")} <strong>{fmtUtc(staleness.generated)}</strong>
        </span>
        <span aria-hidden="true" className="sep">
          ·
        </span>
        <span>
          {t("monitoring.staleness.next")} <strong>{fmtUtc(staleness.next_run)}</strong>
        </span>
        {failed ? <span className="staleness-warn">{t("monitoring.staleness.failed")}</span> : null}
      </div>
    </div>
  );
}
