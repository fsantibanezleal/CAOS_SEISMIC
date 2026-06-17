import type { ReactNode } from "react";

/**
 * A small bordered note used to carry the honesty framing inline (the creed, the
 * "complement not compete" stance, the "still small in absolute terms" reminders, the
 * banned-anti-pattern callouts).
 *
 * Tone maps to a left-border accent ONLY — never a filled red/green panel — so the page
 * never inadvertently reads as a traffic-light alarm. `tone: "honest"` is the default warm
 * accent; `"note"` is neutral; `"strong"` uses the accent blue for the load-bearing creed.
 */
export type CalloutTone = "honest" | "note" | "strong";

export interface CalloutProps {
  tone?: CalloutTone;
  title?: ReactNode;
  children: ReactNode;
}

export function Callout({ tone = "honest", title, children }: CalloutProps) {
  return (
    <aside className={`callout callout-${tone}`}>
      {title ? <p className="callout-title">{title}</p> : null}
      <div className="callout-body">{children}</div>
    </aside>
  );
}
