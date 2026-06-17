import { useId, useState, type ReactNode } from "react";

/**
 * Second-level tabs nested INSIDE a top-level `Tabs` panel — e.g. the individual classical
 * models within Methodology's "Theoretical approaches" tab, or sub-sections within the ML
 * tab. Rendered as a compact set of pill / chip controls (accent-soft active), optionally
 * stacked vertically as a left rail on wide viewports (`orientation="vertical"`).
 *
 * Like `Tabs` it is a fully accessible roving-tabindex tablist (arrow-key navigation,
 * `aria-selected` / `aria-controls`), but visually lighter so the nesting reads clearly.
 * Each item is `{ id, label, content }`; labels are pre-translated by the caller.
 */
export interface SubTabDef {
  id: string;
  label: ReactNode;
  content: ReactNode;
}

export interface SubTabsProps {
  tabs: SubTabDef[];
  /** id of the initially-selected sub-tab; defaults to the first. */
  initial?: string;
  /** Accessible label for the tablist. */
  ariaLabel?: string;
  /** "horizontal" (pill row, default) or "vertical" (left rail on wide screens). */
  orientation?: "horizontal" | "vertical";
}

export function SubTabs({ tabs, initial, ariaLabel, orientation = "horizontal" }: SubTabsProps) {
  const baseId = useId();
  const first = tabs[0]?.id ?? "";
  const [active, setActive] = useState<string>(initial ?? first);
  const vertical = orientation === "vertical";

  function onKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, idx: number) {
    const fwd = vertical ? "ArrowDown" : "ArrowRight";
    const back = vertical ? "ArrowUp" : "ArrowLeft";
    if (e.key !== fwd && e.key !== back && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    let next = idx;
    if (e.key === fwd) next = (idx + 1) % tabs.length;
    else if (e.key === back) next = (idx - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    const target = tabs[next];
    if (target) {
      setActive(target.id);
      document.getElementById(`${baseId}-subtab-${target.id}`)?.focus();
    }
  }

  return (
    <div className={vertical ? "subtabs subtabs-vertical" : "subtabs"}>
      <div
        className="subtablist"
        role="tablist"
        aria-label={ariaLabel}
        aria-orientation={vertical ? "vertical" : "horizontal"}
      >
        {tabs.map((tab, idx) => {
          const selected = tab.id === active;
          return (
            <button
              key={tab.id}
              id={`${baseId}-subtab-${tab.id}`}
              role="tab"
              type="button"
              aria-selected={selected}
              aria-controls={`${baseId}-subpanel-${tab.id}`}
              tabIndex={selected ? 0 : -1}
              className={selected ? "subtab active" : "subtab"}
              onClick={() => setActive(tab.id)}
              onKeyDown={(e) => onKeyDown(e, idx)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <div className="subtabpanels">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            id={`${baseId}-subpanel-${tab.id}`}
            role="tabpanel"
            aria-labelledby={`${baseId}-subtab-${tab.id}`}
            hidden={tab.id !== active}
            tabIndex={0}
            className="subtabpanel"
          >
            {tab.id === active ? tab.content : null}
          </div>
        ))}
      </div>
    </div>
  );
}
