import { useId, useState, type ReactNode } from "react";

/**
 * A minimal, accessible tab strip for the Methodology page (classical theories vs the
 * analytical / ML tab). No external dependency — a roving-tabindex tablist with arrow-key
 * navigation and `aria-selected` / `aria-controls` wiring.
 *
 * Each tab is `{ id, label, content }`; the label is pre-translated by the caller.
 */
export interface TabDef {
  id: string;
  label: ReactNode;
  content: ReactNode;
}

export interface TabsProps {
  tabs: TabDef[];
  /** id of the initially-selected tab; defaults to the first. */
  initial?: string;
  /** Accessible label for the tablist. */
  ariaLabel?: string;
}

export function Tabs({ tabs, initial, ariaLabel }: TabsProps) {
  const baseId = useId();
  const first = tabs[0]?.id ?? "";
  const [active, setActive] = useState<string>(initial ?? first);

  function onKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, idx: number) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft" && e.key !== "Home" && e.key !== "End") {
      return;
    }
    e.preventDefault();
    let next = idx;
    if (e.key === "ArrowRight") next = (idx + 1) % tabs.length;
    else if (e.key === "ArrowLeft") next = (idx - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    const target = tabs[next];
    if (target) {
      setActive(target.id);
      document.getElementById(`${baseId}-tab-${target.id}`)?.focus();
    }
  }

  return (
    <div className="tabs">
      <div className="tablist" role="tablist" aria-label={ariaLabel}>
        {tabs.map((tab, idx) => {
          const selected = tab.id === active;
          return (
            <button
              key={tab.id}
              id={`${baseId}-tab-${tab.id}`}
              role="tab"
              type="button"
              aria-selected={selected}
              aria-controls={`${baseId}-panel-${tab.id}`}
              tabIndex={selected ? 0 : -1}
              className={selected ? "tab active" : "tab"}
              onClick={() => setActive(tab.id)}
              onKeyDown={(e) => onKeyDown(e, idx)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          id={`${baseId}-panel-${tab.id}`}
          role="tabpanel"
          aria-labelledby={`${baseId}-tab-${tab.id}`}
          hidden={tab.id !== active}
          tabIndex={0}
          className="tabpanel"
        >
          {tab.id === active ? tab.content : null}
        </div>
      ))}
    </div>
  );
}
