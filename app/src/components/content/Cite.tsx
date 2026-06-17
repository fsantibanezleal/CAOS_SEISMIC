import { CITATIONS, ALL_CITATIONS, citationHref, type Citation, type CitationId } from "@/lib/citations";

/**
 * Citation rendering helpers shared across the content pages.
 *
 * `Cite` renders an inline reference (the short label, linked to its DOI/URL when present),
 * e.g. "(Geller, Jackson, Kagan & Mulargia 1997)". `ReferenceList` renders a full, ordered
 * bibliography block from an explicit subset of ids (a page lists only what it cites).
 *
 * Citations are data (lib/citations.ts), not i18n strings: a DOI/author line is identical in
 * every language. The surrounding prose carries the translation.
 */

export interface CiteProps {
  /** The citation key in `CITATIONS`. */
  id: CitationId;
  /** Wrap the label in parentheses (inline parenthetical style). Default true. */
  paren?: boolean;
}

export function Cite({ id, paren = true }: CiteProps) {
  const c: Citation = CITATIONS[id];
  const href = citationHref(c);
  const label = href ? (
    <a href={href} target="_blank" rel="noreferrer noopener">
      {c.label}
    </a>
  ) : (
    <span>{c.label}</span>
  );
  return (
    <cite className="cite-inline">
      {paren ? "(" : null}
      {label}
      {paren ? ")" : null}
    </cite>
  );
}

export interface ReferenceListProps {
  /** Ordered ids to render. Omit to render every citation (the master list). */
  ids?: CitationId[];
  /** Heading text (already translated by the caller). */
  heading?: string;
}

export function ReferenceList({ ids, heading }: ReferenceListProps) {
  const items: Citation[] = ids ? ids.map((k) => CITATIONS[k]) : ALL_CITATIONS;
  return (
    <section className="references" aria-label={heading ?? "References"}>
      {heading ? <h2>{heading}</h2> : null}
      <ol className="reference-list">
        {items.map((c) => {
          const href = citationHref(c);
          return (
            <li key={c.id} id={`ref-${c.id}`}>
              <span>{c.full}</span>{" "}
              {href ? (
                <a href={href} target="_blank" rel="noreferrer noopener" className="faint">
                  {c.doi ? `doi:${c.doi}` : "link"}
                </a>
              ) : null}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
