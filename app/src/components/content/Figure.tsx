import type { ReactNode } from "react";

/**
 * A generic figure wrapper for the deep content pages: it renders an inline SVG diagram
 * (or any block visual) passed as `children` together with a numbered/labelled caption.
 *
 * The diagram itself is expected to be theme-aware SVG that reads the CSS palette variables
 * (fills/strokes via the `.figure-svg` classes in globals.css), so it follows light/dark
 * automatically — see PipelineDiagram for the pattern. `Figure` only owns the framing
 * (centering, max-width, the caption line and accessible labelling).
 *
 * Pass `title` for a short bold lead-in and `caption` for the descriptive line; either may
 * be omitted. `id` enables deep-linking and is wired to the caption via aria.
 */
export interface FigureProps {
  children: ReactNode;
  /** Short bold lead-in shown before the caption text (e.g. "Figure 1"). */
  title?: ReactNode;
  /** Descriptive caption line under the figure. */
  caption?: ReactNode;
  /** Optional id for deep-linking / aria. */
  id?: string;
}

export function Figure({ children, title, caption, id }: FigureProps) {
  const captionId = id ? `${id}-caption` : undefined;
  return (
    <figure className="figure" id={id} aria-describedby={captionId}>
      <div className="figure-canvas">{children}</div>
      {title || caption ? (
        <figcaption className="figure-caption" id={captionId}>
          {title ? <span className="figure-caption-title">{title}</span> : null}
          {title && caption ? " " : null}
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}
