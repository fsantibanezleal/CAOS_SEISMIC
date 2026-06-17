import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";

/**
 * Thin wrappers around react-katex so pages render the field's REAL governing equations
 * (Gutenberg–Richter, Omori–Utsu, the ETAS conditional intensity, the exceedance map, the
 * CSEP information gain). KaTeX CSS is imported here once; importing it in this module keeps
 * the dependency local to the (code-split) content components.
 *
 * `BlockEquation` adds an optional caption line under the display math — used to attach the
 * one-line "what this encodes / which reference" note the methodology page needs.
 */

export function Inline({ math }: { math: string }) {
  return <InlineMath math={math} />;
}

export interface BlockEquationProps {
  math: string;
  /** Optional caption rendered under the equation (e.g. the reference / what it encodes). */
  caption?: React.ReactNode;
  /** Optional id for deep-linking / aria. */
  id?: string;
}

export function BlockEquation({ math, caption, id }: BlockEquationProps) {
  return (
    <figure className="equation" id={id}>
      <BlockMath math={math} />
      {caption ? <figcaption className="equation-caption">{caption}</figcaption> : null}
    </figure>
  );
}
