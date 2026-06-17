// Ambient type declarations for `react-katex` (no bundled @types, none on npm).
//
// Only the two components this project uses are declared. KaTeX rendering options are passed
// through `settings`; we keep the surface minimal and typed enough for strict mode.
declare module "react-katex" {
  import type { ComponentType } from "react";

  export interface MathComponentProps {
    /** The LaTeX/TeX string to render. */
    math: string;
    /** Render a fallback element on parse error instead of throwing. */
    errorColor?: string;
    renderError?: (error: Error) => React.ReactNode;
    /** KaTeX settings passthrough. */
    settings?: Record<string, unknown>;
    as?: string | ComponentType<unknown>;
  }

  export const InlineMath: ComponentType<MathComponentProps>;
  export const BlockMath: ComponentType<MathComponentProps>;
}
