import { createBrowserRouter, RouterProvider } from "react-router-dom";

import App from "@/App";
import NotFound from "@/pages/NotFound";
import Introduction from "@/pages/Introduction";
import Problem from "@/pages/Problem";
import Methodology from "@/pages/Methodology";
import Implementation from "@/pages/Implementation";
import BackAnalysis from "@/pages/BackAnalysis";
import Monitoring from "@/pages/Monitoring";

/**
 * The six product routes (web-app-spec.md §1), wrapped by the App layout shell.
 *
 * Paths mirror src/lib/routes.ts (the nav reads the same list), so the header navigation
 * and the router can never drift. The basename is the Vite deploy base so the app works
 * under a sub-path (e.g. GitHub Pages `/CAOS_SEISMIC/`).
 *
 * Pages are imported eagerly for now — every route is a light text scaffold. When the real
 * Monitoring content lands (MapLibre + deck.gl), it is code-split behind that route per the
 * bundle discipline in web-app-spec.md §7.2.
 */
const router = createBrowserRouter(
  [
    {
      path: "/",
      element: <App />,
      children: [
        { index: true, element: <Introduction /> },
        { path: "problem", element: <Problem /> },
        { path: "methodology", element: <Methodology /> },
        { path: "implementation", element: <Implementation /> },
        { path: "back-analysis", element: <BackAnalysis /> },
        { path: "monitoring", element: <Monitoring /> },
        { path: "*", element: <NotFound /> },
      ],
    },
  ],
  { basename: import.meta.env.BASE_URL },
);

/** Top-level router host mounted by main.tsx. */
export default function AppRouter() {
  return <RouterProvider router={router} />;
}
