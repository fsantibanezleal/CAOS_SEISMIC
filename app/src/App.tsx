import { Outlet } from "react-router-dom";

import Layout from "@/components/Layout";

/**
 * The application layout shell. Renders the persistent header/footer (Layout) around the
 * active route's element (`<Outlet />`). The router (router.tsx) mounts this as the root
 * element and nests the six page routes beneath it.
 */
export default function App() {
  return (
    <Layout>
      <Outlet />
    </Layout>
  );
}
