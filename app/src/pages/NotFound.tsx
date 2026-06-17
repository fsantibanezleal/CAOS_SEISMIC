import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

/** Catch-all route for unknown paths. Keeps the honest framing (no alarm language). */
export default function NotFound() {
  const { t } = useTranslation();
  return (
    <article className="page-body">
      <h1>{t("common.notFoundTitle")}</h1>
      <p className="muted">{t("common.notFoundBody")}</p>
      <p>
        <Link to="/">{t("nav.introduction")}</Link>
      </p>
    </article>
  );
}
