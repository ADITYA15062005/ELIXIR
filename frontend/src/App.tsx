import { useEffect, useState, type FormEvent } from "react";
import {
  fetchHealth,
  postRecommend,
  type HealthResponse,
  type RecommendResponse,
} from "./api";

type LoadState = "idle" | "loading" | "done" | "error";

export default function App() {
  const [query, setQuery] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [submitState, setSubmitState] = useState<LoadState>("idle");
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const h = await fetchHealth();
        if (!cancelled) {
          setHealth(h);
          setHealthError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setHealth(null);
          setHealthError(err instanceof Error ? err.message : "Health check failed");
        }
      }
    };
    void load();
    const id = window.setInterval(load, 30_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setSubmitState("loading");
    setSubmitError(null);
    setResult(null);
    try {
      const data = await postRecommend(q);
      setResult(data);
      setSubmitState("done");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Recommend failed");
      setSubmitState("error");
    }
  }

  const indexLabel =
    health?.index_version ??
    (healthError ? "unreachable" : health?.index_loaded === false ? "none" : "…");

  const statusTone: "ok" | "warn" | "bad" = healthError
    ? "bad"
    : health?.status === "ok" && health.index_loaded
      ? "ok"
      : "warn";

  return (
    <div className="page">
      <div className="atmosphere" aria-hidden="true" />
      <header className="topbar">
        <p className="brand-mark">Elixir</p>
        <div className={`health-chip tone-${statusTone}`} title={healthError ?? undefined}>
          <span className="health-dot" />
          <span>
            {healthError
              ? "API offline"
              : health?.index_loaded
                ? `Index ${indexLabel}`
                : `Degraded · ${indexLabel}`}
          </span>
        </div>
      </header>

      <main>
        <section className="hero">
          <h1 className="brand-hero">Elixir</h1>
          <p className="headline">Taste, then discover.</p>
          <p className="support">
            Describe what you want in a glass — oak, citrus, spice, budget — and
            get a recommendation grounded in the catalog index.
          </p>

          <form className="recommend-form" onSubmit={onSubmit}>
            <label className="sr-only" htmlFor="taste-query">
              Taste query
            </label>
            <input
              id="taste-query"
              className="query-input"
              type="text"
              autoComplete="off"
              placeholder="e.g. crisp mineral white under ₹2000"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={submitState === "loading"}
            />
            <button
              className="cta"
              type="submit"
              disabled={submitState === "loading" || !query.trim()}
            >
              {submitState === "loading" ? "Searching…" : "Recommend"}
            </button>
          </form>
        </section>

        <section className="result-region" aria-live="polite">
          {submitError && (
            <p className="result-error" role="alert">
              {submitError}
            </p>
          )}
          {result && (
            <article className="answer-panel">
              <header className="answer-meta">
                <span>Recommendation</span>
                <span>
                  k={result.k}
                  {result.index_version ? ` · ${result.index_version}` : ""}
                </span>
              </header>
              <p className="answer-body">{result.answer}</p>
              <p className="answer-query">For “{result.query}”</p>
            </article>
          )}
        </section>
      </main>

      <footer className="footer">
        <span>Product UI · FastAPI recommend</span>
        <span>Stitch assets pending · Apple-inspired brief</span>
      </footer>
    </div>
  );
}
