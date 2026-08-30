export type HealthResponse = {
  status: string;
  index_loaded: boolean;
  index_version: string | null;
};

export type RecommendResponse = {
  answer: string;
  query: string;
  k: number;
  index_version: string;
  sources: Record<string, unknown>[];
};

export type ApiErrorBody = {
  detail?: string | { msg?: string }[];
};

/** Empty string = same-origin / Vite proxy. Override with VITE_API_BASE. */
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

function url(path: string): string {
  return `${API_BASE}${path}`;
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as ApiErrorBody;
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((d) => d.msg ?? JSON.stringify(d)).join("; ");
    }
  } catch {
    /* ignore */
  }
  return res.statusText || `HTTP ${res.status}`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(url("/health"));
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<HealthResponse>;
}

export async function postRecommend(
  query: string,
  k?: number,
): Promise<RecommendResponse> {
  const res = await fetch(url("/v1/recommend"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, ...(k != null ? { k } : {}) }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<RecommendResponse>;
}
