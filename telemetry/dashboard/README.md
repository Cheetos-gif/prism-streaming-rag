# telemetry/dashboard

Single-file dashboard SPA, served as static files by the FastAPI app at
`/dashboard/` (`controller/main.py`).

| File | Tracked | Purpose |
| :--- | :--- | :--- |
| `index.html` | yes | Markup plus the one inline script that talks to the API |
| `styles.css` | yes | **Generated** — the only stylesheet the page loads |
| `tailwind.config.js` | yes | Design tokens (the "Obsidian Prism" palette) and content globs |
| `tailwind.input.css` | yes | `@tailwind` directives plus the hand-written page CSS |
| `package.json`, `package-lock.json` | yes | Build-only tooling, pinned to the versions the Play CDN used to serve |
| `node_modules/` | no | Installed by `make css`, git- and docker-ignored |

## Rebuilding the stylesheet

```bash
make css         # needs Node, writes telemetry/dashboard/styles.css
make css-check   # same build, fails when the committed file is stale (CI runs this)
```

Edit `index.html`, `tailwind.config.js` or `tailwind.input.css`, run `make css`,
and commit `styles.css`. Tailwind reads `index.html` as plain text, so classes
that only appear inside the inline script's template strings are picked up too.

## Why the stylesheet is precompiled

`index.html` used to load the Tailwind Play CDN and configure it at runtime.
That is what broke in production: the deployment sits behind Cloudflare with
Rocket Loader enabled, which rewrites every `<script>` to a deferred blob. On
the deployed page the CDN script still executes, but only Preflight is emitted —
none of the utilities — so the site renders unstyled while `localhost`, where
the scripts run inline in document order, looks fine. Verified by comparing the
generated CSSOM: 64 rules (reset only) in production against 288 locally.

A precompiled stylesheet cannot fail that way: styling is a plain
`<link rel="stylesheet">` and needs no script execution, no network access to a
third-party CDN, and no flash of unstyled content. Cloudflare's warnings about
the Play CDN are gone with it.

Google Fonts (`Inter`, `Plus Jakarta Sans`, `JetBrains Mono`,
`Material Symbols Outlined`) are still loaded from the CDN; they degrade to
system fonts if unreachable, which does not affect layout.
