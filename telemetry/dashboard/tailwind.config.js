/**
 * Tailwind configuration for the PRISM dashboard.
 *
 * The design tokens below are the "Obsidian Prism" palette used by
 * telemetry/dashboard/index.html. styles.css is pre-compiled from this file
 * (see `make css`); the page must never depend on the Tailwind Play CDN at
 * runtime — see the note in telemetry/dashboard/README.md.
 */
const path = require("path");

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [path.join(__dirname, "index.html")],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "on-primary-fixed": "#001f26",
        "inverse-on-surface": "#2f3036",
        "tertiary-fixed-dim": "#4edea3",
        "surface-container-high": "#292a2f",
        "surface-bright": "#38393f",
        "on-surface": "#e3e1e9",
        "on-tertiary-container": "#00452e",
        "on-tertiary": "#003824",
        "on-error": "#690005",
        error: "#ffb4ab",
        "on-error-container": "#ffdad6",
        outline: "#869397",
        "secondary-container": "#3131c0",
        "on-secondary-fixed-variant": "#2f2ebe",
        tertiary: "#4edea3",
        background: "#090a0f",
        "on-primary": "#003640",
        "error-container": "#93000a",
        "on-tertiary-fixed": "#002113",
        "surface-container": "#181a22",
        "on-secondary": "#1000a9",
        "inverse-surface": "#e3e1e9",
        "on-secondary-fixed": "#07006c",
        "surface-dim": "#0e1017",
        primary: "#4cd7f6",
        "surface-variant": "#232631",
        surface: "#0d0f16",
        "surface-container-low": "#13151e",
        "surface-tint": "#4cd7f6",
        "on-primary-container": "#00424f",
        "primary-container": "#06b6d4",
        "on-primary-fixed-variant": "#004e5c",
        "on-background": "#e3e1e9",
        "on-surface-variant": "#bcc9cd",
        "secondary-fixed-dim": "#c0c1ff",
        "primary-fixed-dim": "#4cd7f6",
        "tertiary-fixed": "#6ffbbe",
        secondary: "#818cf8",
        "surface-container-highest": "#2e313d",
        "on-tertiary-fixed-variant": "#005236",
        "tertiary-container": "#1bbd85",
        "inverse-primary": "#00687a",
        "secondary-fixed": "#e1e0ff",
        "primary-fixed": "#acedff",
        "on-secondary-container": "#b0b2ff",
        "outline-variant": "#343845",
        "surface-container-lowest": "#07080d",
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "sans-serif"],
        display: ["Plus Jakarta Sans", "Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [require("@tailwindcss/forms"), require("@tailwindcss/container-queries")],
};
