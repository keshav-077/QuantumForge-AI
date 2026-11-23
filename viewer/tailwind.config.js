/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#FAF9F7",
        surface: "#FFFFFF",
        subtle: "#F4F3F0",
        hairline: "#E5E7EB",
        ink: {
          primary: "#111827",
          secondary: "#4B5563",
          muted: "#6B7280",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
