/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Söhne"', '"Inter"', 'system-ui', 'sans-serif'],
        serif: ['"Tiempos Text"', '"Source Serif Pro"', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        ink: {
          DEFAULT: '#1a1a1a',
          muted: '#525252',
          subtle: '#737373',
        },
        paper: {
          DEFAULT: '#fafaf7',
          raised: '#ffffff',
        },
        accent: {
          DEFAULT: '#2c5f5d',
          hover: '#1f4644',
        },
        warn: '#b45309',
        review: '#a16207',
      },
    },
  },
  plugins: [],
};
