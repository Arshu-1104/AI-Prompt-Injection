import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        safe: { DEFAULT: "#1B6CA8", light: "#E8F4FD", dark: "#0D3D63" },
        suspicious: { DEFAULT: "#B45309", light: "#FEF3C7", dark: "#6B3A07" },
        malicious: { DEFAULT: "#B91C1C", light: "#FEE2E2", dark: "#6B0F0F" },
      },
    },
  },
  plugins: [],
};

export default config;
