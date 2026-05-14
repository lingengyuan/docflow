const colorVar = variable => ({ opacityValue }) => {
  if (opacityValue === undefined) return `rgb(var(${variable}))`;
  return `rgb(var(${variable}) / ${opacityValue})`;
};

module.exports = {
  content: ["./frontend/index.html", "./frontend/partials/**/*.html", "./frontend/js/**/*.js"],
  theme: {
    extend: {
      colors: {
        "surface-dim": colorVar("--color-surface-dim"),
        background: colorVar("--color-background"),
        "on-primary-container": colorVar("--color-on-primary-container"),
        "surface-bright": colorVar("--color-surface-bright"),
        outline: colorVar("--color-outline"),
        "on-primary": colorVar("--color-on-primary"),
        "on-tertiary": colorVar("--color-on-tertiary"),
        "on-surface-variant": colorVar("--color-on-surface-variant"),
        "tertiary-container": colorVar("--color-tertiary-container"),
        "surface-container-highest": colorVar("--color-surface-container-highest"),
        "on-tertiary-container": colorVar("--color-on-tertiary-container"),
        "primary-dim": colorVar("--color-primary-dim"),
        tertiary: colorVar("--color-tertiary"),
        primary: colorVar("--color-primary"),
        "on-secondary-container": colorVar("--color-on-secondary-container"),
        "surface-container": colorVar("--color-surface-container"),
        "primary-container": colorVar("--color-primary-container"),
        "surface-container-high": colorVar("--color-surface-container-high"),
        "inverse-primary": colorVar("--color-inverse-primary"),
        "on-background": colorVar("--color-on-background"),
        "surface-container-low": colorVar("--color-surface-container-low"),
        surface: colorVar("--color-surface"),
        "on-surface": colorVar("--color-on-surface"),
        "surface-container-lowest": colorVar("--color-surface-container-lowest"),
        "outline-variant": colorVar("--color-outline-variant"),
        error: colorVar("--color-error"),
        "on-error": colorVar("--color-on-error"),
        "secondary-container": colorVar("--color-secondary-container"),
      },
      fontFamily: {
        sans: ["Avenir Next", "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "sans-serif"],
      },
      borderRadius: {
        DEFAULT: "0.5rem",
        lg: "0.75rem",
        xl: "1rem",
        full: "9999px",
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
    require("@tailwindcss/container-queries"),
  ],
};
