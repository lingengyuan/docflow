# Design System Specification: The Ethereal Canvas

## 1. Overview & Creative North Star
This design system is built upon the **Creative North Star: The Silent Architect.** It moves away from the rigid, grid-locked density of traditional enterprise software and toward an editorial, ambient experience. Inspired by the spatial clarity of Arc and the functional purity of Linear, the system treats the screen not as a container for data, but as a series of layered, light-filled planes.

We achieve a "High-End" feel by embracing **Intentional Asymmetry**. Do not feel obligated to center-align every element; use large, purposeful "breathing rooms" (Scale 16-24) to frame content. The goal is a UI that feels curated, not generated.

---

## 2. Colors & Tonal Depth
The palette is rooted in low-energy, sophisticated cool tones. We use color not for decoration, but to define physical space.

### The "No-Line" Rule
**Explicit Instruction:** Use of `1px solid` borders for sectioning is prohibited. Boundaries must be defined solely through background color shifts or tonal transitions. To separate a sidebar from a main content area, use `surface_container_low` against `surface`.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers. Use the following tiers to create depth:
- **Base Layer:** `surface` (#f7f9fb) – The infinite canvas.
- **Mid Layer:** `surface_container_low` (#f0f4f7) – Used for secondary navigation or sidebars.
- **Top Layer:** `surface_container_lowest` (#ffffff) – Used for primary content cards or floating modals to create a "lifted" effect.

### The "Glass & Gradient" Rule
For floating elements (Command Palettes, Popovers), use **Glassmorphism**. Combine `surface_container_lowest` at 80% opacity with a `backdrop-blur` of 20px. 
*   **Signature Texture:** Use a subtle linear gradient on primary Action Buttons: `primary` (#515f74) to `primary_dim` (#455368) at a 135-degree angle. This adds a "weighted" premium feel that flat hex codes lack.

---

## 3. Typography: The Editorial Voice
We prioritize a system font stack to ensure the UI feels like a native extension of the OS, specifically optimized for Chinese legibility.

- **Font Stack:** `-apple-system, "PingFang SC", "Noto Sans SC", "Inter", sans-serif`.
- **Hierarchy as Identity:**
    - **Display & Headlines:** Use `display-md` (2.75rem) for landing states with `-0.02em` letter spacing. This "tight" tracking provides an authoritative, premium look.
    - **Body:** Use `body-md` (0.875rem) for all functional text. Ensure a line-height of `1.6` to maintain the "Notion-esque" readability.
    - **Labels:** Use `label-md` in `on_surface_variant` (#566166) with `0.05em` letter spacing and All-Caps for metadata to differentiate it from actionable text.

---

## 4. Elevation & Depth
Depth is a psychological cue for hierarchy. We use **Tonal Layering** over structural lines.

- **The Layering Principle:** Place a `surface_container_lowest` card on a `surface_container` background. The slight shift in luminosity creates a soft, natural edge.
- **Ambient Shadows:** For floating elements, use a "Cloud Shadow": 
  `box-shadow: 0 12px 40px rgba(42, 52, 57, 0.06);` 
  The shadow color is derived from `on_surface` to mimic natural light.
- **The "Ghost Border" Fallback:** If a border is required for accessibility (e.g., input fields), use `outline_variant` at **15% opacity**. Never use 100% opacity borders.

---

## 5. Components

### Buttons
- **Primary:** `primary` background, `on_primary` text. Radius: `md` (0.75rem). No border.
- **Secondary:** `secondary_container` background, `on_secondary_container` text.
- **Tertiary:** No background. Text in `primary`. Interaction state uses a `surface_container_high` ghost background on hover.

### Cards & Lists
**Rule:** Forbid the use of divider lines. 
- Separate list items using `spacing-2` (0.7rem) of vertical white space.
- For hover states, shift the background to `surface_container_highest` (#d9e4ea) rather than showing an outline.

### Input Fields
- **Default State:** `surface_container_lowest` background with a 15% `outline_variant` ghost border.
- **Focus State:** Increase shadow spread and change the ghost border to `primary` at 40% opacity. Transition should be `200ms ease-out`.

### Chips (Action & Filter)
- Use `tertiary_container` (pale indigo/mint) for active states to provide a calm, non-aggressive "selected" cue.
- Shape: `full` (9999px) for a "pill" look that contrasts against the `md` (0.75rem) radius of the main containers.

---

## 6. Do's and Don'ts

### Do
- **DO** use the `spacing-16` (5.5rem) and `spacing-20` (7rem) values for top-level page margins. Large margins signal luxury and focus.
- **DO** use clean, SVG linear icons with a `1.5px` stroke weight to match the "Linear" aesthetic.
- **DO** use `surface_bright` to highlight active zones without adding visual weight.

### Don't
- **DON'T** use emojis. They disrupt the "Calm & Professional" personality. Use icons or typography to convey emotion.
- **DON'T** use pure black (#000000). Always use `on_background` (#2a3439) for text to maintain a soft, paper-like contrast.
- **DON'T** use hard-edged 0px corners. Every element must adhere to the `8px-12px` (DEFAULT to md) radius scale to feel approachable.