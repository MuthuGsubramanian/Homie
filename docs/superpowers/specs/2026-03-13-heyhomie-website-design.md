# HeyHomie.app Website Design Spec

## Overview

Marketing website for Homie AI — a fully local, privacy-first personal AI assistant. The site serves as the public face of the project: landing page, about, download, and legal pages. Built with a retro 90s RPG game aesthetic (stat bars, skill cards, quest logs, level indicators) layered with subtle Iron Man / JARVIS HUD references (arc reactor glow, status readouts, HUD borders). The tone is minimal, funky, and comical — approachable for general users with technical depth available for developers.

**Domain:** https://heyhomie.app (currently on GoDaddy Website Builder — will be replaced)

**Audience:** Both technical and non-technical users. Lead with approachable, playful messaging; provide technical depth for those who want it.

## Tech Stack

- **Framework:** Astro (static output, zero JS by default)
- **Styling:** Tailwind CSS
- **Fonts:** Press Start 2P (pixel headings), VT323 (retro body/stats), system sans-serif (legal page body text)
- **Deployment:** Static HTML output — deployable to Vercel, Netlify, GitHub Pages, or any static host
- **Future expansion:** Astro Starlight for docs, React/Svelte islands for dashboard

### Why Astro

Astro ships zero JavaScript by default. Pages are pure static HTML, which gives instant load times for a site that is mostly content. When docs are needed later, Astro has first-class support via `@astrojs/starlight`. When a dashboard is needed, interactive islands can be added per-component without converting the whole site to a SPA. This matches Homie's philosophy: lightweight, no unnecessary bloat.

## Project Structure

```
website/
├── src/
│   ├── layouts/
│   │   └── BaseLayout.astro          # HUD frame, nav, footer, meta tags
│   ├── pages/
│   │   ├── index.astro               # Landing page
│   │   ├── about.astro               # About page
│   │   ├── download.astro            # Download page
│   │   ├── terms.astro               # Terms of Service
│   │   └── privacy.astro             # Privacy Policy
│   ├── components/
│   │   ├── Nav.astro                 # Top nav bar (HUD-style)
│   │   ├── Footer.astro              # Footer with legal links
│   │   ├── StatBar.astro             # Reusable RPG stat bar
│   │   ├── FeatureCard.astro         # Feature as skill card
│   │   ├── HeroSection.astro         # Landing hero with arc reactor
│   │   ├── StatsRow.astro            # Character stats row
│   │   ├── QuestLog.astro            # Getting started steps
│   │   ├── PlatformCard.astro        # OS selection card
│   │   ├── ModuleRow.astro           # Optional module install row
│   │   └── SpecGrid.astro            # System requirements grid
│   └── styles/
│       └── global.css                # Pixel fonts, HUD theme, animations
├── public/
│   └── fonts/                        # Local font files (fallback)
├── astro.config.mjs
├── tailwind.config.mjs
└── package.json
```

## Visual Identity

### Color Palette

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Background | Near-black | `#0d0d1a` | Page background |
| Primary accent | Gold/amber | `#f1c40f` | Headings, arc reactor, CTAs, active nav |
| Success/online | Teal green | `#2ecc71` | Status indicators, "online" states |
| Info | Blue | `#3498db` | Secondary feature accent |
| Special | Purple | `#9b59b6` | Tertiary accent (plugins, inventory) |
| Warning/defeated | Soft red | `#e74c3c` | "Defeated" states, alerts |
| Body text | Light gray | `#bbb` | Paragraph text |
| Muted text | Mid gray | `#666`–`#888` | Secondary labels, descriptions |
| Borders | Gold translucent | `rgba(241,196,15,0.15)` | HUD divider lines |

### Typography

| Element | Font | Size | Style |
|---------|------|------|-------|
| Page titles | Press Start 2P | 22-28px | 900 weight, letter-spacing: 6-8px |
| Status lines | Press Start 2P | 9px | Letter-spacing: 4px, uppercase |
| Stat labels | VT323 / Courier New | 10-11px | Letter-spacing: 2px, uppercase |
| Body copy | VT323 / Courier New | 11-12px | Line-height: 1.8-2.0 |
| Legal body text | System sans-serif | 14-16px | Clean, readable, standard line height |
| Nav links | Courier New | 10px | Letter-spacing: 2px, uppercase |

### Iron Man / JARVIS Grace Notes

These references are subtle — never explicit Marvel branding:

- **Arc reactor logo:** CSS-only concentric circles with gold glow (`box-shadow`), gentle pulse animation. Used in nav and hero.
- **Status readouts:** JARVIS-style language — "SYSTEMS ONLINE", "ALL MODULES OPERATIONAL", "DEPLOYMENT READY", "DOSSIER ACCESS GRANTED", "LEGAL PROTOCOLS ACTIVE".
- **HUD borders:** Thin gold gradient dividers between sections (`linear-gradient` with transparency fade at edges).
- **Navy-to-dark gradients:** Background echoes suit palette.
- **"SUIT UP":** Download page title — subtle nod, natural in context.
- **"FULL SUIT":** Label for `homie-ai[all]` install — playful, deniable.
- **Character profile / dossier framing:** About page treats the creator like a character profile.

## Page Designs

### Landing Page (/)

**Sections in order:**

1. **Nav bar** — Arc reactor logo (CSS glow ring) + "HOMIE" wordmark left, nav links right (HOME, ABOUT, DOWNLOAD). Gold border-bottom. Active link in gold, others muted.

2. **Hero** — Centered layout:
   - Status line: "■ SYSTEMS ONLINE • ALL MODULES OPERATIONAL" (green, typing animation on load)
   - Arc reactor motif: concentric CSS circles with gold glow, pulse animation
   - Title: "HEY HOMIE" (28px, 900 weight, letter-spacing: 8px)
   - Subtitle: "YOUR LOCAL AI COMPANION" (11px, muted)
   - Description: "A fully local, privacy-first AI assistant that runs entirely on your machine. No cloud. No tracking. Just you and your homie."
   - CTA: "▶ GET STARTED" — gold border button, links to /download

3. **Equipped Abilities** — Label: "EQUIPPED ABILITIES"
   - 2x2 grid of feature skill cards, each containing:
     - Feature name (colored, uppercase, letter-spacing)
     - Level indicator ("LVL MAX" or slot count)
     - Description text (muted)
     - Stat bar (colored fill, animated on scroll)
   - Cards: Voice (gold), Memory (green), Privacy (blue), Plugins (purple, "12 SLOTS")

4. **Character Stats** — Label: "CHARACTER STATS"
   - Horizontal row of 4 big numbers:
     - 100% LOCAL | 0 CLOUD CALLS | 20+ MODULES | ∞ PRIVACY

5. **Quest Log** — Label: "QUEST LOG"
   - 3 numbered steps (getting started):
     - 01: INSTALL HOMIE — "pip install homie-ai. That's it. No accounts, no API keys."
     - 02: CHOOSE YOUR MODEL — "Pick a local LLM that fits your GPU. Homie downloads it for you."
     - 03: SAY "HEY HOMIE" — "Voice, text, or hotkey — your homie is always ready."

6. **Footer** — Copyright left, TERMS / PRIVACY / GITHUB links right.

### About Page (/about)

**Sections in order:**

1. **Nav** — same as landing, ABOUT highlighted
2. **Header** — Status: "■ DOSSIER ACCESS GRANTED". Title: "ABOUT HOMIE". Subtitle: "ORIGIN STORY • MISSION • CREATOR"
3. **Origin Story** — Narrative copy:
   - "Every hero needs a companion. Not one that lives in some corporate data center, harvesting your habits and selling your secrets. One that lives with you. On your machine. Under your rules."
   - "Homie was born from a simple idea: what if your AI assistant was actually *yours*? No subscriptions. No cloud dependencies. No 'we updated our privacy policy' emails. Just a smart, local companion that gets better the more you use it."
4. **Mission Parameters** — 4 mission items with colored square bullets:
   - Gold: "PRIVACY IS NOT A FEATURE" — "It's the foundation. Your data stays on your machine, encrypted in a vault even we can't open."
   - Green: "LOCAL MEANS LOCAL" — "Your LLM runs on your GPU. Your voice stays on your mic. Zero cloud calls required."
   - Blue: "OPEN SOURCE, OPEN HEART" — "MPL-2.0 licensed. Read the code, fork the code, make it yours. No vendor lock-in."
   - Purple: "YOUR AI SHOULD KNOW YOU" — "Not to exploit you — to help you. Homie learns your patterns, anticipates your needs, and never shares a byte."
5. **Creator Profile** — Card with gold border:
   - Name: "MSG" (gold) / "MUTHU G. SUBRAMANIAN" (muted)
   - "ONLINE" status badge (green)
   - Bio: "Builder of things that respect people. Believes AI should empower individuals, not corporations. Made Homie because he wanted an AI assistant he could actually trust."
   - Links: GITHUB ↗, EMAIL ↗
6. **Inventory** — Tech stack as 3x2 grid:
   - Python (gold, CORE), llama.cpp (green, ENGINE), SQLite (blue, STORAGE)
   - ChromaDB (purple, VECTORS), Whisper (red, VOICE), AES-256 (orange, VAULT)
7. **Footer**

### Download Page (/download)

**Sections in order:**

1. **Nav** — same, DOWNLOAD highlighted
2. **Header** — Status: "■ DEPLOYMENT READY". Title: "SUIT UP". Subtitle: "SELECT YOUR PLATFORM • INSTALL • LAUNCH"
3. **Quick Deploy** — Terminal-style box:
   - Label: "TERMINAL"
   - Command: `$ pip install homie-ai` (green text)
   - Copy button top-right ("COPY" → "COPIED ✓" on click)
   - Note below: "Works on all platforms. Requires Python 3.11+"
4. **Platform Select** — 3-column grid:
   - Windows (⊞), Linux (🐧), macOS (🍎)
   - JavaScript auto-detects user's OS, highlights with gold border + "DETECTED ✓"
   - Others show muted borders
5. **Minimum Specs** — 3x2 grid:
   - CPU: "Any modern x86/ARM"
   - RAM: "8 GB minimum"
   - GPU: "Optional (CUDA/Metal)" + "+10x speed boost" (green)
   - Disk: "2 GB + model size"
   - Python: "3.11 or newer"
   - Internet: "Only for install" + "then fully offline" (gold)
6. **Optional Modules** — Stacked rows:
   - Voice: `pip install homie-ai[voice]` ~2 GB
   - Email: `pip install homie-ai[email]` ~5 MB
   - Neural: `pip install homie-ai[neural]` ~50 MB
   - Everything: `pip install homie-ai[all]` — "FULL SUIT" (gold highlight)
7. **Footer**

### Terms of Service (/terms)

**Visual treatment:** Same dark theme and nav/footer. Header: status "■ LEGAL PROTOCOLS ACTIVE", title "TERMS OF SERVICE". Body text uses system sans-serif for readability. Section headings stay in retro monospace.

**Content sections:**

1. **Acceptance** — By downloading or using Homie, you agree to these terms.
2. **License** — MPL-2.0. Plain-English explanation: you can use, modify, and distribute Homie. Modifications to MPL-licensed files must remain open source. You can combine Homie with proprietary code in a larger project.
3. **What Homie Is** — Local software that runs on your machine. Not a cloud service. Not a SaaS product. We do not host, operate, or control your Homie instance.
4. **Your Responsibilities** — You are responsible for your hardware, your data, and the models you choose to run. You are responsible for complying with model licenses (e.g., Llama, Mistral, Qwen community licenses).
5. **No Warranty** — Homie is alpha software, provided as-is. No guarantees of fitness, availability, or correctness.
6. **Third-Party Models** — Language models you download through Homie have their own licenses and terms. We do not control or take responsibility for third-party model outputs.
7. **Optional Cloud Connections** — If you choose to connect Gmail, social media, or other external services, those providers' terms of service apply. Homie facilitates the connection; we do not intermediary or store that data on our servers.
8. **Limitation of Liability** — To the maximum extent permitted by law, Hey Homie and its contributors shall not be liable for any indirect, incidental, special, consequential, or punitive damages.
9. **Changes to Terms** — We may update these terms. Changes will be posted on this page with an updated effective date. No email spam.
10. **Contact** — muthu.g.subramanian@outlook.com

### Privacy Policy (/privacy)

**Visual treatment:** Same as Terms page.

**Content sections:**

1. **The Short Version** — "We don't collect your data. Period. Homie runs entirely on your machine. We have no servers, no analytics pipeline, no way to see what you do with Homie."
2. **What Data Homie Processes Locally** — Observations (work patterns, routines), memory (working, episodic, semantic), voice (STT/TTS), emails (if Gmail connected), browser history (if enabled), financial reminders (if enabled). All processed and stored locally in `~/.homie/`.
3. **How Local Data Is Stored** — SQLite databases + ChromaDB vector store. Sensitive data (credentials, tokens, personal info) encrypted with AES-256-GCM in a vault protected by OS keyring (Windows Credential Manager, macOS Keychain, Linux Secret Service) with PBKDF2 password layer.
4. **What We (Hey Homie the Organization) Collect** — Website: standard web server logs (IP, user agent, pages visited) if hosted on a platform that collects them. We do not run analytics or tracking scripts. The Homie application: zero telemetry, zero crash reporting, zero phone-home. There is no code in Homie that transmits data to us or anyone.
5. **Optional Cloud Connections** — Gmail: OAuth 2.0, tokens stored in local vault. Emails fetched and processed locally. Social media: API tokens stored in local vault. Data flows between your machine and the platform — never through us.
6. **Voice Data** — Processed locally via faster-whisper (STT) and Piper/Kokoro/MeloTTS (TTS). Audio never leaves your machine. No recordings are stored unless you explicitly enable logging.
7. **No Tracking, No Telemetry** — No analytics SDK. No crash reporting. No update checks that transmit usage data. Homie does not know how many users it has.
8. **Data Retention & Control** — Configurable retention period (default: 30 days). Maximum storage limit (default: 512 MB). Delete everything: remove `~/.homie/`. Export: `homie backup --to <path>` creates an encrypted archive. You own your data directory — we never touch it.
9. **Your Rights** — You have full control. View, export, or delete any data at any time. No request needed — it is your filesystem.
10. **Children's Privacy** — Homie is not designed for children under 13. We do not knowingly process data of children.
11. **Changes to This Policy** — Updates posted on this page with a new effective date.
12. **Contact** — muthu.g.subramanian@outlook.com

## Responsive Design

### Breakpoints

| Breakpoint | Width | Layout Changes |
|-----------|-------|----------------|
| Mobile | <768px | Single column. Nav collapses to hamburger (pixel "≡"). Skill cards stack. Stats wrap to 2x2. Platform cards stack. |
| Tablet | 768-1024px | 2-column grid for skill cards. Stats stay in row. Platform cards stay in row. |
| Desktop | 1024px+ | Full layout as designed. Max content width ~800px centered. |

### Mobile Nav

Hamburger icon styled as pixel "≡" in gold. Opens a full-screen overlay with nav links stacked vertically, monospace, centered. Close button as pixel "✕".

## Micro-Interactions

All CSS-only except where noted:

| Element | Animation | Implementation |
|---------|-----------|----------------|
| Arc reactor logo | Gentle pulse glow | CSS `@keyframes` on `box-shadow` |
| "SYSTEMS ONLINE" | Typing effect on load | CSS `steps()` animation on `width` |
| Stat bars | Fill from 0 to value | CSS `@keyframes` + JS `IntersectionObserver` to trigger on scroll |
| Skill cards | Border glow on hover | CSS `transition` on `border-color` and `box-shadow` |
| Nav links | Gold underline slide-in | CSS `::after` pseudo-element with `transform: scaleX()` transition |
| CTA buttons | Border pulse on hover | CSS `@keyframes` on `border-color` |
| Copy button | Text swap on click | Vanilla JS, ~10 lines |
| Platform detect | Highlight user's OS | Vanilla JS `navigator.userAgentData?.platform` with `navigator.platform` fallback, ~20 lines |

## Performance Targets

- Zero JS shipped by default (Astro static)
- Vanilla JS only where needed: copy button, platform detection, stat bar scroll trigger
- Total page weight: <100KB per page (excluding fonts)
- Fonts: Self-hosted in `public/fonts/` with `font-display: swap` (no external CDN dependency)
- All pages static HTML — no SSR, no hydration

## SEO

- `<title>` and `<meta name="description">` per page
- Open Graph tags (og:title, og:description, og:image) per page
- Twitter card meta tags
- JSON-LD structured data for SoftwareApplication on landing/download pages
- Semantic HTML: `<header>`, `<nav>`, `<main>`, `<article>`, `<section>`, `<footer>`
- Canonical URLs
- Sitemap.xml (Astro generates automatically)

## Future Expansion

These are not in scope for v1 but inform architectural decisions:

- **/docs** — Astro Starlight integration for documentation
- **Web dashboard** — React/Svelte island for remote Homie management
- **Blog** — Markdown-based blog posts with Astro content collections
- **Themes** — Additional color schemes (light mode, different retro palettes)
