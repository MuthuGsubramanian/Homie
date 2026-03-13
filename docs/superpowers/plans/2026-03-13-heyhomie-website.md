# HeyHomie.app Website Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy the heyhomie.app marketing website with landing, about, download, terms, and privacy pages using a retro 90s RPG game aesthetic with subtle Iron Man/JARVIS HUD references.

**Architecture:** Astro static site with Tailwind CSS, self-hosted retro pixel fonts, zero JS by default. Components are Astro `.astro` files — no React/Svelte needed for v1. All pages share a BaseLayout with HUD-style nav and footer.

**Tech Stack:** Astro 5.x, Tailwind CSS 4.x, TypeScript, self-hosted Press Start 2P + VT323 fonts

**Spec:** `docs/superpowers/specs/2026-03-13-heyhomie-website-design.md`

---

## Chunk 1: Project Scaffold & Global Theme

### Task 1: Initialize Astro Project

**Files:**
- Create: `website/package.json`
- Create: `website/astro.config.mjs`
- Create: `website/tsconfig.json`
- Create: `website/tailwind.config.mjs`

- [ ] **Step 1: Create the website directory and initialize Astro**

```bash
cd C:/Users/muthu/PycharmProjects/Homie
mkdir website && cd website
npm create astro@latest -- . --template minimal --no-install --no-git --typescript strict
```

- [ ] **Step 2: Install dependencies**

```bash
cd C:/Users/muthu/PycharmProjects/Homie/website
npm install
npx astro add tailwind -y
```

- [ ] **Step 3: Verify the dev server starts**

```bash
cd C:/Users/muthu/PycharmProjects/Homie/website
npm run dev
```

Expected: Server starts on localhost:4321, default Astro page loads.

- [ ] **Step 4: Commit**

```bash
cd C:/Users/muthu/PycharmProjects/Homie
git add website/
git commit -m "feat(website): initialize Astro project with Tailwind CSS"
```

---

### Task 2: Self-Hosted Fonts & Global CSS

**Files:**
- Create: `website/public/fonts/PressStart2P-Regular.woff2`
- Create: `website/public/fonts/VT323-Regular.woff2`
- Create: `website/src/styles/global.css`

- [ ] **Step 1: Download fonts**

Download Press Start 2P and VT323 from Google Fonts as woff2 files into `website/public/fonts/`.

```bash
cd C:/Users/muthu/PycharmProjects/Homie/website/public
mkdir -p fonts
# Download from Google Fonts API:
curl -o fonts/PressStart2P-Regular.woff2 "https://fonts.gstatic.com/s/pressstart2p/v15/e3t4euO8T-267oIAQAu6jDQyK3nVivM.woff2"
curl -o fonts/VT323-Regular.woff2 "https://fonts.gstatic.com/s/vt323/v17/pxiKyp0ihIEF2isfFJA.woff2"
```

- [ ] **Step 2: Create global.css with font faces, color variables, and base theme**

Create `website/src/styles/global.css`:

```css
/* === Font Faces === */
@font-face {
  font-family: 'Press Start 2P';
  src: url('/fonts/PressStart2P-Regular.woff2') format('woff2');
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}

@font-face {
  font-family: 'VT323';
  src: url('/fonts/VT323-Regular.woff2') format('woff2');
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}

/* === CSS Variables === */
:root {
  --bg-primary: #0d0d1a;
  --bg-card: rgba(241, 196, 15, 0.03);
  --accent-gold: #f1c40f;
  --accent-green: #2ecc71;
  --accent-blue: #3498db;
  --accent-purple: #9b59b6;
  --accent-red: #e74c3c;
  --accent-orange: #f39c12;
  --text-primary: #ffffff;
  --text-body: #bbbbbb;
  --text-muted: #888888;
  --text-dim: #666666;
  --text-faint: #555555;
  --text-ghost: #333333;
  --border-gold: rgba(241, 196, 15, 0.15);
  --border-subtle: rgba(255, 255, 255, 0.06);
  --font-pixel: 'Press Start 2P', monospace;
  --font-retro: 'VT323', 'Courier New', monospace;
  --font-legal: system-ui, -apple-system, sans-serif;
}

/* === Base === */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  background: var(--bg-primary);
  color: var(--text-body);
  font-family: var(--font-retro);
  font-size: 14px;
  line-height: 1.8;
  min-height: 100vh;
}

/* === HUD Divider === */
.hud-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-gold), transparent);
  margin: 0 24px;
}

/* === Section Label === */
.section-label {
  font-family: var(--font-pixel);
  font-size: 9px;
  letter-spacing: 4px;
  text-align: center;
  margin-bottom: 24px;
  text-transform: uppercase;
}

/* === Status Line === */
.status-line {
  font-family: var(--font-pixel);
  font-size: 9px;
  color: var(--accent-green);
  letter-spacing: 4px;
  text-align: center;
  margin-bottom: 16px;
}

/* === Page Title === */
.page-title {
  font-family: var(--font-pixel);
  font-size: 22px;
  font-weight: 900;
  letter-spacing: 6px;
  color: var(--text-primary);
  text-align: center;
  margin-bottom: 8px;
}

/* === Page Subtitle === */
.page-subtitle {
  font-family: var(--font-retro);
  font-size: 10px;
  color: var(--text-dim);
  letter-spacing: 2px;
  text-align: center;
  text-transform: uppercase;
}

/* === Arc Reactor Pulse === */
@keyframes reactor-pulse {
  0%, 100% { box-shadow: 0 0 12px rgba(241, 196, 15, 0.3); }
  50% { box-shadow: 0 0 20px rgba(241, 196, 15, 0.5); }
}

.reactor-glow {
  animation: reactor-pulse 3s ease-in-out infinite;
}

/* === Typing Animation === */
@keyframes typing {
  from { width: 0; }
  to { width: 100%; }
}

.typing-effect {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  animation: typing 1.5s steps(40) forwards;
}

/* === Stat Bar Fill Animation === */
@keyframes fill-bar {
  from { width: 0; }
}

.stat-bar-fill {
  animation: fill-bar 1s ease-out forwards;
}

.stat-bar-fill.animate {
  animation: fill-bar 1s ease-out forwards;
}

/* === Button Hover === */
.cta-button {
  border: 2px solid var(--accent-gold);
  color: var(--accent-gold);
  padding: 10px 28px;
  font-family: var(--font-retro);
  font-size: 14px;
  letter-spacing: 3px;
  background: transparent;
  cursor: pointer;
  transition: background 0.2s, color 0.2s;
  text-decoration: none;
  display: inline-block;
}

.cta-button:hover {
  background: var(--accent-gold);
  color: var(--bg-primary);
}

/* === Card Hover === */
.skill-card {
  border: 1px solid var(--border-subtle);
  padding: 16px;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.skill-card:hover {
  border-color: var(--border-gold);
  box-shadow: 0 0 12px rgba(241, 196, 15, 0.1);
}
```

- [ ] **Step 3: Verify fonts load and theme applies**

Start dev server and check browser — page should show dark background, fonts should load.

- [ ] **Step 4: Commit**

```bash
git add website/public/fonts/ website/src/styles/
git commit -m "feat(website): add self-hosted retro fonts and global HUD theme CSS"
```

---

### Task 3: BaseLayout, Nav, and Footer Components

**Files:**
- Create: `website/src/layouts/BaseLayout.astro`
- Create: `website/src/components/Nav.astro`
- Create: `website/src/components/Footer.astro`

- [ ] **Step 1: Create Nav.astro**

```astro
---
// Nav.astro — HUD-style navigation bar
interface Props {
  activePage?: string;
}
const { activePage = 'home' } = Astro.props;
const links = [
  { name: 'HOME', href: '/', id: 'home' },
  { name: 'ABOUT', href: '/about', id: 'about' },
  { name: 'DOWNLOAD', href: '/download', id: 'download' },
];
---

<nav class="flex justify-between items-center px-6 py-4 border-b" style="border-color: var(--border-gold);">
  <a href="/" class="flex items-center gap-3 no-underline">
    <div class="reactor-glow" style="width: 28px; height: 28px; border-radius: 50%; border: 2px solid var(--accent-gold); display: flex; align-items: center; justify-content: center;">
      <div style="width: 10px; height: 10px; border-radius: 50%; background: var(--accent-gold); box-shadow: 0 0 8px var(--accent-gold);"></div>
    </div>
    <span style="font-family: var(--font-pixel); font-size: 14px; letter-spacing: 3px; color: var(--accent-gold);">HOMIE</span>
  </a>

  <!-- Desktop links -->
  <div class="hidden md:flex gap-5">
    {links.map(link => (
      <a
        href={link.href}
        class="nav-link no-underline"
        style={`font-family: var(--font-retro); font-size: 13px; letter-spacing: 2px; color: ${activePage === link.id ? 'var(--accent-gold)' : 'var(--text-dim)'};`}
      >
        {link.name}
      </a>
    ))}
  </div>

  <!-- Mobile hamburger -->
  <button id="mobile-menu-btn" class="md:hidden" style="font-family: var(--font-pixel); font-size: 16px; color: var(--accent-gold); background: none; border: none; cursor: pointer;">
    ≡
  </button>
</nav>

<!-- Mobile overlay -->
<div id="mobile-menu" class="fixed inset-0 z-50 hidden" style="background: var(--bg-primary);">
  <div class="flex justify-end p-6">
    <button id="mobile-menu-close" style="font-family: var(--font-pixel); font-size: 16px; color: var(--accent-gold); background: none; border: none; cursor: pointer;">✕</button>
  </div>
  <div class="flex flex-col items-center justify-center gap-8" style="min-height: 60vh;">
    {links.map(link => (
      <a
        href={link.href}
        class="no-underline"
        style={`font-family: var(--font-pixel); font-size: 14px; letter-spacing: 4px; color: ${activePage === link.id ? 'var(--accent-gold)' : 'var(--text-muted)'};`}
      >
        {link.name}
      </a>
    ))}
  </div>
</div>

<script>
  const btn = document.getElementById('mobile-menu-btn');
  const menu = document.getElementById('mobile-menu');
  const close = document.getElementById('mobile-menu-close');
  btn?.addEventListener('click', () => menu?.classList.remove('hidden'));
  close?.addEventListener('click', () => menu?.classList.add('hidden'));
</script>

<style>
  .nav-link {
    position: relative;
  }
  .nav-link::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 100%;
    height: 1px;
    background: var(--accent-gold);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.2s;
  }
  .nav-link:hover::after {
    transform: scaleX(1);
  }
</style>
```

- [ ] **Step 2: Create Footer.astro**

```astro
---
// Footer.astro — Site footer with legal links
---

<div class="hud-divider"></div>
<footer class="flex flex-col sm:flex-row justify-between items-center px-6 py-5 gap-3">
  <div style="font-family: var(--font-retro); font-size: 12px; color: var(--text-ghost); letter-spacing: 2px;">
    © 2026 HEY HOMIE
  </div>
  <div class="flex gap-5">
    <a href="/terms" class="no-underline" style="font-family: var(--font-retro); font-size: 12px; color: var(--text-faint); letter-spacing: 1px;">TERMS</a>
    <a href="/privacy" class="no-underline" style="font-family: var(--font-retro); font-size: 12px; color: var(--text-faint); letter-spacing: 1px;">PRIVACY</a>
    <a href="https://github.com/MSG-88/Homie" target="_blank" rel="noopener" class="no-underline" style="font-family: var(--font-retro); font-size: 12px; color: var(--text-faint); letter-spacing: 1px;">GITHUB</a>
  </div>
</footer>
```

- [ ] **Step 3: Create BaseLayout.astro**

```astro
---
// BaseLayout.astro — Shared layout with HUD frame, nav, footer, meta tags
import Nav from '../components/Nav.astro';
import Footer from '../components/Footer.astro';
import '../styles/global.css';

interface Props {
  title: string;
  description: string;
  activePage?: string;
  ogImage?: string;
}

const { title, description, activePage = 'home', ogImage = '/og-default.png' } = Astro.props;
const canonicalURL = new URL(Astro.url.pathname, Astro.site);
---

<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <meta name="description" content={description} />
    <link rel="canonical" href={canonicalURL} />

    <!-- Open Graph -->
    <meta property="og:title" content={title} />
    <meta property="og:description" content={description} />
    <meta property="og:url" content={canonicalURL} />
    <meta property="og:image" content={ogImage} />
    <meta property="og:type" content="website" />

    <!-- Twitter -->
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content={title} />
    <meta name="twitter:description" content={description} />
    <meta name="twitter:image" content={ogImage} />

    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
  </head>
  <body>
    <Nav activePage={activePage} />
    <main class="max-w-4xl mx-auto">
      <slot />
    </main>
    <Footer />
  </body>
</html>
```

- [ ] **Step 4: Create a minimal index.astro to verify layout**

Replace `website/src/pages/index.astro` with:

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
---

<BaseLayout title="Hey Homie — Your Local AI Companion" description="Fully local, privacy-first AI assistant" activePage="home">
  <div class="py-20 text-center">
    <p class="status-line">■ SYSTEMS ONLINE</p>
    <h1 class="page-title">HEY HOMIE</h1>
    <p class="page-subtitle">LAYOUT TEST</p>
  </div>
</BaseLayout>
```

- [ ] **Step 5: Verify nav, footer, fonts, and theme render correctly**

```bash
cd C:/Users/muthu/PycharmProjects/Homie/website && npm run dev
```

Check: dark background, gold nav with arc reactor glow, footer links, pixel font headings, mobile hamburger on narrow viewport.

- [ ] **Step 6: Commit**

```bash
git add website/src/layouts/ website/src/components/ website/src/pages/index.astro
git commit -m "feat(website): add BaseLayout, Nav, and Footer components with HUD theme"
```

---

## Chunk 2: Landing Page Components & Page

### Task 4: HeroSection Component

**Files:**
- Create: `website/src/components/HeroSection.astro`

- [ ] **Step 1: Create HeroSection.astro**

```astro
---
// HeroSection.astro — Landing hero with arc reactor motif and CTA
---

<section class="py-16 px-6 text-center">
  <p class="status-line typing-effect">■ SYSTEMS ONLINE • ALL MODULES OPERATIONAL</p>

  <!-- Arc reactor -->
  <div class="reactor-glow mx-auto mb-6" style="width: 80px; height: 80px; border-radius: 50%; border: 2px solid rgba(241,196,15,0.2); display: flex; align-items: center; justify-content: center;">
    <div style="width: 40px; height: 40px; border-radius: 50%; border: 2px solid rgba(241,196,15,0.4); display: flex; align-items: center; justify-content: center;">
      <div style="width: 16px; height: 16px; border-radius: 50%; background: var(--accent-gold); box-shadow: 0 0 20px var(--accent-gold);"></div>
    </div>
  </div>

  <h1 style="font-family: var(--font-pixel); font-size: 28px; font-weight: 900; letter-spacing: 8px; color: var(--text-primary);">HEY HOMIE</h1>
  <p style="font-family: var(--font-retro); font-size: 14px; color: var(--text-muted); letter-spacing: 2px; margin-top: 8px;">YOUR LOCAL AI COMPANION</p>

  <p style="font-family: var(--font-retro); font-size: 15px; color: var(--text-body); max-width: 460px; margin: 30px auto; line-height: 1.8; letter-spacing: 0.5px;">
    A fully local, privacy-first AI assistant that runs entirely on your machine. No cloud. No tracking. Just you and your homie.
  </p>

  <a href="/download" class="cta-button">▶ GET STARTED</a>
</section>
```

- [ ] **Step 2: Commit**

```bash
git add website/src/components/HeroSection.astro
git commit -m "feat(website): add HeroSection component with arc reactor and CTA"
```

---

### Task 5: StatBar and FeatureCard Components

**Files:**
- Create: `website/src/components/StatBar.astro`
- Create: `website/src/components/FeatureCard.astro`

- [ ] **Step 1: Create StatBar.astro**

```astro
---
// StatBar.astro — Reusable RPG stat bar with fill animation
interface Props {
  percent: number;
  color: string;
}
const { percent, color } = Astro.props;
---

<div style="margin-top: 8px; height: 3px; background: #1a1a2e; border-radius: 2px;">
  <div
    class="stat-bar-fill"
    style={`height: 100%; width: ${percent}%; background: ${color}; border-radius: 2px;`}
  ></div>
</div>
```

- [ ] **Step 2: Create FeatureCard.astro**

```astro
---
// FeatureCard.astro — Feature displayed as an RPG skill card
import StatBar from './StatBar.astro';

interface Props {
  name: string;
  color: string;
  level: string;
  description: string;
  percent: number;
}
const { name, color, level, description, percent } = Astro.props;
---

<div class="skill-card" style={`border-color: ${color}20; background: ${color}08;`}>
  <div class="flex justify-between items-center mb-2">
    <span style={`font-family: var(--font-retro); font-size: 13px; color: ${color}; letter-spacing: 2px;`}>{name}</span>
    <span style="font-family: var(--font-pixel); font-size: 8px; color: var(--accent-green);">{level}</span>
  </div>
  <p style="font-family: var(--font-retro); font-size: 13px; color: var(--text-muted); line-height: 1.6;">{description}</p>
  <StatBar percent={percent} color={color} />
</div>
```

- [ ] **Step 3: Commit**

```bash
git add website/src/components/StatBar.astro website/src/components/FeatureCard.astro
git commit -m "feat(website): add StatBar and FeatureCard RPG components"
```

---

### Task 6: StatsRow and QuestLog Components

**Files:**
- Create: `website/src/components/StatsRow.astro`
- Create: `website/src/components/QuestLog.astro`

- [ ] **Step 1: Create StatsRow.astro**

```astro
---
// StatsRow.astro — Character stats row (big numbers)
const stats = [
  { value: '100%', label: 'LOCAL', color: 'var(--accent-gold)' },
  { value: '0', label: 'CLOUD CALLS', color: 'var(--accent-green)' },
  { value: '20+', label: 'MODULES', color: 'var(--accent-blue)' },
  { value: '∞', label: 'PRIVACY', color: 'var(--accent-red)' },
];
---

<section class="py-10 px-6 text-center">
  <p class="section-label" style="color: var(--text-faint);">CHARACTER STATS</p>
  <div class="flex justify-center gap-8 md:gap-12 flex-wrap">
    {stats.map(stat => (
      <div>
        <div style={`font-family: var(--font-pixel); font-size: 24px; color: ${stat.color}; font-weight: bold;`}>{stat.value}</div>
        <div style="font-family: var(--font-retro); font-size: 12px; color: var(--text-dim); letter-spacing: 2px; margin-top: 4px;">{stat.label}</div>
      </div>
    ))}
  </div>
</section>
```

- [ ] **Step 2: Create QuestLog.astro**

```astro
---
// QuestLog.astro — Getting started steps as quest items
const steps = [
  {
    num: '01',
    title: 'INSTALL HOMIE',
    desc: "pip install homie-ai. That's it. No accounts, no API keys.",
  },
  {
    num: '02',
    title: 'CHOOSE YOUR MODEL',
    desc: 'Pick a local LLM that fits your GPU. Homie downloads it for you.',
  },
  {
    num: '03',
    title: 'SAY "HEY HOMIE"',
    desc: 'Voice, text, or hotkey — your homie is always ready.',
  },
];
---

<section class="py-10 px-6">
  <p class="section-label" style="color: var(--text-faint);">QUEST LOG</p>
  <div class="max-w-md mx-auto flex flex-col gap-5">
    {steps.map(step => (
      <div class="flex gap-4 items-start">
        <span style="font-family: var(--font-retro); font-size: 14px; color: var(--accent-gold); min-width: 24px;">{step.num}</span>
        <div>
          <div style="font-family: var(--font-retro); font-size: 14px; color: var(--text-primary); letter-spacing: 1px;">{step.title}</div>
          <div style="font-family: var(--font-retro); font-size: 13px; color: var(--text-dim); margin-top: 4px;">{step.desc}</div>
        </div>
      </div>
    ))}
  </div>
</section>
```

- [ ] **Step 3: Commit**

```bash
git add website/src/components/StatsRow.astro website/src/components/QuestLog.astro
git commit -m "feat(website): add StatsRow and QuestLog components"
```

---

### Task 7: Assemble Landing Page

**Files:**
- Modify: `website/src/pages/index.astro`

- [ ] **Step 1: Write the full landing page**

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import HeroSection from '../components/HeroSection.astro';
import FeatureCard from '../components/FeatureCard.astro';
import StatsRow from '../components/StatsRow.astro';
import QuestLog from '../components/QuestLog.astro';

const features = [
  { name: 'VOICE', color: 'var(--accent-gold)', level: 'LVL MAX', description: 'Talk to your AI. It talks back. Wake word, push-to-talk, or full conversation.', percent: 95 },
  { name: 'MEMORY', color: 'var(--accent-green)', level: 'LVL MAX', description: 'Learns your habits, remembers your context. Working, episodic & semantic memory.', percent: 90 },
  { name: 'PRIVACY', color: 'var(--accent-blue)', level: 'LVL MAX', description: 'AES-256 vault. Zero telemetry. Your data never leaves your machine. Ever.', percent: 100 },
  { name: 'PLUGINS', color: 'var(--accent-purple)', level: '12 SLOTS', description: 'Browser, clipboard, IDE, git, terminal, health, music, notes — and more.', percent: 85 },
];
---

<BaseLayout
  title="Hey Homie — Your Local AI Companion"
  description="Fully local, privacy-first AI assistant that runs entirely on your machine. No cloud. No tracking. Just you and your homie."
  activePage="home"
>
  <HeroSection />

  <div class="hud-divider"></div>

  <!-- Equipped Abilities -->
  <section class="py-10 px-6">
    <p class="section-label" style="color: var(--text-faint);">EQUIPPED ABILITIES</p>
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-xl mx-auto">
      {features.map(f => (
        <FeatureCard
          name={f.name}
          color={f.color}
          level={f.level}
          description={f.description}
          percent={f.percent}
        />
      ))}
    </div>
  </section>

  <div class="hud-divider"></div>

  <StatsRow />

  <div class="hud-divider"></div>

  <QuestLog />
</BaseLayout>
```

- [ ] **Step 2: Verify the full landing page renders**

```bash
cd C:/Users/muthu/PycharmProjects/Homie/website && npm run dev
```

Check: Hero with arc reactor, 4 feature cards in grid, stats row, quest log, all styled correctly. Test mobile view (narrow browser).

- [ ] **Step 3: Commit**

```bash
git add website/src/pages/index.astro
git commit -m "feat(website): assemble full landing page with all sections"
```

---

## Chunk 3: About, Download, Terms, Privacy Pages

### Task 8: About Page

**Files:**
- Create: `website/src/pages/about.astro`

- [ ] **Step 1: Create about.astro**

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';

const missions = [
  { color: 'var(--accent-gold)', title: 'PRIVACY IS NOT A FEATURE', desc: "It's the foundation. Your data stays on your machine, encrypted in a vault even we can't open." },
  { color: 'var(--accent-green)', title: 'LOCAL MEANS LOCAL', desc: 'Your LLM runs on your GPU. Your voice stays on your mic. Zero cloud calls required.' },
  { color: 'var(--accent-blue)', title: 'OPEN SOURCE, OPEN HEART', desc: 'MPL-2.0 licensed. Read the code, fork the code, make it yours. No vendor lock-in.' },
  { color: 'var(--accent-purple)', title: 'YOUR AI SHOULD KNOW YOU', desc: 'Not to exploit you — to help you. Homie learns your patterns, anticipates your needs, and never shares a byte.' },
];

const inventory = [
  { name: 'Python', label: 'CORE', color: 'var(--accent-gold)' },
  { name: 'llama.cpp', label: 'ENGINE', color: 'var(--accent-green)' },
  { name: 'SQLite', label: 'STORAGE', color: 'var(--accent-blue)' },
  { name: 'ChromaDB', label: 'VECTORS', color: 'var(--accent-purple)' },
  { name: 'Whisper', label: 'VOICE', color: 'var(--accent-red)' },
  { name: 'AES-256', label: 'VAULT', color: 'var(--accent-orange)' },
];
---

<BaseLayout
  title="About — Hey Homie"
  description="The origin story, mission, and team behind Homie AI."
  activePage="about"
>
  <!-- Header -->
  <section class="pt-12 pb-8 px-6 text-center">
    <p class="status-line">■ DOSSIER ACCESS GRANTED</p>
    <h1 class="page-title">ABOUT HOMIE</h1>
    <p class="page-subtitle">ORIGIN STORY • MISSION • CREATOR</p>
  </section>

  <div class="hud-divider"></div>

  <!-- Origin Story -->
  <section class="py-10 px-6 max-w-lg mx-auto">
    <p class="section-label" style="color: var(--accent-gold);">ORIGIN STORY</p>
    <p style="font-family: var(--font-retro); font-size: 15px; color: var(--text-body); line-height: 2; letter-spacing: 0.5px;">
      Every hero needs a companion. Not one that lives in some corporate data center, harvesting your habits and selling your secrets. One that lives with you. On your machine. Under your rules.
    </p>
    <p style="font-family: var(--font-retro); font-size: 15px; color: var(--text-muted); line-height: 2; letter-spacing: 0.5px; margin-top: 16px;">
      Homie was born from a simple idea: what if your AI assistant was actually <span style="color: var(--accent-gold);">yours</span>? No subscriptions. No cloud dependencies. No "we updated our privacy policy" emails. Just a smart, local companion that gets better the more you use it.
    </p>
  </section>

  <div class="hud-divider"></div>

  <!-- Mission Parameters -->
  <section class="py-10 px-6 max-w-lg mx-auto">
    <p class="section-label" style="color: var(--accent-blue);">MISSION PARAMETERS</p>
    <div class="flex flex-col gap-4">
      {missions.map(m => (
        <div class="flex gap-3 items-start">
          <div style={`min-width: 8px; height: 8px; margin-top: 6px; background: ${m.color};`}></div>
          <div>
            <div style={`font-family: var(--font-retro); font-size: 14px; color: var(--text-primary); letter-spacing: 2px; margin-bottom: 4px;`}>{m.title}</div>
            <div style="font-family: var(--font-retro); font-size: 13px; color: var(--text-dim); line-height: 1.6;">{m.desc}</div>
          </div>
        </div>
      ))}
    </div>
  </section>

  <div class="hud-divider"></div>

  <!-- Creator Profile -->
  <section class="py-10 px-6 max-w-lg mx-auto">
    <p class="section-label" style="color: var(--accent-purple);">CREATOR PROFILE</p>
    <div class="skill-card" style="border-color: var(--border-gold); background: var(--bg-card);">
      <div class="flex justify-between items-center mb-3">
        <div>
          <div style="font-family: var(--font-pixel); font-size: 14px; color: var(--accent-gold); letter-spacing: 2px;">MSG</div>
          <div style="font-family: var(--font-retro); font-size: 12px; color: var(--text-dim); letter-spacing: 1px; margin-top: 4px;">MUTHU G. SUBRAMANIAN</div>
        </div>
        <div style="font-family: var(--font-pixel); font-size: 8px; color: var(--accent-green); letter-spacing: 2px; border: 1px solid rgba(46,204,113,0.3); padding: 3px 8px;">ONLINE</div>
      </div>
      <p style="font-family: var(--font-retro); font-size: 14px; color: var(--text-muted); line-height: 1.8;">
        Builder of things that respect people. Believes AI should empower individuals, not corporations. Made Homie because he wanted an AI assistant he could actually trust.
      </p>
      <div class="flex gap-4 mt-4">
        <a href="https://github.com/MSG-88" target="_blank" rel="noopener" class="no-underline" style="font-family: var(--font-retro); font-size: 13px; color: var(--text-faint); letter-spacing: 1px;">GITHUB ↗</a>
        <a href="mailto:muthu.g.subramanian@outlook.com" class="no-underline" style="font-family: var(--font-retro); font-size: 13px; color: var(--text-faint); letter-spacing: 1px;">EMAIL ↗</a>
      </div>
    </div>
  </section>

  <div class="hud-divider"></div>

  <!-- Inventory -->
  <section class="py-10 px-6 max-w-lg mx-auto">
    <p class="section-label" style="color: var(--accent-red);">INVENTORY</p>
    <div class="grid grid-cols-3 gap-2">
      {inventory.map(item => (
        <div style="border: 1px solid var(--border-subtle); padding: 10px; text-align: center;">
          <div style={`font-family: var(--font-retro); font-size: 14px; color: ${item.color}; letter-spacing: 1px;`}>{item.name}</div>
          <div style="font-family: var(--font-pixel); font-size: 7px; color: var(--text-faint); margin-top: 3px;">{item.label}</div>
        </div>
      ))}
    </div>
  </section>
</BaseLayout>
```

- [ ] **Step 2: Verify about page renders at /about**

- [ ] **Step 3: Commit**

```bash
git add website/src/pages/about.astro
git commit -m "feat(website): add About page with origin story, mission, creator profile, inventory"
```

---

### Task 9: Download Page

**Files:**
- Create: `website/src/pages/download.astro`

- [ ] **Step 1: Create download.astro**

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';

const specs = [
  { label: 'CPU', value: 'Any modern x86/ARM', note: '' },
  { label: 'RAM', value: '8 GB minimum', note: '' },
  { label: 'GPU', value: 'Optional (CUDA/Metal)', note: '+10x speed boost', noteColor: 'var(--accent-green)' },
  { label: 'DISK', value: '2 GB + model size', note: '' },
  { label: 'PYTHON', value: '3.11 or newer', note: '' },
  { label: 'INTERNET', value: 'Only for install', note: 'then fully offline', noteColor: 'var(--accent-gold)' },
];

const modules = [
  { name: 'Voice', color: 'var(--accent-gold)', cmd: 'pip install homie-ai[voice]', size: '~2 GB' },
  { name: 'Email', color: 'var(--accent-green)', cmd: 'pip install homie-ai[email]', size: '~5 MB' },
  { name: 'Neural', color: 'var(--accent-blue)', cmd: 'pip install homie-ai[neural]', size: '~50 MB' },
];
---

<BaseLayout
  title="Download — Hey Homie"
  description="Install Homie AI on Windows, Linux, or macOS. Fully local, privacy-first AI assistant."
  activePage="download"
>
  <!-- Header -->
  <section class="pt-12 pb-8 px-6 text-center">
    <p class="status-line">■ DEPLOYMENT READY</p>
    <h1 class="page-title">SUIT UP</h1>
    <p class="page-subtitle">SELECT YOUR PLATFORM • INSTALL • LAUNCH</p>
  </section>

  <div class="hud-divider"></div>

  <!-- Quick Deploy -->
  <section class="py-10 px-6 text-center max-w-lg mx-auto">
    <p class="section-label" style="color: var(--accent-gold);">QUICK DEPLOY</p>
    <div style="background: #111; border: 1px solid var(--border-gold); padding: 16px 20px; text-align: left; position: relative;">
      <div style="font-family: var(--font-pixel); font-size: 8px; color: var(--text-faint); letter-spacing: 2px; margin-bottom: 8px;">TERMINAL</div>
      <div style="font-family: var(--font-retro); font-size: 16px; color: var(--accent-green);">
        <span style="color: var(--text-dim);">$</span> pip install homie-ai
      </div>
      <button
        id="copy-btn"
        style="position: absolute; top: 16px; right: 16px; font-family: var(--font-pixel); font-size: 8px; color: var(--text-faint); letter-spacing: 1px; border: 1px solid var(--text-ghost); padding: 2px 8px; background: none; cursor: pointer;"
      >COPY</button>
    </div>
    <p style="font-family: var(--font-retro); font-size: 13px; color: var(--text-faint); margin-top: 10px; letter-spacing: 1px;">
      Works on all platforms. Requires Python 3.11+
    </p>
  </section>

  <div class="hud-divider"></div>

  <!-- Platform Select -->
  <section class="py-10 px-6 max-w-lg mx-auto">
    <p class="section-label" style="color: var(--accent-blue);">PLATFORM SELECT</p>
    <div class="grid grid-cols-3 gap-3">
      <div id="platform-windows" class="platform-card" style="border: 1px solid var(--border-subtle); padding: 20px 12px; text-align: center;">
        <div style="font-size: 22px; margin-bottom: 8px;">⊞</div>
        <div style="font-family: var(--font-retro); font-size: 14px; color: var(--text-muted); letter-spacing: 2px;">WINDOWS</div>
        <div id="detect-windows" class="platform-detect" style="font-family: var(--font-pixel); font-size: 8px; color: var(--accent-green); margin-top: 6px; letter-spacing: 1px; visibility: hidden;">DETECTED ✓</div>
        <div style="font-family: var(--font-retro); font-size: 11px; color: var(--text-faint); margin-top: 8px;">Win 10/11</div>
      </div>
      <div id="platform-linux" class="platform-card" style="border: 1px solid var(--border-subtle); padding: 20px 12px; text-align: center;">
        <div style="font-size: 22px; margin-bottom: 8px;">🐧</div>
        <div style="font-family: var(--font-retro); font-size: 14px; color: var(--text-muted); letter-spacing: 2px;">LINUX</div>
        <div id="detect-linux" class="platform-detect" style="font-family: var(--font-pixel); font-size: 8px; color: var(--accent-green); margin-top: 6px; letter-spacing: 1px; visibility: hidden;">DETECTED ✓</div>
        <div style="font-family: var(--font-retro); font-size: 11px; color: var(--text-faint); margin-top: 8px;">Ubuntu/Fedora/Arch</div>
      </div>
      <div id="platform-macos" class="platform-card" style="border: 1px solid var(--border-subtle); padding: 20px 12px; text-align: center;">
        <div style="font-size: 22px; margin-bottom: 8px;">🍎</div>
        <div style="font-family: var(--font-retro); font-size: 14px; color: var(--text-muted); letter-spacing: 2px;">MACOS</div>
        <div id="detect-macos" class="platform-detect" style="font-family: var(--font-pixel); font-size: 8px; color: var(--accent-green); margin-top: 6px; letter-spacing: 1px; visibility: hidden;">DETECTED ✓</div>
        <div style="font-family: var(--font-retro); font-size: 11px; color: var(--text-faint); margin-top: 8px;">13+ (Ventura)</div>
      </div>
    </div>
  </section>

  <div class="hud-divider"></div>

  <!-- Minimum Specs -->
  <section class="py-10 px-6 max-w-lg mx-auto">
    <p class="section-label" style="color: var(--accent-purple);">MINIMUM SPECS</p>
    <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
      {specs.map(s => (
        <div style="border: 1px solid var(--border-subtle); padding: 14px;">
          <div style="font-family: var(--font-pixel); font-size: 8px; color: var(--text-faint); letter-spacing: 2px; margin-bottom: 6px;">{s.label}</div>
          <div style="font-family: var(--font-retro); font-size: 14px; color: var(--text-body);">{s.value}</div>
          {s.note && <div style={`font-family: var(--font-retro); font-size: 11px; color: ${s.noteColor || 'var(--text-faint)'}; margin-top: 3px;`}>{s.note}</div>}
        </div>
      ))}
    </div>
  </section>

  <div class="hud-divider"></div>

  <!-- Optional Modules -->
  <section class="py-10 px-6 max-w-lg mx-auto">
    <p class="section-label" style="color: var(--accent-red);">OPTIONAL MODULES</p>
    <div class="flex flex-col gap-2">
      {modules.map(m => (
        <div class="flex justify-between items-center" style="border: 1px solid var(--border-subtle); padding: 10px 14px;">
          <div>
            <span style={`font-family: var(--font-retro); font-size: 14px; color: ${m.color}; letter-spacing: 1px;`}>{m.name}</span>
            <span style="font-family: var(--font-retro); font-size: 12px; color: var(--text-faint); margin-left: 8px;">{m.cmd}</span>
          </div>
          <span style="font-family: var(--font-retro); font-size: 12px; color: var(--text-faint);">{m.size}</span>
        </div>
      ))}
      <!-- Full suit -->
      <div class="flex justify-between items-center" style="border: 1px solid var(--border-gold); padding: 10px 14px; background: var(--bg-card);">
        <div>
          <span style="font-family: var(--font-retro); font-size: 14px; color: var(--accent-gold); letter-spacing: 1px;">Everything</span>
          <span style="font-family: var(--font-retro); font-size: 12px; color: var(--text-faint); margin-left: 8px;">pip install homie-ai[all]</span>
        </div>
        <span style="font-family: var(--font-pixel); font-size: 8px; color: var(--accent-gold);">FULL SUIT</span>
      </div>
    </div>
  </section>
</BaseLayout>

<script>
  // Copy button
  const copyBtn = document.getElementById('copy-btn');
  copyBtn?.addEventListener('click', () => {
    navigator.clipboard.writeText('pip install homie-ai');
    copyBtn.textContent = 'COPIED ✓';
    setTimeout(() => { copyBtn.textContent = 'COPY'; }, 2000);
  });

  // Platform detection
  function detectPlatform(): string {
    const uad = (navigator as any).userAgentData;
    if (uad?.platform) return uad.platform.toLowerCase();
    const p = navigator.platform.toLowerCase();
    if (p.includes('win')) return 'windows';
    if (p.includes('mac')) return 'macos';
    if (p.includes('linux')) return 'linux';
    return 'unknown';
  }

  const platform = detectPlatform();
  const map: Record<string, string> = { windows: 'platform-windows', macos: 'platform-macos', linux: 'platform-linux' };
  const cardId = map[platform];
  if (cardId) {
    const card = document.getElementById(cardId);
    if (card) {
      card.style.borderColor = 'rgba(241, 196, 15, 0.4)';
      card.style.borderWidth = '2px';
      card.querySelector('.platform-detect')?.setAttribute('style',
        card.querySelector('.platform-detect')?.getAttribute('style')?.replace('visibility: hidden', 'visibility: visible') || ''
      );
      const nameEl = card.querySelector('div:nth-child(2)') as HTMLElement;
      if (nameEl) nameEl.style.color = 'var(--accent-gold)';
    }
  }
</script>
```

- [ ] **Step 2: Verify download page at /download**

- [ ] **Step 3: Commit**

```bash
git add website/src/pages/download.astro
git commit -m "feat(website): add Download page with platform detection, specs, modules"
```

---

### Task 10: Terms of Service Page

**Files:**
- Create: `website/src/pages/terms.astro`

- [ ] **Step 1: Create terms.astro**

Full legal content page with system sans-serif body text and retro section headings. Content as specified in the design spec — Acceptance, License (MPL-2.0), What Homie Is, Your Responsibilities, No Warranty, Third-Party Models, Optional Cloud Connections, Limitation of Liability, Changes, Contact.

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
---

<BaseLayout
  title="Terms of Service — Hey Homie"
  description="Terms of Service for Homie AI, the fully local privacy-first AI assistant."
  activePage=""
>
  <section class="pt-12 pb-8 px-6 text-center">
    <p class="status-line">■ LEGAL PROTOCOLS ACTIVE</p>
    <h1 class="page-title">TERMS OF SERVICE</h1>
    <p class="page-subtitle">EFFECTIVE: MARCH 2026</p>
  </section>

  <div class="hud-divider"></div>

  <article class="py-10 px-6 max-w-2xl mx-auto legal-content">
    <section class="legal-section">
      <h2>1. Acceptance of Terms</h2>
      <p>By downloading, installing, or using Homie AI ("Homie," "the Software"), you agree to be bound by these Terms of Service. If you do not agree, do not use the Software.</p>
    </section>

    <section class="legal-section">
      <h2>2. License</h2>
      <p>Homie is open-source software released under the <strong>Mozilla Public License 2.0 (MPL-2.0)</strong>. In plain terms:</p>
      <ul>
        <li>You may use, copy, modify, and distribute Homie freely.</li>
        <li>If you modify files covered by the MPL, those modifications must remain open source under the same license.</li>
        <li>You may combine Homie with proprietary software in a larger project — the MPL only applies to the Homie files themselves.</li>
        <li>The full license text is available at <a href="https://www.mozilla.org/en-US/MPL/2.0/" target="_blank" rel="noopener">mozilla.org/MPL/2.0</a>.</li>
      </ul>
    </section>

    <section class="legal-section">
      <h2>3. What Homie Is</h2>
      <p>Homie is local software that runs entirely on your machine. It is not a cloud service, not a SaaS product, and not a hosted platform. We do not operate servers that process your data, and we do not host, control, or monitor your Homie instance in any way.</p>
    </section>

    <section class="legal-section">
      <h2>4. Your Responsibilities</h2>
      <p>You are responsible for:</p>
      <ul>
        <li>Your hardware and its maintenance.</li>
        <li>Your data and any backups thereof.</li>
        <li>The language models you choose to download and run, including compliance with their respective licenses (e.g., Llama Community License, Mistral License, Qwen License).</li>
        <li>Any external services you connect to Homie (Gmail, social media platforms, etc.).</li>
      </ul>
    </section>

    <section class="legal-section">
      <h2>5. No Warranty</h2>
      <p>Homie is currently in <strong>alpha</strong> and is provided "as-is" without warranties of any kind, whether express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, or non-infringement. We do not guarantee that Homie will be error-free, uninterrupted, or suitable for any specific use case.</p>
    </section>

    <section class="legal-section">
      <h2>6. Third-Party Models and Software</h2>
      <p>Language models, voice models, and other AI components that you download through or use with Homie are third-party software with their own licenses and terms. We do not create, control, or take responsibility for the outputs of these models.</p>
    </section>

    <section class="legal-section">
      <h2>7. Optional Cloud Connections</h2>
      <p>If you choose to connect external services (such as Gmail, social media platforms, or other APIs), those providers' terms of service and privacy policies apply to your use of those services. Homie facilitates the connection on your local machine — we do not act as an intermediary, and no data flows through our servers.</p>
    </section>

    <section class="legal-section">
      <h2>8. Limitation of Liability</h2>
      <p>To the maximum extent permitted by applicable law, Hey Homie and its contributors shall not be liable for any indirect, incidental, special, consequential, or punitive damages, including but not limited to loss of data, loss of profits, or damages arising from the use or inability to use the Software.</p>
    </section>

    <section class="legal-section">
      <h2>9. Changes to These Terms</h2>
      <p>We may update these Terms from time to time. Changes will be posted on this page with an updated effective date. We will not send you emails about terms changes — because we don't have your email.</p>
    </section>

    <section class="legal-section">
      <h2>10. Contact</h2>
      <p>For questions about these terms, contact: <a href="mailto:muthu.g.subramanian@outlook.com">muthu.g.subramanian@outlook.com</a></p>
    </section>
  </article>
</BaseLayout>

<style>
  .legal-content {
    font-family: var(--font-legal);
    color: var(--text-body);
    line-height: 1.8;
  }
  .legal-section {
    margin-bottom: 32px;
  }
  .legal-section h2 {
    font-family: var(--font-retro);
    font-size: 16px;
    color: var(--accent-gold);
    letter-spacing: 2px;
    margin-bottom: 12px;
    text-transform: uppercase;
  }
  .legal-section p {
    font-size: 15px;
    margin-bottom: 12px;
  }
  .legal-section ul {
    padding-left: 20px;
    margin-bottom: 12px;
  }
  .legal-section li {
    font-size: 15px;
    margin-bottom: 6px;
  }
  .legal-section a {
    color: var(--accent-gold);
    text-decoration: underline;
  }
  .legal-section strong {
    color: var(--text-primary);
  }
</style>
```

- [ ] **Step 2: Verify at /terms**

- [ ] **Step 3: Commit**

```bash
git add website/src/pages/terms.astro
git commit -m "feat(website): add Terms of Service page"
```

---

### Task 11: Privacy Policy Page

**Files:**
- Create: `website/src/pages/privacy.astro`

- [ ] **Step 1: Create privacy.astro**

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
---

<BaseLayout
  title="Privacy Policy — Hey Homie"
  description="Privacy Policy for Homie AI. We don't collect your data. Period."
  activePage=""
>
  <section class="pt-12 pb-8 px-6 text-center">
    <p class="status-line">■ LEGAL PROTOCOLS ACTIVE</p>
    <h1 class="page-title">PRIVACY POLICY</h1>
    <p class="page-subtitle">EFFECTIVE: MARCH 2026</p>
  </section>

  <div class="hud-divider"></div>

  <article class="py-10 px-6 max-w-2xl mx-auto legal-content">

    <section class="legal-section highlight-section">
      <h2>The Short Version</h2>
      <p><strong>We don't collect your data. Period.</strong> Homie runs entirely on your machine. We have no servers, no analytics pipeline, and no way to see what you do with Homie. This privacy policy exists because the law says we need one — but the honest answer is: your data is yours, and we never see it.</p>
    </section>

    <section class="legal-section">
      <h2>1. What Data Homie Processes Locally</h2>
      <p>Homie processes data on your machine to function as your personal AI assistant. This may include:</p>
      <ul>
        <li><strong>Observations:</strong> Work patterns, routines, and habits (if enabled in privacy settings).</li>
        <li><strong>Memory:</strong> Working memory (current context), episodic memory (past interactions), and semantic memory (learned knowledge).</li>
        <li><strong>Voice:</strong> Speech-to-text and text-to-speech audio, processed locally via faster-whisper and Piper/Kokoro/MeloTTS.</li>
        <li><strong>Email:</strong> Gmail messages if you connect your account — fetched and processed locally.</li>
        <li><strong>Browser history:</strong> If enabled, browsing data from local browser profiles.</li>
        <li><strong>Financial reminders:</strong> If enabled, extracted from emails and documents locally.</li>
      </ul>
      <p>All of this data is processed and stored locally in <code>~/.homie/</code> on your machine. None of it is transmitted anywhere.</p>
    </section>

    <section class="legal-section">
      <h2>2. How Local Data Is Stored</h2>
      <p>Homie stores data in two forms:</p>
      <ul>
        <li><strong>Databases:</strong> SQLite databases and ChromaDB vector stores in <code>~/.homie/</code>.</li>
        <li><strong>Encrypted vault:</strong> Sensitive data (credentials, tokens, personal information) is encrypted with <strong>AES-256-GCM</strong> in a vault protected by your OS keyring (Windows Credential Manager, macOS Keychain, or Linux Secret Service). An optional password layer uses PBKDF2 with 600,000 iterations.</li>
      </ul>
    </section>

    <section class="legal-section">
      <h2>3. What We (Hey Homie) Collect</h2>
      <p><strong>From the Homie application:</strong> Nothing. Zero telemetry, zero crash reporting, zero phone-home. There is no code in Homie that transmits data to us or to anyone else.</p>
      <p><strong>From this website:</strong> Standard web server logs (IP address, user agent, pages visited) as collected by our hosting provider. We do not run analytics scripts, tracking pixels, or advertising technology on this website.</p>
    </section>

    <section class="legal-section">
      <h2>4. Optional Cloud Connections</h2>
      <p>If you choose to connect external services:</p>
      <ul>
        <li><strong>Gmail:</strong> Uses OAuth 2.0. Tokens are stored in your local encrypted vault. Emails are fetched and processed on your machine. No email data passes through our servers.</li>
        <li><strong>Social media:</strong> API tokens stored in your local vault. Data flows between your machine and the platform directly — never through us.</li>
      </ul>
      <p>These services have their own privacy policies. We encourage you to review them.</p>
    </section>

    <section class="legal-section">
      <h2>5. Voice Data</h2>
      <p>Voice data is processed entirely on your machine using local models (faster-whisper for speech-to-text, Piper/Kokoro/MeloTTS for text-to-speech). Audio is never transmitted over the network. No voice recordings are stored unless you explicitly enable voice logging in your configuration.</p>
    </section>

    <section class="legal-section">
      <h2>6. No Tracking, No Telemetry</h2>
      <p>Homie contains:</p>
      <ul>
        <li>No analytics SDK</li>
        <li>No crash reporting</li>
        <li>No update checks that transmit usage data</li>
        <li>No feature flags or A/B testing</li>
        <li>No advertising identifiers</li>
      </ul>
      <p>Homie does not know how many users it has. We like it that way.</p>
    </section>

    <section class="legal-section">
      <h2>7. Data Retention and Control</h2>
      <p>You have full control over your data:</p>
      <ul>
        <li><strong>Retention period:</strong> Configurable (default: 30 days). Older data is automatically purged.</li>
        <li><strong>Storage limit:</strong> Configurable (default: 512 MB).</li>
        <li><strong>Delete everything:</strong> Remove the <code>~/.homie/</code> directory. That's it. There's nothing else anywhere.</li>
        <li><strong>Export:</strong> <code>homie backup --to &lt;path&gt;</code> creates an encrypted archive of your data.</li>
      </ul>
    </section>

    <section class="legal-section">
      <h2>8. Your Rights</h2>
      <p>You have full control over all data Homie processes. You can view, export, modify, or delete any data at any time by accessing your <code>~/.homie/</code> directory or using the CLI tools. No request to us is needed — it is your filesystem, your machine, your data.</p>
    </section>

    <section class="legal-section">
      <h2>9. Children's Privacy</h2>
      <p>Homie is not designed for or directed at children under the age of 13. We do not knowingly collect or process data from children. Since Homie processes data locally and we collect nothing, this is a formality — but we state it for compliance with COPPA and similar regulations.</p>
    </section>

    <section class="legal-section">
      <h2>10. Changes to This Policy</h2>
      <p>We may update this Privacy Policy from time to time. Changes will be posted on this page with an updated effective date. Since we don't have your email (or any way to contact you), checking this page is the only way to see updates.</p>
    </section>

    <section class="legal-section">
      <h2>11. Contact</h2>
      <p>For questions about this privacy policy, contact: <a href="mailto:muthu.g.subramanian@outlook.com">muthu.g.subramanian@outlook.com</a></p>
    </section>

  </article>
</BaseLayout>

<style>
  .legal-content {
    font-family: var(--font-legal);
    color: var(--text-body);
    line-height: 1.8;
  }
  .legal-section {
    margin-bottom: 32px;
  }
  .legal-section h2 {
    font-family: var(--font-retro);
    font-size: 16px;
    color: var(--accent-gold);
    letter-spacing: 2px;
    margin-bottom: 12px;
    text-transform: uppercase;
  }
  .legal-section p {
    font-size: 15px;
    margin-bottom: 12px;
  }
  .legal-section ul {
    padding-left: 20px;
    margin-bottom: 12px;
  }
  .legal-section li {
    font-size: 15px;
    margin-bottom: 6px;
  }
  .legal-section a {
    color: var(--accent-gold);
    text-decoration: underline;
  }
  .legal-section strong {
    color: var(--text-primary);
  }
  .legal-section code {
    font-family: var(--font-retro);
    background: rgba(241, 196, 15, 0.1);
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 14px;
    color: var(--accent-gold);
  }
  .highlight-section {
    border-left: 3px solid var(--accent-gold);
    padding-left: 16px;
  }
</style>
```

- [ ] **Step 2: Verify at /privacy**

- [ ] **Step 3: Commit**

```bash
git add website/src/pages/privacy.astro
git commit -m "feat(website): add Privacy Policy page"
```

---

## Chunk 4: Polish & Deployment

### Task 12: Favicon, OG Image, and SEO Extras

**Files:**
- Create: `website/public/favicon.svg`
- Create: `website/public/og-default.png`
- Modify: `website/astro.config.mjs` (add site URL)

- [ ] **Step 1: Create favicon.svg (arc reactor motif)**

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="4" fill="#0d0d1a"/>
  <circle cx="16" cy="16" r="12" fill="none" stroke="#f1c40f" stroke-width="1" opacity="0.3"/>
  <circle cx="16" cy="16" r="7" fill="none" stroke="#f1c40f" stroke-width="1.5" opacity="0.5"/>
  <circle cx="16" cy="16" r="3" fill="#f1c40f"/>
</svg>
```

- [ ] **Step 2: Create a simple OG image**

Generate a 1200x630 PNG with dark background, "HEY HOMIE" text, arc reactor motif, and tagline. Can be created via HTML-to-image or a simple canvas script. For now, create a placeholder SVG-based approach and convert later.

- [ ] **Step 3: Update astro.config.mjs with site URL**

```javascript
import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://heyhomie.app',
  integrations: [tailwind()],
});
```

- [ ] **Step 4: Commit**

```bash
git add website/public/ website/astro.config.mjs
git commit -m "feat(website): add favicon, OG image placeholder, set site URL"
```

---

### Task 13: Build and Verify Static Output

**Files:**
- No new files

- [ ] **Step 1: Run production build**

```bash
cd C:/Users/muthu/PycharmProjects/Homie/website
npm run build
```

Expected: Clean build, static output in `website/dist/`.

- [ ] **Step 2: Preview the production build**

```bash
cd C:/Users/muthu/PycharmProjects/Homie/website
npm run preview
```

Verify all 5 pages load: /, /about, /download, /terms, /privacy. Check mobile responsive. Check all links work.

- [ ] **Step 3: Check page sizes**

Verify each page is under 100KB (excluding fonts).

- [ ] **Step 4: Commit any build fixes**

```bash
git add -A website/
git commit -m "fix(website): build fixes for production output"
```

---

### Task 14: Deploy to Hosting

**Files:**
- Create: `website/netlify.toml` or `website/vercel.json` (depending on choice)

- [ ] **Step 1: Choose deployment platform**

Recommended: **Netlify** (free tier, custom domain support, auto-deploy from git).

Create `website/netlify.toml`:

```toml
[build]
  command = "npm run build"
  publish = "dist"

[build.environment]
  NODE_VERSION = "20"
```

Or for **Vercel**, create `website/vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "astro"
}
```

- [ ] **Step 2: Deploy**

Option A — Netlify CLI:
```bash
cd C:/Users/muthu/PycharmProjects/Homie/website
npx netlify-cli deploy --prod --dir=dist
```

Option B — Vercel CLI:
```bash
cd C:/Users/muthu/PycharmProjects/Homie/website
npx vercel --prod
```

Option C — Push to GitHub and connect repo to Netlify/Vercel dashboard. Set base directory to `website/`.

- [ ] **Step 3: Configure custom domain**

Point heyhomie.app DNS to the deployment platform:
- Add custom domain in Netlify/Vercel dashboard
- Update DNS records at GoDaddy (CNAME or A record)
- Enable HTTPS/SSL

- [ ] **Step 4: Verify live site**

Visit https://heyhomie.app and check all 5 pages.

- [ ] **Step 5: Commit deployment config**

```bash
git add website/netlify.toml  # or vercel.json
git commit -m "feat(website): add deployment configuration"
```

---

### Task 15: Add .gitignore and Final Cleanup

**Files:**
- Create: `website/.gitignore`
- Modify: `.gitignore` (root)

- [ ] **Step 1: Create website/.gitignore**

```
node_modules/
dist/
.astro/
```

- [ ] **Step 2: Add .superpowers/ to root .gitignore**

Append to root `.gitignore`:
```
.superpowers/
```

- [ ] **Step 3: Final commit**

```bash
git add website/.gitignore .gitignore
git commit -m "chore(website): add gitignore for website build artifacts"
```
