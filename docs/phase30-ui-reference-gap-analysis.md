# UI Reference Gap Analysis - Phase 30

Date: 2026-05-08

## Scope

This compares the saved reference UI images against the current Phase 29 browser screenshots. Per the current roadmap, these UI gaps are now treated as Phase 30 and must be repaired before the later post-review engineering phases continue.

Reference images:

- `docs/ui-concepts/phase26/01-overview-workspace.png`
- `docs/ui-concepts/phase26/02-library.png`
- `docs/ui-concepts/phase26/03-notes.png`
- `docs/ui-concepts/phase26/04-settings.png`
- `docs/ui-concepts/phase26/05-source-preview.png`

Current screenshots:

- `docs/phase29-chat-desktop.png`
- `docs/phase29-library-desktop.png`
- `docs/phase29-notes-desktop.png`
- `docs/phase29-settings-desktop.png`

Same-size comparison screenshots were generated under:

- `output/playwright/ui-reference-comparison-current/`
- `output/playwright/ui-reference-comparison-sheets/`

## Overall Result

The current UI is visually close in broad style: pale local-first shell, left navigation, top search, central workspace, and right context panel. It is not yet a 100% reference restoration.

The main gap is not color. The main gap is information architecture and working-state completeness:

- Chat is captured in an empty welcome state, while the reference is an answered conversation with citations, actions, and progress.
- Library still feels like a file table plus upload zone, while the reference feels like a dense personal library manager.
- Notes lacks the full editor, import options, template cards, saved-answer panel, and processing timeline.
- Settings has removed developer wording correctly, but its layout no longer matches the reference density and status/storage structure.
- Source Preview is not implemented as a dedicated page matching the reference.

## Pair 1: Overview Workspace vs Chat

Reference: `01-overview-workspace.png`  
Current: `phase29-chat-desktop.png`

### Major Differences

| Area | Reference | Current | Impact |
|---|---|---|---|
| Screen state | Shows a real answered conversation, source-backed answer, table, answer actions, and input. | Shows first-run/welcome state with three starter cards. | Cannot verify the main Q&A experience against the reference. |
| Main density | Central answer area is filled and readable. | Large blank space dominates the middle of the page. | Looks unfinished when compared to the target screenshot. |
| User message | Top-right user question bubble is visible. | No user question shown. | Missing the core conversation shape. |
| Answer body | Rich answer card, structured summary, table, and cited evidence. | Only welcome text and quick-start buttons. | Reference answer layout is not restored. |
| Answer actions | Copy, save as note, add to library, generate mind map. | These are not visible in the captured state. | Key post-answer workflow is not visible. |
| Citation panel | Shows real citations with confidence percentages. | Placeholder says citations will show after asking. | Right panel lacks the reference's evidence density. |
| Source preview panel | Shows a real selected source excerpt. | Not visible in current Chat screenshot. | Source-checking loop is weaker. |
| Model panel | Shows model name, context usage, speed, and temperature. | Shows a compact count/model/status summary. | Reference runtime feedback is richer. |
| Background tasks | Shows active task progress bars. | Shows empty task state. | Reference work-progress surface is not restored. |
| Language consistency | Page labels are Chinese. | Main title is `DocFlow`; body is mixed Chinese and English. | Still feels less localized and less like the reference. |

### Required Fixes

1. Capture and design the default README/acceptance Chat screenshot in a real answered state, not only the empty welcome state.
2. Make the answered state match the reference: user bubble, answer card, citation count, answer action row, and source-backed right panel.
3. Expand the model status panel to show useful runtime details when available.
4. Show queue/task progress in the right panel when tasks exist, with the same compact progress treatment as the reference.

## Pair 2: Library

Reference: `02-library.png`  
Current: `phase29-library-desktop.png`

### Major Differences

| Area | Reference | Current | Impact |
|---|---|---|---|
| Top toolbar | Toolbar is inside the Library panel, compact and aligned under the title. | Toolbar sits in the page header, separated from the table surface. | Less visually tied to the library workspace. |
| Upload zone | No giant upload drop zone in the default table view. | Large dashed upload zone takes the first screen row. | The file manager feels less dense and less like the reference. |
| Left filters | Groups, collections, and tags are persistent with counts. | Groups plus dropdown filters; collections/tags are less visible. | Weaker library browsing model. |
| Counts | Reference counts are coherent across all groups. | Current type counts show `0` for PDF/Markdown/images/code while rows exist. | Looks wrong and undermines trust. |
| Table row design | Rows show file icons, favorite star, collection, tags, type, status, pages, chunks, and updated time. | Rows show radio selector, filename/path, collection, tag, type, status, pages, but some columns are cramped or cut. | Reference is more scannable and product-like. |
| Paths | Reference hides raw absolute paths. | Current rows show long local paths. | Makes the UI feel more technical and noisy. |
| Status | Reference status chips include indexed/in-progress states. | Current has simple `done` chips. | Less expressive and less localized. |
| Right detail | Reference has file header, tabs, collection, tags, recent citations, index status, and suggestions. | Current has stats, source preview, actions, and chunks, but no tab model. | Detail panel is useful but not structured like the reference. |
| Pagination | Reference has pagination and per-page controls. | Current appears as one long scrolling table. | Less controlled for large libraries. |

### Required Fixes

1. Move the Library toolbar into the main Library panel and make it match the reference toolbar density.
2. Reduce the upload zone into a secondary action or compact drop area; do not let it dominate the default file list.
3. Fix the type/group counts so PDF/Markdown/images/code counts reflect real rows.
4. Replace raw absolute paths in the table with source/location metadata that is less technical.
5. Add persistent collection and tag sections in the left filter rail.
6. Rework the detail panel into tabs: details, preview, content, related.
7. Add pagination or a clearer large-list control.

## Pair 3: Notes

Reference: `03-notes.png`  
Current: `phase29-notes-desktop.png`

### Major Differences

| Area | Reference | Current | Impact |
|---|---|---|---|
| Page title | `笔记` in Chinese. | `Notes` in English. | Localization mismatch. |
| Note editor | Full Markdown editor with title count, collection selector, tag input, toolbar, word count, mode indicator. | Simple title/collection/tag inputs and textarea. | Much less capable than the reference. |
| Editor toolbar | H1/H2/H3, bold, italic, quote, lists, checkbox, link, image, table, preview, expand. | No toolbar. | Reference writing experience not restored. |
| Web import | Has URL input plus extraction options and save destination. | Has URL/title/collection/tag fields and an explanation card. | Less control and less like the reference. |
| Knowledge output | Four template cards with selected state. | Single dropdown selector. | Less visual, less discoverable. |
| Model selector | Visible at generation footer. | Not visible in current Notes workflow. | Missing generation control from reference. |
| Right panel | Today knowledge outputs, saved answers, processing timeline. | Recent captures empty state and processing hint only. | Current right panel is much less useful. |
| Recent captures | Bottom table with files, collection, tags, status, time. | Not visible in current first screen. | Missing important workspace continuity. |
| Layout density | Reference uses a three-zone working dashboard. | Current leaves large blank areas and has taller card spacing. | Feels less finished and less information-rich. |

### Required Fixes

1. Localize Notes title and visible page labels consistently.
2. Add a real Markdown editor toolbar and metadata footer.
3. Add import options: extract text, keep images, extract metadata, save destination.
4. Replace the knowledge-output dropdown with template cards matching the reference.
5. Add saved answers and a real processing timeline to the right panel.
6. Restore the recent captures table to the first screen when data exists.

## Pair 4: Settings

Reference: `04-settings.png`  
Current: `phase29-settings-desktop.png`

Important note: the saved Settings reference still contains maintenance commands and recovery wording. The latest project rule explicitly forbids exposing those in the normal UI, so those parts should not be restored literally. The correct target is the same structure and density, but with ordinary user-facing wording.

### Major Differences

| Area | Reference | Current | Impact |
|---|---|---|---|
| Page title | `设置` in Chinese. | `Settings` in English. | Localization mismatch. |
| System status | Horizontal cards for SQLite, Qdrant, Python, ports, Ollama. | Tall vertical grouped status list. | Current consumes too much vertical space and does not match the reference. |
| Local models | Table with model type, status, current model, cache size, action. | Compact model selector plus two model rows. | Less informative than reference. |
| Watched folders | Table with folder names, paths, status, recursive toggle, edit/delete actions. | Right panel only says monitored folders exist; no table. | Missing reference's core Settings management surface. |
| Storage | Donut chart with category breakdown. | No storage visualization. | Major missing reference section. |
| Suggestions | Reference has actionable suggestion cards. | Current has status hints only. | Similar intent but not same structure. |
| Recent results | Reference has recent status list; latest rule says not to show maintenance commands. | Removed correctly. | Do not restore command names; replace with ordinary activity history if needed. |
| Warning card | Reference has optional model warning with buttons. | Current only says enhanced ability incomplete. | Missing clear ordinary-user next action. |
| Main balance | Reference fills left and right columns evenly. | Current left health column is tall; center/right have large blank areas below cards. | Layout looks unfinished. |

### Required Fixes

1. Localize Settings title and subtitle.
2. Convert system status into compact horizontal cards.
3. Replace the model selector-only view with a model table or richer model rows: type, status, current model, size/availability, action.
4. Add monitored-folder table with recursive toggle and edit/delete actions if those actions can be real.
5. Add storage usage card. If exact storage categories are unavailable, implement truthful local storage metrics first.
6. Add ordinary-user suggestion cards, not maintenance commands.
7. Add a clean optional-model warning/action area without terminal wording.

## Pair 5: Source Preview

Reference: `05-source-preview.png`  
Current: no matching dedicated page. Current closest surface is the Library right-side source review.

### Major Differences

| Area | Reference | Current | Impact |
|---|---|---|---|
| Dedicated view | Has `来源预览` as its own navigation/page state. | No dedicated Source Preview page. | The biggest visual and functional gap. |
| Search/result rail | Ranked source list with confidence labels and result count. | No equivalent page-level result rail. | Cannot review evidence like the reference. |
| Document viewer | Central PDF/document viewer with page controls, zoom, fit, highlights. | `openSourceByPath` opens raw preview in a new tab; Library panel shows chunks only. | Reference restoration is not implemented. |
| Highlighting | Exact answer-supporting spans are highlighted in the document. | Chunk cards show text snippets; no original-position highlight. | Evidence verification is weaker. |
| Citation detail | Right panel explains similarity, evidence strength, position, relation, keywords, parent context. | Current Library panel explains chunks and shows chunk cards. | Useful but much less complete. |
| Timeline | Source timeline/page marker is present. | No timeline. | Missing visual navigation. |
| Actions | Save as note, open original, rerun search, optimize search. | Current has view chunks, open original, save chunk. | Partial overlap only. |

### Required Fixes

1. Add a real Source Preview view or route.
2. Build a three-column source-review layout: ranked sources, document preview, citation detail.
3. Add PDF/page controls and text highlight where technically possible.
4. Carry citation metadata into the page: similarity, evidence strength, position weight, keywords, parent context.
5. Link Chat citations and Library source review into this page instead of only opening raw file previews.

## Cross-Page Differences

### Must Fix

- Current screenshots mix English page titles with Chinese UI body: `DocFlow`, `Notes`, `Settings`. Reference uses Chinese page titles.
- Sidebar bottom lacks the reference's storage usage bar and capacity text.
- Top-right status uses different controls: reference has local-running status, theme, add, menu; current has status, document/import-like icon, refresh, sliders. Current icons need clearer user meaning or should match the reference.
- Current UI uses too many placeholder/empty states in final screenshots. Reference uses representative populated states.
- Several pages expose raw local paths too prominently, especially Library.
- Current first-screen density is lower than the reference; large blank areas make the app feel less complete.
- Source Preview is the largest missing screen.

### Should Fix

- Align active navigation icon style and brand icon with the reference.
- Reduce oversized cards and keep page grids visually balanced.
- Make status chips consistently Chinese and product-facing.
- Add richer right-context panels on Chat and Notes.
- Add table/list footer controls where reference shows pagination, counts, or progress.

## Suggested Next Work Order

1. Source Preview dedicated page.
2. Chat answered-state screenshot and answer workflow parity.
3. Library density and metadata parity, including count correctness.
4. Notes editor and generation-template parity.
5. Settings structure parity with user-facing wording only.
6. Global polish: Chinese titles, storage footer, top-right controls, spacing balance.

## Current Restoration Estimate

| Page | Visual/structural parity estimate |
|---|---:|
| Chat / Overview | 45% |
| Library | 60% |
| Notes | 55% |
| Settings | 50% |
| Source Preview | 20% |

The UI is directionally aligned, but not yet a one-to-one restoration.
