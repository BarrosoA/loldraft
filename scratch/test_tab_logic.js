const fs = require('fs');

// Check that index.html contains all expected elements
const html = fs.readFileSync('docs/index.html', 'utf8');

const checks = [
  { name: 'hudSegmentedControl', test: html.includes('id="hudSegmentedControl"') },
  { name: 'role="tablist"', test: html.includes('role="tablist"') },
  { name: 'data-tab="best"', test: html.includes('data-tab="best"') },
  { name: 'data-tab="worst"', test: html.includes('data-tab="worst"') },
  { name: 'role="tab"', test: html.includes('role="tab"') },
  { name: 'aria-selected="true"', test: html.includes('aria-selected="true"') },
  { name: 'aria-controls="recommendationsList"', test: html.includes('aria-controls="recommendationsList"') },
  { name: 'app.js?v=2.1.5 cache buster', test: html.includes('app.js?v=2.1.5') },
  { name: 'style.css?v=2.1.6 cache buster', test: html.includes('style.css?v=2.1.6') },
  { name: 'center-workspace present in html', test: html.includes('class="center-workspace"') },
  { name: 'top roleButtons removed from html', test: !html.includes('id="roleButtons"') },
  { name: 'Drag order text removed from html', test: !html.includes('Drag order') },
  { name: 'Favorable Matchup legend removed from html', test: !html.includes('Favorable Matchup') },
  { name: '(BLUE) removed from title', test: !html.includes('(BLUE)') },
  { name: '(RED) removed from title', test: !html.includes('(RED)') },
  { name: 'hudCandidateMeta removed from html', test: !html.includes('hudCandidateMeta') }
];

console.log('--- HTML CHECKS ---');
checks.forEach(c => {
  console.log(`${c.name}: ${c.test ? 'PASS' : 'FAIL'}`);
  if (!c.test) process.exit(1);
});

// Check that app.js contains the modern tab control handler and dynamic role selection
const js = fs.readFileSync('docs/app.js', 'utf8');

const jsChecks = [
  { name: 'state.hudView initialized', test: js.includes("hudView: 'best'") },
  { name: 'segControl event delegation', test: js.includes("const btn = e.target.closest('.seg-btn');") },
  { name: 'setHudView function definition', test: js.includes("function setHudView(view)") },
  { name: 'aria-selected update', test: js.includes("btn.setAttribute('aria-selected', isSelected ? 'true' : 'false');") },
  { name: 'hudPanel data-view update', test: js.includes("hudPanel.setAttribute('data-view', view);") },
  { name: 'worst title prefix update', test: js.includes("hudTitlePrefix.textContent = view === 'worst' ? 'Picks to Avoid' : 'Suggested Picks'") },
  { name: 'worst card slice logic', test: js.includes("allCandidates.slice(-10).reverse()") },
  { name: 'rec-card click to assign removed', test: !js.slice(js.indexOf('function renderRecommendations')).includes("card.addEventListener('click'") },
  { name: 'empty search state handler', test: js.includes("empty-search-state") },
  { name: 'getActiveAllyRole function', test: js.includes("function getActiveAllyRole()") },
  { name: 'slot-active-badge rendered', test: js.includes("slot-active-badge") },
  { name: 'syncAlliesOrderFromDOM function', test: js.includes("function syncAlliesOrderFromDOM()") },
  { name: 'slot-role-abbr rendered', test: js.includes("slot-role-abbr") },
  { name: 'Risky Blind updated text format without trailing dot', test: js.includes("Risky Blind: Punished hard by counters (-${blindVuln.toFixed(1)}% avg)") },
  { name: 'Browser V8 Engine text removed from JS', test: !js.includes("Browser V8 Engine") }
];

console.log('\n--- JS CHECKS ---');
jsChecks.forEach(c => {
  console.log(`${c.name}: ${c.test ? 'PASS' : 'FAIL'}`);
  if (!c.test) process.exit(1);
});

// Check that style.css contains the styling rules for segmented control and champion grid
const css = fs.readFileSync('docs/style.css', 'utf8');

const cssChecks = [
  { name: '.segmented-control styles', test: css.includes('.segmented-control') },
  { name: '.seg-btn styles', test: css.includes('.seg-btn') },
  { name: '.seg-btn.active[data-tab="best"]', test: css.includes('.seg-btn.active[data-tab="best"]') },
  { name: '.seg-btn.active[data-tab="worst"]', test: css.includes('.seg-btn.active[data-tab="worst"]') },
  { name: '.hud-panel[data-view="worst"] styling', test: css.includes('.hud-panel[data-view="worst"]') },
  { name: '.worst-pick styling', test: css.includes('.worst-pick') },
  { name: '.champions-grid align-content: start', test: css.includes('align-content: start') },
  { name: '.champions-grid grid-auto-rows: max-content', test: css.includes('grid-auto-rows: max-content') },
  { name: '.champ-card height: fit-content', test: css.includes('height: fit-content') },
  { name: '.empty-search-state styles', test: css.includes('.empty-search-state') },
  { name: '.slot-active-badge styles', test: css.includes('.slot-active-badge') },
  { name: 'grid-template-columns 350px width', test: css.includes('350px 1fr 350px') },
  { name: '.slot-role-abbr styles', test: css.includes('.slot-role-abbr') },
  { name: '.rec-card hover animations/styles completely removed', test: !css.includes('.rec-card:hover') && !css.includes('.rec-card.worst-pick:hover') },
  { name: '.team-panel align-self: start', test: css.includes('align-self: start;') },
  { name: '.recommendations-carousel 2-row wrapping grid', test: css.includes('repeat(auto-fill, minmax(210px, 1fr))') },
  { name: '.rec-card cursor default', test: css.includes('cursor: default;') }
];

console.log('\n--- CSS CHECKS ---');
cssChecks.forEach(c => {
  console.log(`${c.name}: ${c.test ? 'PASS' : 'FAIL'}`);
  if (!c.test) process.exit(1);
});

console.log('\nALL VERIFICATION CHECKS PASSED SUCCESSFULLY!');

