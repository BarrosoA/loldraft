// LolDraft Interactive Simulator Client (Standalone Client-Side Engine)
let MATRIX = null;
let catalog = [];
let assignedRole = 'top';

const ALL_ROLES = ["top", "jungle", "middle", "bottom", "support"];
const ROLE_NAMES = {
  top: 'TOP',
  jungle: 'JGL',
  middle: 'MID',
  bottom: 'BOT',
  support: 'SUP'
};
const EPSILON = 0.005;

// Draft state
const state = {
  allies: [
    { role: 'TOP', cid: null, name: '', icon: '' },
    { role: 'JGL', cid: null, name: '', icon: '' },
    { role: 'MID', cid: null, name: '', icon: '' },
    { role: 'BOT', cid: null, name: '', icon: '' },
    { role: 'SUP', cid: null, name: '', icon: '' }
  ],
  enemies: [
    { slotIndex: 1, cid: null, name: '', icon: '', inference: '' },
    { slotIndex: 2, cid: null, name: '', icon: '', inference: '' },
    { slotIndex: 3, cid: null, name: '', icon: '', inference: '' },
    { slotIndex: 4, cid: null, name: '', icon: '', inference: '' },
    { slotIndex: 5, cid: null, name: '', icon: '', inference: '' }
  ],
  activeTarget: { team: 'ally', index: 0 },
  searchFilter: '',
  roleFilter: 'all'
};

// DOM Elements
const allySlotsContainer = document.getElementById('allySlots');
const enemySlotsContainer = document.getElementById('enemySlots');
const championsGrid = document.getElementById('championsGrid');
const recommendationsList = document.getElementById('recommendationsList');
const searchInput = document.getElementById('searchInput');
const clearSearchBtn = document.getElementById('clearSearchBtn');
const roleFilterPills = document.getElementById('pickerRoleFilter');
const roleButtons = document.getElementById('roleButtons');
const hudRoleLabel = document.getElementById('hudRoleLabel');
const hudCandidateMeta = document.getElementById('hudCandidateMeta');
const allyCountBadge = document.getElementById('allyCount');
const enemyCountBadge = document.getElementById('enemyCount');
const bannerTargetName = document.getElementById('bannerTargetName');
const dmgPhysBar = document.getElementById('dmgPhysBar');
const dmgMagicBar = document.getElementById('dmgMagicBar');
const physPctLabel = document.getElementById('physPctLabel');
const magicPctLabel = document.getElementById('magicPctLabel');
const damageSummaryText = document.getElementById('damageSummaryText');
const resetBtn = document.getElementById('resetBtn');
const patchBadge = document.getElementById('patchBadge');

// ============================================================================
// Bayesian Role Inference & Scoring in Client-Side JavaScript
// ============================================================================

function getPermutations(arr, k) {
  if (k === 1) return arr.map(el => [el]);
  const perms = [];
  arr.forEach((el, i) => {
    const rest = arr.filter((_, idx) => idx !== i);
    const subPerms = getPermutations(rest, k - 1);
    subPerms.forEach(sub => perms.push([el, ...sub]));
  });
  return perms;
}

function inferRolesJS(lockedEnemyCids) {
  if (!lockedEnemyCids || lockedEnemyCids.length === 0) return {};
  const champions = MATRIX.champions || {};
  const k = lockedEnemyCids.length;

  const enemyPriors = {};
  for (const cid of lockedEnemyCids) {
    const champ = champions[cid] || {};
    const priors = champ.role_priors || {};
    const dist = {};
    for (const r of ALL_ROLES) {
      dist[r] = Math.max(priors[r] || 0.0, EPSILON);
    }
    const total = Object.values(dist).reduce((a, b) => a + b, 0);
    enemyPriors[cid] = {};
    for (const r of ALL_ROLES) {
      enemyPriors[cid][r] = dist[r] / total;
    }
  }

  if (k === 1) return enemyPriors;

  let totalLikelihood = 0.0;
  const roleJointSums = {};
  lockedEnemyCids.forEach(cid => {
    roleJointSums[cid] = { top: 0, jungle: 0, middle: 0, bottom: 0, support: 0 };
  });

  const perms = getPermutations(ALL_ROLES, k);
  for (const perm of perms) {
    let assignmentProb = 1.0;
    for (let i = 0; i < k; i++) {
      const cid = lockedEnemyCids[i];
      const assigned = perm[i];
      assignmentProb *= enemyPriors[cid][assigned];
    }
    totalLikelihood += assignmentProb;
    for (let i = 0; i < k; i++) {
      const cid = lockedEnemyCids[i];
      const assigned = perm[i];
      roleJointSums[cid][assigned] += assignmentProb;
    }
  }

  const results = {};
  for (const cid of lockedEnemyCids) {
    results[cid] = {};
    for (const r of ALL_ROLES) {
      const prob = totalLikelihood > 0 ? roleJointSums[cid][r] / totalLikelihood : 0.2;
      results[cid][r] = Math.round(prob * 10000) / 10000;
    }
  }
  return results;
}

function getChampionDamageProfile(cid, preferredRole) {
  const champ = MATRIX.champions[cid];
  if (!champ) return { pct_physical: 50, pct_magic: 50, pct_true: 0 };
  const roles = champ.roles || {};
  if (preferredRole && roles[preferredRole] && roles[preferredRole].damage_profile) {
    return roles[preferredRole].damage_profile;
  }
  const priors = champ.role_priors || {};
  const topRole = Object.entries(priors).sort((a, b) => b[1] - a[1])[0]?.[0] || Object.keys(roles)[0];
  if (topRole && roles[topRole] && roles[topRole].damage_profile) {
    return roles[topRole].damage_profile;
  }
  return { pct_physical: 50, pct_magic: 50, pct_true: 0 };
}

function scoreCandidateJS(candidateCid, targetRole, lockedAllies, lockedEnemies, enemyRoles) {
  const champ = MATRIX.champions[candidateCid];
  if (!champ) return null;
  const candName = champ.name;
  const roles = champ.roles || {};
  if (!roles[targetRole]) return null;

  const roleData = roles[targetRole];
  const baselineWr = roleData.baseline_wr || 50.0;
  const blindVuln = roleData.blind_vulnerability || 1.5;
  const candDmg = roleData.damage_profile || { pct_physical: 50, pct_magic: 50, pct_true: 0 };
  const counters = roleData.counters || {};
  const threats = roleData.threats || {};
  const synergies = roleData.synergies || {};
  const weights = MATRIX.weights || { base: 1.0, lane: 1.35, synergy: 0.85, threat: 0.60, blind: 0.90 };

  let expectedLaneDelta = 0.0;
  let expectedThreatDelta = 0.0;
  let totalLaneProb = 0.0;

  for (const eCid of lockedEnemies) {
    const pLane = enemyRoles[eCid]?.[targetRole] || 0.0;
    totalLaneProb += pLane;
    const laneDelta = counters[eCid] || 0.0;
    const threatDelta = threats[eCid] || 0.0;
    expectedLaneDelta += pLane * laneDelta;
    expectedThreatDelta += (1.0 - pLane) * threatDelta;
  }
  totalLaneProb = Math.min(totalLaneProb, 1.0);

  let totalSynergyDelta = 0.0;
  for (const aCid of lockedAllies) {
    totalSynergyDelta += synergies[aCid] || 0.0;
  }

  let compAdjustment = 0.0;
  const rationale = [];
  let projectedPhys = candDmg.pct_physical || 50.0;
  let projectedMagic = candDmg.pct_magic || 50.0;

  if (lockedAllies.length > 0) {
    const alliesDmg = lockedAllies.map(aCid => getChampionDamageProfile(aCid, null));
    const allDmgList = [...alliesDmg, candDmg];
    projectedPhys = allDmgList.reduce((s, d) => s + (d.pct_physical || 0), 0) / allDmgList.length;
    projectedMagic = allDmgList.reduce((s, d) => s + (d.pct_magic || 0), 0) / allDmgList.length;

    const existingPhys = alliesDmg.reduce((s, d) => s + (d.pct_physical || 0), 0) / alliesDmg.length;
    const existingMagic = alliesDmg.reduce((s, d) => s + (d.pct_magic || 0), 0) / alliesDmg.length;

    if (lockedAllies.length >= 2) {
      if (existingPhys >= 75.0) {
        if ((candDmg.pct_physical || 0) >= 65.0) {
          compAdjustment -= 3.0;
          rationale.push("Composition Penalty: Heavy Physical redundancy (-3.00%). Enemy team can easily stack Armor.");
        } else if ((candDmg.pct_magic || 0) >= 60.0) {
          compAdjustment += 2.5;
          rationale.push("Composition Bonus: Critical AP diversification (+2.50%). Prevents enemy Armor stacking.");
        }
      } else if (existingMagic >= 75.0) {
        if ((candDmg.pct_magic || 0) >= 65.0) {
          compAdjustment -= 3.0;
          rationale.push("Composition Penalty: Heavy Magic redundancy (-3.00%). Enemy team can easily stack Magic Resist.");
        } else if ((candDmg.pct_physical || 0) >= 60.0) {
          compAdjustment += 2.5;
          rationale.push("Composition Bonus: Critical AD diversification (+2.50%). Prevents enemy Magic Resist stacking.");
        }
      }
    }
  }

  const isBlind = totalLaneProb < 0.25;
  let blindPenalty = 0.0;
  if (isBlind) {
    const unrevealedFactor = 1.0 - totalLaneProb;
    blindPenalty = weights.blind * blindVuln * unrevealedFactor;
    if (blindVuln < 2.0) {
      rationale.push(`Safe Blind: Low vulnerability rating (${blindVuln.toFixed(1)}% avg counter severity).`);
    } else {
      rationale.push(`Risky Blind: Punished hard by counters (${blindVuln.toFixed(1)}% avg counter severity).`);
    }
  } else {
    if (expectedLaneDelta > 1.0) {
      rationale.push(`Favorable Lane Matchup (+${expectedLaneDelta.toFixed(2)}% expected delta).`);
    } else if (expectedLaneDelta < -1.0) {
      rationale.push(`Unfavorable Lane Matchup (${expectedLaneDelta.toFixed(2)}% expected delta).`);
    }
  }

  if (totalSynergyDelta > 0.8) {
    rationale.push(`Strong Team Synergy (+${totalSynergyDelta.toFixed(2)}%).`);
  } else if (totalSynergyDelta < -0.8) {
    rationale.push(`Negative Team Synergy (${totalSynergyDelta.toFixed(2)}%).`);
  }

  const compositeScore = (
    weights.base * baselineWr
    + weights.lane * expectedLaneDelta
    + weights.threat * expectedThreatDelta
    + weights.synergy * totalSynergyDelta
    + compAdjustment
    - blindPenalty
  );

  return {
    cid: candidateCid,
    name: candName,
    role: targetRole,
    viable: true,
    composite_score: Math.round(compositeScore * 100) / 100,
    baseline_wr: baselineWr,
    expected_lane_delta: Math.round(expectedLaneDelta * 100) / 100,
    expected_threat_delta: Math.round(expectedThreatDelta * 100) / 100,
    synergy_delta: Math.round(totalSynergyDelta * 100) / 100,
    composition_adjustment: Math.round(compAdjustment * 100) / 100,
    blind_penalty: Math.round(blindPenalty * 100) / 100,
    lane_opponent_revealed_prob: Math.round(totalLaneProb * 100) / 100,
    damage_profile: candDmg,
    projected_team_damage: {
      pct_physical: Math.round(projectedPhys * 10) / 10,
      pct_magic: Math.round(projectedMagic * 10) / 10
    },
    icon: `https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/${candidateCid}.png`,
    rationale: rationale
  };
}

// ============================================================================
// Initialization & Data Loading
// ============================================================================

async function init() {
  setupEventListeners();
  hudCandidateMeta.textContent = 'Loading 4 MB matrix into browser RAM...';

  try {
    // 1. Load standalone matrix file (works on GitHub Pages & Vercel)
    let res = await fetch('./matrix.json');
    if (!res.ok) {
      res = await fetch('/api/champions');
    }
    MATRIX = await res.json();

    // If loaded from /api/champions wrapper, fetch raw matrix
    if (!MATRIX.champions || Array.isArray(MATRIX.champions)) {
      const fullRes = await fetch('./matrix.json');
      MATRIX = await fullRes.json();
    }

    if (MATRIX.patch) {
      patchBadge.innerHTML = `<span class="pulse-dot"></span> PATCH ${MATRIX.patch} • EMERALD+`;
    }

    // Build catalog from matrix
    catalog = Object.entries(MATRIX.champions).map(([cid, champ]) => ({
      cid: cid,
      name: champ.name,
      slug: champ.slug,
      roles: Object.keys(champ.roles || {}),
      role_priors: champ.role_priors || {},
      icon: `https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/${cid}.png`
    }));
    catalog.sort((a, b) => a.name.localeCompare(b.name));

    renderDraftSlots();
    renderChampionGrid();
    computeRecommendationsClientSide();
  } catch (err) {
    console.error('Failed to load matrix:', err);
    hudCandidateMeta.textContent = 'Error loading matrix data.';
  }
}

// Setup Event Listeners
function setupEventListeners() {
  // Role buttons
  roleButtons.addEventListener('click', (e) => {
    const btn = e.target.closest('.role-btn');
    if (!btn) return;
    document.querySelectorAll('.role-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    assignedRole = btn.dataset.role;
    hudRoleLabel.textContent = ROLE_NAMES[assignedRole] || assignedRole.toUpperCase();
    computeRecommendationsClientSide();
  });

  // Search input
  searchInput.addEventListener('input', (e) => {
    state.searchFilter = e.target.value.toLowerCase().trim();
    renderChampionGrid();
  });

  clearSearchBtn.addEventListener('click', () => {
    searchInput.value = '';
    state.searchFilter = '';
    renderChampionGrid();
  });

  // Role filter pills for champion picker
  roleFilterPills.addEventListener('click', (e) => {
    const pill = e.target.closest('.filter-pill');
    if (!pill) return;
    document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    state.roleFilter = pill.dataset.filter;
    renderChampionGrid();
  });

  // Reset Draft
  resetBtn.addEventListener('click', () => {
    state.allies.forEach(a => { a.cid = null; a.name = ''; a.icon = ''; });
    state.enemies.forEach(e => { e.cid = null; e.name = ''; e.icon = ''; e.inference = ''; });
    state.activeTarget = { team: 'ally', index: 0 };
    renderDraftSlots();
    renderChampionGrid();
    computeRecommendationsClientSide();
  });
}

let draggedAllyIndex = null;
let isDraggingAlly = false;

function attachAllySlotDragHandlers(card, idx) {
  card.addEventListener('dragstart', (e) => {
    isDraggingAlly = true;
    draggedAllyIndex = idx;
    card.classList.add('is-dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', String(idx));
  });

  card.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const rect = card.getBoundingClientRect();
    const isAfter = (e.clientY - rect.top) > (rect.height / 2);
    card.classList.remove('drag-over-top', 'drag-over-bottom');
    if (isAfter) {
      card.classList.add('drag-over-bottom');
    } else {
      card.classList.add('drag-over-top');
    }
  });

  card.addEventListener('dragleave', () => {
    card.classList.remove('drag-over-top', 'drag-over-bottom');
  });

  card.addEventListener('dragend', () => {
    card.classList.remove('is-dragging');
    document.querySelectorAll('.slot-card').forEach(c => {
      c.classList.remove('drag-over-top', 'drag-over-bottom', 'is-dragging');
    });
    setTimeout(() => {
      isDraggingAlly = false;
      draggedAllyIndex = null;
    }, 60);
  });

  card.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();

    document.querySelectorAll('.slot-card').forEach(c => {
      c.classList.remove('drag-over-top', 'drag-over-bottom', 'is-dragging');
    });

    const fromIndex = draggedAllyIndex !== null ? draggedAllyIndex : parseInt(e.dataTransfer.getData('text/plain'), 10);
    if (isNaN(fromIndex) || fromIndex === null || fromIndex < 0 || fromIndex >= state.allies.length) return;

    const rect = card.getBoundingClientRect();
    const isAfter = (e.clientY - rect.top) > (rect.height / 2);
    let dropIndex = idx;
    if (isAfter) dropIndex++;
    if (fromIndex < dropIndex) dropIndex--;

    if (fromIndex !== dropIndex) {
      const activeAllyRef = (state.activeTarget.team === 'ally' && state.allies[state.activeTarget.index])
        ? state.allies[state.activeTarget.index]
        : null;

      const [movedItem] = state.allies.splice(fromIndex, 1);
      state.allies.splice(dropIndex, 0, movedItem);

      if (activeAllyRef) {
        const newActiveIdx = state.allies.indexOf(activeAllyRef);
        if (newActiveIdx !== -1) {
          state.activeTarget.index = newActiveIdx;
        }
      }

      renderDraftSlots();
      computeRecommendationsClientSide();
    }
  });

  // Touch support for touchscreen devices
  card.addEventListener('touchstart', (e) => {
    if (e.target.closest('.slot-remove-btn')) return;
    const touch = e.touches[0];
    card._touchStartY = touch.clientY;
    card._touchStarted = false;
  }, { passive: true });

  card.addEventListener('touchmove', (e) => {
    const touch = e.touches[0];
    const diff = Math.abs(touch.clientY - (card._touchStartY || touch.clientY));
    if (diff > 8 && !card._touchStarted) {
      card._touchStarted = true;
      isDraggingAlly = true;
      draggedAllyIndex = idx;
      card.classList.add('is-dragging');
    }
    if (card._touchStarted) {
      e.preventDefault();
      const targetEl = document.elementFromPoint(touch.clientX, touch.clientY);
      const targetCard = targetEl ? targetEl.closest('.slot-card.draggable-slot') : null;
      document.querySelectorAll('.slot-card').forEach(c => c.classList.remove('drag-over-top', 'drag-over-bottom'));
      if (targetCard) {
        const rect = targetCard.getBoundingClientRect();
        const isAfter = (touch.clientY - rect.top) > (rect.height / 2);
        targetCard.classList.add(isAfter ? 'drag-over-bottom' : 'drag-over-top');
      }
    }
  }, { passive: false });

  card.addEventListener('touchend', (e) => {
    if (card._touchStarted) {
      const touch = e.changedTouches[0];
      const targetEl = document.elementFromPoint(touch.clientX, touch.clientY);
      const targetCard = targetEl ? targetEl.closest('.slot-card.draggable-slot') : null;
      const fromIndex = draggedAllyIndex;

      document.querySelectorAll('.slot-card').forEach(c => {
        c.classList.remove('drag-over-top', 'drag-over-bottom', 'is-dragging');
      });

      if (targetCard && fromIndex !== null) {
        const targetIdx = parseInt(targetCard.dataset.idx, 10);
        if (!isNaN(targetIdx)) {
          const rect = targetCard.getBoundingClientRect();
          const isAfter = (touch.clientY - rect.top) > (rect.height / 2);
          let dropIndex = targetIdx;
          if (isAfter) dropIndex++;
          if (fromIndex < dropIndex) dropIndex--;

          if (fromIndex !== dropIndex) {
            const activeAllyRef = (state.activeTarget.team === 'ally' && state.allies[state.activeTarget.index])
              ? state.allies[state.activeTarget.index]
              : null;

            const [movedItem] = state.allies.splice(fromIndex, 1);
            state.allies.splice(dropIndex, 0, movedItem);

            if (activeAllyRef) {
              const newActiveIdx = state.allies.indexOf(activeAllyRef);
              if (newActiveIdx !== -1) {
                state.activeTarget.index = newActiveIdx;
              }
            }

            renderDraftSlots();
            computeRecommendationsClientSide();
          }
        }
      }

      setTimeout(() => {
        isDraggingAlly = false;
        draggedAllyIndex = null;
      }, 60);
    }
  });
}

// Render Draft Slots (Allies and Enemies)
function renderDraftSlots() {
  // 1. Allies
  allySlotsContainer.innerHTML = '';
  let lockedAllies = 0;
  state.allies.forEach((slot, idx) => {
    if (slot.cid) lockedAllies++;
    const isTarget = state.activeTarget.team === 'ally' && state.activeTarget.index === idx;
    const card = document.createElement('div');
    card.className = `slot-card draggable-slot ${isTarget ? 'active-target' : ''}`;
    card.draggable = true;
    card.dataset.idx = idx;
    card.innerHTML = `
      <div class="slot-drag-handle" title="Drag to reorder pick order">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
          <circle cx="8" cy="5" r="2.2"/>
          <circle cx="16" cy="5" r="2.2"/>
          <circle cx="8" cy="12" r="2.2"/>
          <circle cx="16" cy="12" r="2.2"/>
          <circle cx="8" cy="19" r="2.2"/>
          <circle cx="16" cy="19" r="2.2"/>
        </svg>
      </div>
      <div class="slot-avatar">
        ${slot.icon ? `<img src="${slot.icon}" alt="${slot.name}">` : `<span>${getRoleEmoji(slot.role)}</span>`}
      </div>
      <div class="slot-meta">
        <div class="slot-role-row">
          <span class="slot-pick-num">#${idx + 1}</span>
          <span class="slot-role-tag">${slot.role}</span>
        </div>
        <span class="slot-champ-name">${slot.name || 'Empty Slot'}</span>
      </div>
      ${slot.cid ? `<button class="slot-remove-btn" title="Remove" data-team="ally" data-idx="${idx}">✕</button>` : ''}
    `;

    card.addEventListener('click', (e) => {
      if (isDraggingAlly) return;
      if (e.target.closest('.slot-remove-btn')) return;
      state.activeTarget = { team: 'ally', index: idx };

      // Sync active HUD role with clicked slot role
      const roleMap = { 'TOP': 'top', 'JGL': 'jungle', 'MID': 'middle', 'BOT': 'bottom', 'SUP': 'support' };
      const targetRole = roleMap[slot.role];
      if (targetRole && assignedRole !== targetRole) {
        assignedRole = targetRole;
        document.querySelectorAll('.role-btn').forEach(b => {
          b.classList.toggle('active', b.dataset.role === assignedRole);
        });
        hudRoleLabel.textContent = ROLE_NAMES[assignedRole] || assignedRole.toUpperCase();
      }

      updateTargetBanner();
      renderDraftSlots();
      computeRecommendationsClientSide();
    });

    attachAllySlotDragHandlers(card, idx);

    allySlotsContainer.appendChild(card);
  });
  allyCountBadge.textContent = `${lockedAllies}/5 Locked`;

  // 2. Enemies
  enemySlotsContainer.innerHTML = '';
  let lockedEnemies = 0;
  state.enemies.forEach((slot, idx) => {
    if (slot.cid) lockedEnemies++;
    const isTarget = state.activeTarget.team === 'enemy' && state.activeTarget.index === idx;
    const card = document.createElement('div');
    card.className = `slot-card ${isTarget ? 'active-target' : ''}`;
    card.innerHTML = `
      <div class="slot-avatar">
        ${slot.icon ? `<img src="${slot.icon}" alt="${slot.name}">` : `<span>⚔️</span>`}
      </div>
      <div class="slot-meta">
        <span class="slot-role-tag">Enemy #${idx + 1}</span>
        <span class="slot-champ-name">${slot.name || 'Unrevealed'}</span>
        ${slot.inference ? `<span class="slot-inference-tag">${slot.inference}</span>` : ''}
      </div>
      ${slot.cid ? `<button class="slot-remove-btn" title="Remove" data-team="enemy" data-idx="${idx}">✕</button>` : ''}
    `;

    card.addEventListener('click', (e) => {
      if (e.target.closest('.slot-remove-btn')) return;
      state.activeTarget = { team: 'enemy', index: idx };
      updateTargetBanner();
      renderDraftSlots();
    });

    enemySlotsContainer.appendChild(card);
  });
  enemyCountBadge.textContent = `${lockedEnemies}/5 Locked`;

  // Remove button handler
  document.querySelectorAll('.slot-remove-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const team = btn.dataset.team;
      const idx = parseInt(btn.dataset.idx, 10);
      if (team === 'ally') {
        state.allies[idx].cid = null;
        state.allies[idx].name = '';
        state.allies[idx].icon = '';
      } else {
        state.enemies[idx].cid = null;
        state.enemies[idx].name = '';
        state.enemies[idx].icon = '';
        state.enemies[idx].inference = '';
      }
      renderDraftSlots();
      renderChampionGrid();
      computeRecommendationsClientSide();
    });
  });

  updateTargetBanner();
}

function updateTargetBanner() {
  const { team, index } = state.activeTarget;
  if (team === 'ally') {
    bannerTargetName.textContent = `Ally (${state.allies[index].role})`;
    bannerTargetName.style.color = 'var(--blue-team)';
  } else {
    bannerTargetName.textContent = `Enemy Slot #${index + 1}`;
    bannerTargetName.style.color = 'var(--red-team)';
  }
}

// Render Champion Picker Grid
function renderChampionGrid() {
  const pickedCids = new Set([
    ...state.allies.map(a => a.cid).filter(Boolean),
    ...state.enemies.map(e => e.cid).filter(Boolean)
  ]);

  const filtered = catalog.filter(c => {
    const matchesSearch = c.name.toLowerCase().includes(state.searchFilter);
    const matchesRole = state.roleFilter === 'all' || (c.roles && c.roles.includes(state.roleFilter));
    return matchesSearch && matchesRole;
  });

  championsGrid.innerHTML = '';
  filtered.forEach(champ => {
    const isPicked = pickedCids.has(champ.cid);
    const card = document.createElement('div');
    card.className = `champ-card ${isPicked ? 'picked' : ''}`;
    card.innerHTML = `
      <div class="champ-avatar-wrap">
        <img src="${champ.icon}" alt="${champ.name}" loading="lazy">
      </div>
      <span class="champ-name-label">${champ.name}</span>
    `;

    if (!isPicked) {
      card.addEventListener('click', () => {
        assignChampion(champ);
      });
    }

    championsGrid.appendChild(card);
  });
}

// Assign Champion to active target slot
function assignChampion(champ) {
  const { team, index } = state.activeTarget;
  if (team === 'ally') {
    state.allies[index].cid = champ.cid;
    state.allies[index].name = champ.name;
    state.allies[index].icon = champ.icon;
    const nextEmpty = state.allies.findIndex(a => !a.cid);
    if (nextEmpty !== -1) {
      state.activeTarget = { team: 'ally', index: nextEmpty };
    }
  } else {
    state.enemies[index].cid = champ.cid;
    state.enemies[index].name = champ.name;
    state.enemies[index].icon = champ.icon;
    const nextEmpty = state.enemies.findIndex(e => !e.cid);
    if (nextEmpty !== -1) {
      state.activeTarget = { team: 'enemy', index: nextEmpty };
    }
  }

  renderDraftSlots();
  renderChampionGrid();
  computeRecommendationsClientSide();
}

// Compute Recommendations (100% In-Browser JavaScript Engine)
function computeRecommendationsClientSide() {
  if (!MATRIX || !MATRIX.champions) return;

  const t0 = performance.now();
  const lockedAllies = state.allies.map(a => a.cid).filter(Boolean);
  const lockedEnemies = state.enemies.map(e => e.cid).filter(Boolean);

  // 1. Infer enemy roles
  const inferredRoles = inferRolesJS(lockedEnemies);

  // Update enemy inferred role tags
  state.enemies.forEach(e => {
    if (e.cid && inferredRoles[e.cid]) {
      const dist = inferredRoles[e.cid];
      const topRole = Object.entries(dist).sort((a, b) => b[1] - a[1])[0];
      if (topRole) {
        e.inference = `Inferred: ${topRole[0].toUpperCase()} (${Math.round(topRole[1] * 100)}%)`;
      }
    }
  });
  renderDraftSlots();

  // 2. Score candidates
  const candidates = [];
  const pickedSet = new Set([...lockedAllies, ...lockedEnemies]);

  for (const cid of Object.keys(MATRIX.champions)) {
    if (pickedSet.has(cid)) continue;
    const res = scoreCandidateJS(cid, assignedRole, lockedAllies, lockedEnemies, inferredRoles);
    if (res && res.viable) {
      candidates.push(res);
    }
  }

  candidates.sort((a, b) => b.composite_score - a.composite_score);
  const elapsed = (performance.now() - t0).toFixed(1);

  hudCandidateMeta.textContent = `Scored ${candidates.length} viable candidates in ${elapsed}ms (Browser V8 Engine)`;

  // 3. Team Damage Balance Meter
  if (lockedAllies.length > 0) {
    const alliesDmg = lockedAllies.map(aCid => getChampionDamageProfile(aCid, null));
    const p = Math.round(alliesDmg.reduce((s, d) => s + (d.pct_physical || 0), 0) / alliesDmg.length);
    const m = Math.round(alliesDmg.reduce((s, d) => s + (d.pct_magic || 0), 0) / alliesDmg.length);

    dmgPhysBar.style.width = `${p}%`;
    dmgMagicBar.style.width = `${m}%`;
    physPctLabel.textContent = `${p}%`;
    magicPctLabel.textContent = `${m}%`;
    damageSummaryText.textContent = `${p}% AD / ${m}% AP`;

    if (p >= 75) {
      damageSummaryText.textContent = `⚠️ Heavy AD (${p}%) - Vulnerable to Armor`;
      damageSummaryText.style.color = '#e74c3c';
    } else if (m >= 75) {
      damageSummaryText.textContent = `⚠️ Heavy AP (${m}%) - Vulnerable to MR`;
      damageSummaryText.style.color = '#e74c3c';
    } else {
      damageSummaryText.style.color = 'var(--text-muted)';
    }
  } else {
    dmgPhysBar.style.width = '50%';
    dmgMagicBar.style.width = '50%';
    physPctLabel.textContent = '50%';
    magicPctLabel.textContent = '50%';
    damageSummaryText.textContent = '50% AD / 50% AP';
    damageSummaryText.style.color = 'var(--text-muted)';
  }

  // 4. Render HUD cards
  renderRecommendations(candidates.slice(0, 10));
}

// Render Recommendation HUD Cards
function renderRecommendations(recs) {
  recommendationsList.innerHTML = '';
  if (recs.length === 0) {
    recommendationsList.innerHTML = `<div style="padding: 14px; color: var(--text-dim); font-size: 13px;">No viable champions found for ${assignedRole.toUpperCase()}.</div>`;
    return;
  }

  recs.forEach((item, index) => {
    const card = document.createElement('div');
    card.className = `rec-card ${index === 0 ? 'rank-1' : ''}`;

    const laneDeltaClass = item.expected_lane_delta >= 0 ? 'delta-pos' : 'delta-neg';
    const laneDeltaSign = item.expected_lane_delta >= 0 ? '+' : '';
    const synDeltaClass = item.synergy_delta >= 0 ? 'delta-pos' : 'delta-neg';
    const synDeltaSign = item.synergy_delta >= 0 ? '+' : '';

    const badgesHtml = (item.rationale || []).map(r => {
      const isWarn = r.toLowerCase().includes('risky') || r.toLowerCase().includes('penalty') || r.toLowerCase().includes('negative');
      return `<span class="rec-badge ${isWarn ? 'badge-warn' : 'badge-good'}">${r}</span>`;
    }).join('');

    card.innerHTML = `
      <div class="rec-card-top">
        <span class="rec-rank-tag">#${index + 1}</span>
        <div class="rec-avatar">
          <img src="${item.icon}" alt="${item.name}" loading="lazy">
        </div>
        <div class="rec-champ-info">
          <span class="rec-champ-name">${item.name}</span>
          <span class="rec-score-value">${item.composite_score}%</span>
        </div>
      </div>

      <div class="rec-deltas-row">
        <span class="rec-delta-pill">Base: ${item.baseline_wr}%</span>
        <span class="rec-delta-pill">Lane: <span class="${laneDeltaClass}">${laneDeltaSign}${item.expected_lane_delta}%</span></span>
        <span class="rec-delta-pill">Syn: <span class="${synDeltaClass}">${synDeltaSign}${item.synergy_delta}%</span></span>
      </div>

      <div class="rec-badges-row">
        ${badgesHtml}
      </div>
    `;

    recommendationsList.appendChild(card);
  });
}

function getRoleEmoji(role) {
  switch (role) {
    case 'TOP': return '🛡️';
    case 'JGL': return '🌲';
    case 'MID': return '⚡';
    case 'BOT': return '🏹';
    case 'SUP': return '✨';
    default: return '⬢';
  }
}

// Start application
window.addEventListener('DOMContentLoaded', init);
