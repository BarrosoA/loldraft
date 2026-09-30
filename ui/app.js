// LolDraft Interactive Simulator Client (Standalone Client-Side Engine)
let MATRIX = null;
let catalog = [];
const ROLE_TO_KEY = {
  'TOP': 'top',
  'JGL': 'jungle',
  'MID': 'middle',
  'BOT': 'bottom',
  'SUP': 'support',
  'top': 'top',
  'jungle': 'jungle',
  'middle': 'middle',
  'bottom': 'bottom',
  'support': 'support'
};

function getActiveAllyRole() {
  if (state.activeTarget.team === 'ally' && state.allies[state.activeTarget.index]) {
    const rawRole = state.allies[state.activeTarget.index].role;
    return ROLE_TO_KEY[rawRole] || rawRole.toLowerCase() || 'top';
  }
  const emptyAlly = state.allies.find(a => !a.cid) || state.allies[0];
  const rawRole = emptyAlly ? emptyAlly.role : 'TOP';
  return ROLE_TO_KEY[rawRole] || 'top';
}

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
    { id: 'top', role: 'TOP', cid: null, name: '', icon: '' },
    { id: 'jgl', role: 'JGL', cid: null, name: '', icon: '' },
    { id: 'mid', role: 'MID', cid: null, name: '', icon: '' },
    { id: 'bot', role: 'BOT', cid: null, name: '', icon: '' },
    { id: 'sup', role: 'SUP', cid: null, name: '', icon: '' }
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
  roleFilter: 'all',
  hudView: 'best'
};

// DOM Elements
const allySlotsContainer = document.getElementById('allySlots');
const enemySlotsContainer = document.getElementById('enemySlots');
const championsGrid = document.getElementById('championsGrid');
const recommendationsList = document.getElementById('recommendationsList');
const searchInput = document.getElementById('searchInput');
const clearSearchBtn = document.getElementById('clearSearchBtn');
const roleFilterPills = document.getElementById('pickerRoleFilter');
const hudRoleLabel = document.getElementById('hudRoleLabel');
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
const tabBest = document.getElementById('tabBest');
const tabWorst = document.getElementById('tabWorst');
const hudIndicator = document.getElementById('hudIndicator');
const hudTitlePrefix = document.getElementById('hudTitlePrefix');
const hudTitleHeading = document.getElementById('hudTitleHeading');

let currentScoredCandidates = [];
let activeHudTab = 'best'; // 'best' | 'worst'

// Recognized Armor-scaling / Heavy Vanguard Tank Champions
const ARMOR_STACKING_CHAMPIONS = new Set([
  '54', '33', '897', '78', '44', '516', '14', '89', '111', '98', '31', '154', '113', '57', '223', '201', '12'
]);

const MR_STACKING_CHAMPIONS = new Set([
  '38', '3', '27', '516', '57', '32', '86'
]);

const ROLE_AP_PRIORS = {
  'MID': 0.68,
  'JGL': 0.28,
  'TOP': 0.24,
  'BOT': 0.05
};

const ROLE_AD_PRIORS = {
  'MID': 0.32,
  'JGL': 0.72,
  'TOP': 0.76,
  'BOT': 0.95
};

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
  const weights = MATRIX.weights || { base: 1.0, lane: 1.35, synergy: 0.85, duo_synergy: 1.35, threat: 0.60, blind: 0.90 };
  const duoWeight = weights.duo_synergy || 1.35;
  const teamSynWeight = weights.synergy || 0.85;

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

  // Distinguish 2v2 duo lane partner from off-lane team allies
  let duoSynergyDelta = 0.0;
  let totalTeamSynergyDelta = 0.0;
  let duoPartnerName = null;

  const targetDuoRoleCode = (targetRole === 'bottom') ? 'SUP' : ((targetRole === 'support' || targetRole === 'utility') ? 'BOT' : null);
  const duoAlly = targetDuoRoleCode ? state.allies.find(a => a.role === targetDuoRoleCode && a.cid) : null;
  const duoAllyCid = duoAlly ? String(duoAlly.cid) : null;

  for (const aCid of lockedAllies) {
    const syn = synergies[aCid] || 0.0;
    if (duoAllyCid && String(aCid) === duoAllyCid) {
      duoSynergyDelta += syn;
      duoPartnerName = duoAlly.name || MATRIX.champions[duoAllyCid]?.name || 'Duo';
    } else {
      totalTeamSynergyDelta += syn;
    }
  }
  const totalSynergyDelta = duoSynergyDelta + totalTeamSynergyDelta;

  // Compositional Guardrails: Enemy-Aware Damage Vulnerability
  let compAdjustment = 0.0;
  const rationale = [];
  let projectedPhys = candDmg.pct_physical || 50.0;
  let projectedMagic = candDmg.pct_magic || 50.0;

  if (lockedAllies.length > 0) {
    const alliesDmg = lockedAllies.map(aCid => getChampionDamageProfile(aCid, null));
    const allDmgList = [...alliesDmg, candDmg];
    projectedPhys = allDmgList.reduce((s, d) => s + (d.pct_physical || 0), 0) / allDmgList.length;
    projectedMagic = allDmgList.reduce((s, d) => s + (d.pct_magic || 0), 0) / allDmgList.length;

    const lockedCoreAllies = state.allies.filter(a => a.cid && a.role !== 'SUP');
    const openCoreRoles = state.allies.filter(a => !a.cid && a.role !== 'SUP').map(a => a.role);

    const hasLockedApCarry = lockedCoreAllies.some(a => {
      const p = getChampionDamageProfile(a.cid, null);
      return (p.pct_magic || 0) >= 50.0;
    });

    const hasLockedAdCarry = lockedCoreAllies.some(a => {
      const p = getChampionDamageProfile(a.cid, null);
      return (p.pct_physical || 0) >= 50.0;
    });

    let enemyArmorFactor = 0.6;
    const armorStackerNames = [];
    for (const eCid of lockedEnemies) {
      if (ARMOR_STACKING_CHAMPIONS.has(String(eCid))) {
        enemyArmorFactor += 0.45;
        const eName = MATRIX.champions[eCid]?.name;
        if (eName) armorStackerNames.push(eName);
      }
    }
    enemyArmorFactor = Math.min(enemyArmorFactor, 1.8);

    let enemyMrFactor = 0.6;
    const mrStackerNames = [];
    for (const eCid of lockedEnemies) {
      if (MR_STACKING_CHAMPIONS.has(String(eCid))) {
        enemyMrFactor += 0.45;
        const eName = MATRIX.champions[eCid]?.name;
        if (eName) mrStackerNames.push(eName);
      }
    }
    enemyMrFactor = Math.min(enemyMrFactor, 1.8);

    const candIsAd = (candDmg.pct_physical || 0) >= 65.0;
    const candIsAp = (candDmg.pct_magic || 0) >= 55.0;

    // Physical Skew Evaluation
    if (!hasLockedApCarry && !candIsAp) {
      const targetRoleKey = targetRole === 'middle' ? 'MID' : (targetRole === 'jungle' ? 'JGL' : (targetRole === 'bottom' ? 'BOT' : (targetRole === 'top' ? 'TOP' : '')));
      const remainingOpenRoles = openCoreRoles.filter(r => r !== targetRoleKey);

      let pZeroApRisk = 1.0;
      if (remainingOpenRoles.length > 0) {
        for (const r of remainingOpenRoles) {
          pZeroApRisk *= (1.0 - (ROLE_AP_PRIORS[r] || 0.25));
        }
      } else {
        pZeroApRisk = 1.0;
      }

      if (pZeroApRisk >= 0.20 && lockedCoreAllies.length >= 1) {
        const penalty = Math.min(4.5, pZeroApRisk * enemyArmorFactor * 3.6);
        compAdjustment -= penalty;
        if (pZeroApRisk >= 0.85) {
          const stackerInfo = armorStackerNames.length > 0 ? ` into ${armorStackerNames.join(', ')}` : '';
          rationale.push(`Draft Trap: Seals Full AD (-${penalty.toFixed(2)}%)${stackerInfo}. Enemy can build pure Armor`);
        } else {
          rationale.push(`Damage Warning: Heavy AD compounding (-${penalty.toFixed(2)}%). Missing primary AP anchor`);
        }
      }
    } else if (!hasLockedApCarry && candIsAp && lockedCoreAllies.length >= 2) {
      const bonus = Math.min(3.5, enemyArmorFactor * 2.5);
      compAdjustment += bonus;
      rationale.push(`Composition Anchor: Crucial AP carry (+${bonus.toFixed(2)}%). Prevents enemy Armor stacking`);
    }

    // Magic Skew Evaluation
    if (!hasLockedAdCarry && !candIsAd) {
      const targetRoleKey = targetRole === 'middle' ? 'MID' : (targetRole === 'jungle' ? 'JGL' : (targetRole === 'bottom' ? 'BOT' : (targetRole === 'top' ? 'TOP' : '')));
      const remainingOpenRoles = openCoreRoles.filter(r => r !== targetRoleKey);

      let pZeroAdRisk = 1.0;
      if (remainingOpenRoles.length > 0) {
        for (const r of remainingOpenRoles) {
          pZeroAdRisk *= (1.0 - (ROLE_AD_PRIORS[r] || 0.70));
        }
      } else {
        pZeroAdRisk = 1.0;
      }

      if (pZeroAdRisk >= 0.20 && lockedCoreAllies.length >= 1) {
        const penalty = Math.min(4.5, pZeroAdRisk * enemyMrFactor * 3.6);
        compAdjustment -= penalty;
        if (pZeroAdRisk >= 0.85) {
          const stackerInfo = mrStackerNames.length > 0 ? ` into ${mrStackerNames.join(', ')}` : '';
          rationale.push(`Draft Trap: Seals Full AP (-${penalty.toFixed(2)}%)${stackerInfo}. Enemy can build pure MR`);
        } else {
          rationale.push(`Damage Warning: Heavy AP compounding (-${penalty.toFixed(2)}%)`);
        }
      }
    } else if (!hasLockedAdCarry && candIsAd && lockedCoreAllies.length >= 2) {
      const bonus = Math.min(3.5, enemyMrFactor * 2.5);
      compAdjustment += bonus;
      rationale.push(`Composition Anchor: Crucial AD carry (+${bonus.toFixed(2)}%). Prevents enemy Magic Resist stacking`);
    }
  }

  const isBlind = totalLaneProb < 0.25;
  let blindPenalty = 0.0;
  if (isBlind) {
    const unrevealedFactor = 1.0 - totalLaneProb;
    blindPenalty = weights.blind * blindVuln * unrevealedFactor;
    if (blindVuln < 2.0) {
      rationale.push(`Safe Blind: Low vulnerability rating (${blindVuln.toFixed(1)}% avg counter severity)`);
    } else {
      rationale.push(`Risky Blind: Punished hard by counters (-${blindVuln.toFixed(1)}% avg)`);
    }
  } else {
    if (expectedLaneDelta > 1.0) {
      rationale.push(`Favorable Lane Matchup (+${expectedLaneDelta.toFixed(2)}% expected delta)`);
    } else if (expectedLaneDelta < -1.0) {
      rationale.push(`Unfavorable Lane Matchup (${expectedLaneDelta.toFixed(2)}% expected delta)`);
    }
  }

  if (duoPartnerName && Math.abs(duoSynergyDelta) >= 0.5) {
    const sign = duoSynergyDelta > 0 ? '+' : '';
    if (duoSynergyDelta > 0) {
      rationale.push(`Bot Duo Synergy (${sign}${duoSynergyDelta.toFixed(2)}% with ${duoPartnerName})`);
    } else {
      rationale.push(`Bot Duo Friction (${sign}${duoSynergyDelta.toFixed(2)}% with ${duoPartnerName})`);
    }
  }

  if (totalTeamSynergyDelta > 0.8) {
    rationale.push(`Strong Team Synergy (+${totalTeamSynergyDelta.toFixed(2)}%)`);
  } else if (totalTeamSynergyDelta < -0.8) {
    rationale.push(`Negative Team Synergy (${totalTeamSynergyDelta.toFixed(2)}%)`);
  }

  const compositeScore = (
    weights.base * baselineWr
    + weights.lane * expectedLaneDelta
    + weights.threat * expectedThreatDelta
    + (duoWeight * duoSynergyDelta)
    + (teamSynWeight * totalTeamSynergyDelta)
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
  }
}

// Setup Event Listeners
function setupEventListeners() {
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

  // Modern Segmented Tab Control (Best / Worst Picks)
  const segControl = document.getElementById('hudSegmentedControl');
  if (segControl) {
    segControl.addEventListener('click', (e) => {
      const btn = e.target.closest('.seg-btn');
      if (!btn) return;
      const view = btn.dataset.tab;
      if (view) setHudView(view);
    });
  }
}

function setHudView(view) {
  state.hudView = view;

  // 1. Update tab button active and ARIA states
  document.querySelectorAll('.seg-btn').forEach(btn => {
    const isSelected = btn.dataset.tab === view;
    btn.classList.toggle('active', isSelected);
    btn.setAttribute('aria-selected', isSelected ? 'true' : 'false');
  });

  // 2. Update panel theme attribute
  const hudPanel = document.getElementById('hudPanel');
  if (hudPanel) {
    hudPanel.setAttribute('data-view', view);
  }

  // 3. Update title prefix text
  const hudTitlePrefix = document.getElementById('hudTitlePrefix');
  if (hudTitlePrefix) {
    hudTitlePrefix.textContent = view === 'worst' ? 'Picks to Avoid' : 'Suggested Picks';
  }

  // 4. Render recommendations
  renderRecommendations(currentScoredCandidates);
}

let draggedCard = null;
let isDraggingAlly = false;

function syncAlliesOrderFromDOM() {
  const cards = Array.from(allySlotsContainer.querySelectorAll('.slot-card.draggable-slot'));
  const activeAllyRef = (state.activeTarget.team === 'ally' && state.allies[state.activeTarget.index])
    ? state.allies[state.activeTarget.index]
    : null;

  const newAllies = [];
  cards.forEach((c, i) => {
    const ally = state.allies.find(a => (a.id || a.role.toLowerCase()) === c.dataset.allyId);
    if (ally) newAllies.push(ally);

    const numEl = c.querySelector('.slot-pick-num');
    if (numEl) numEl.textContent = `#${i + 1}`;
    c.dataset.idx = i;

    const removeBtn = c.querySelector('.slot-remove-btn');
    if (removeBtn) removeBtn.dataset.idx = i;
  });

  if (newAllies.length === state.allies.length) {
    state.allies = newAllies;
  }

  if (activeAllyRef) {
    const newIdx = state.allies.indexOf(activeAllyRef);
    if (newIdx !== -1) {
      state.activeTarget.index = newIdx;
    }
  }
}

function attachAllySlotDragHandlers(card, idx) {
  card.addEventListener('dragstart', (e) => {
    isDraggingAlly = true;
    draggedCard = card;
    card.classList.add('is-dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', card.dataset.allyId || String(idx));
  });

  card.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (!draggedCard || draggedCard === card) return;

    const rect = card.getBoundingClientRect();
    const isAfter = (e.clientY - rect.top) > (rect.height / 2);

    if (isAfter) {
      if (card.nextSibling !== draggedCard) {
        allySlotsContainer.insertBefore(draggedCard, card.nextSibling);
        syncAlliesOrderFromDOM();
      }
    } else {
      if (draggedCard.nextSibling !== card) {
        allySlotsContainer.insertBefore(draggedCard, card);
        syncAlliesOrderFromDOM();
      }
    }
  });

  card.addEventListener('dragend', () => {
    if (draggedCard) {
      draggedCard.classList.remove('is-dragging');
    }
    syncAlliesOrderFromDOM();
    updateTargetBanner();
    computeRecommendationsClientSide();

    setTimeout(() => {
      isDraggingAlly = false;
      draggedCard = null;
    }, 60);
  });

  // Touch support for touchscreen devices with real-time DOM swap
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
      draggedCard = card;
      card.classList.add('is-dragging');
    }
    if (card._touchStarted && draggedCard) {
      e.preventDefault();
      const targetEl = document.elementFromPoint(touch.clientX, touch.clientY);
      const targetCard = targetEl ? targetEl.closest('.slot-card.draggable-slot') : null;
      if (targetCard && targetCard !== draggedCard && targetCard.parentElement === allySlotsContainer) {
        const rect = targetCard.getBoundingClientRect();
        const isAfter = (touch.clientY - rect.top) > (rect.height / 2);
        if (isAfter) {
          if (targetCard.nextSibling !== draggedCard) {
            allySlotsContainer.insertBefore(draggedCard, targetCard.nextSibling);
            syncAlliesOrderFromDOM();
          }
        } else {
          if (draggedCard.nextSibling !== targetCard) {
            allySlotsContainer.insertBefore(draggedCard, targetCard);
            syncAlliesOrderFromDOM();
          }
        }
      }
    }
  }, { passive: false });

  card.addEventListener('touchend', () => {
    if (card._touchStarted) {
      if (draggedCard) {
        draggedCard.classList.remove('is-dragging');
      }
      syncAlliesOrderFromDOM();
      updateTargetBanner();
      computeRecommendationsClientSide();
      renderDraftSlots();
      setTimeout(() => {
        isDraggingAlly = false;
        draggedCard = null;
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
    card.dataset.allyId = slot.id || slot.role.toLowerCase();
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
        ${slot.icon ? `<img src="${slot.icon}" alt="${slot.name}">` : `<span class="slot-role-abbr">${slot.role}</span>`}
      </div>
      <div class="slot-meta">
        <div class="slot-role-row">
          <span class="slot-pick-num">#${idx + 1}</span>
          <span class="slot-role-tag">${slot.role}</span>
          ${isTarget ? `<span class="slot-active-badge">ACTIVE</span>` : ''}
        </div>
        <span class="slot-champ-name">${slot.name || 'Empty Slot'}</span>
      </div>
      ${slot.cid ? `<button class="slot-remove-btn" title="Remove" data-team="ally" data-idx="${idx}">✕</button>` : ''}
    `;

    card.addEventListener('click', (e) => {
      if (isDraggingAlly) return;
      if (e.target.closest('.slot-remove-btn')) return;
      state.activeTarget = { team: 'ally', index: idx };

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
        ${slot.icon ? `<img src="${slot.icon}" alt="${slot.name}">` : `<span class="slot-role-abbr slot-enemy-abbr">#${idx + 1}</span>`}
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
  if (!bannerTargetName) return;
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
  if (filtered.length === 0) {
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-search-state';
    emptyState.innerHTML = `<span class="empty-search-icon">🔍</span><span class="empty-search-text"></span>`;
    emptyState.querySelector('.empty-search-text').textContent = `No champions found matching "${state.searchFilter}"`;
    championsGrid.appendChild(emptyState);
    return;
  }

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
  const currentRole = getActiveAllyRole();
  if (hudRoleLabel) {
    hudRoleLabel.textContent = ROLE_NAMES[currentRole] || currentRole.toUpperCase();
  }

  const candidates = [];
  const pickedSet = new Set([...lockedAllies, ...lockedEnemies]);

  for (const cid of Object.keys(MATRIX.champions)) {
    if (pickedSet.has(cid)) continue;
    const res = scoreCandidateJS(cid, currentRole, lockedAllies, lockedEnemies, inferredRoles);
    if (res && res.viable) {
      candidates.push(res);
    }
  }

  candidates.sort((a, b) => b.composite_score - a.composite_score);

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
      damageSummaryText.textContent = `⚠️ Heavy AD (${p}%)`;
      damageSummaryText.title = `Heavy AD skew (${p}%) - Vulnerable to Armor stacking`;
      damageSummaryText.style.color = '#e74c3c';
    } else if (m >= 75) {
      damageSummaryText.textContent = `⚠️ Heavy AP (${m}%)`;
      damageSummaryText.title = `Heavy AP skew (${m}%) - Vulnerable to MR stacking`;
      damageSummaryText.style.color = '#e74c3c';
    } else {
      damageSummaryText.title = '';
      damageSummaryText.style.color = 'var(--text-muted)';
    }
  } else {
    dmgPhysBar.style.width = '50%';
    dmgMagicBar.style.width = '50%';
    physPctLabel.textContent = '50%';
    magicPctLabel.textContent = '50%';
    damageSummaryText.textContent = '50% AD / 50% AP';
    damageSummaryText.title = '';
    damageSummaryText.style.color = 'var(--text-muted)';
  }

  // 4. Render HUD cards
  currentScoredCandidates = candidates;
  renderRecommendations(currentScoredCandidates);
}

// Render Recommendation HUD Cards
function renderRecommendations(allCandidates) {
  recommendationsList.innerHTML = '';
  if (!allCandidates || allCandidates.length === 0) {
    const currentRole = getActiveAllyRole();
    recommendationsList.innerHTML = `<div style="padding: 14px; color: var(--text-dim); font-size: 13px;">No viable champions found for ${ROLE_NAMES[currentRole] || currentRole.toUpperCase()}.</div>`;
    return;
  }

  const isWorst = (state.hudView || 'best') === 'worst';
  const recs = isWorst 
    ? allCandidates.slice(-10).reverse() 
    : allCandidates.slice(0, 10);

  recs.forEach((item, index) => {
    const card = document.createElement('div');
    const isTopOne = index === 0;
    const cardClass = isWorst 
      ? `rec-card worst-pick ${isTopOne ? 'worst-rank-1' : ''}`
      : `rec-card ${isTopOne ? 'rank-1' : ''}`;
    card.className = cardClass;

    const rankLabel = isWorst ? `#${index + 1} AVOID` : `#${index + 1}`;
    const rankClass = isWorst ? `rec-rank-tag worst-rank` : `rec-rank-tag`;
    const scoreClass = isWorst ? `rec-score-value worst-score` : `rec-score-value`;

    const laneDeltaClass = item.expected_lane_delta >= 0 ? 'delta-pos' : 'delta-neg';
    const laneDeltaSign = item.expected_lane_delta >= 0 ? '+' : '';
    const synDeltaClass = item.synergy_delta >= 0 ? 'delta-pos' : 'delta-neg';
    const synDeltaSign = item.synergy_delta >= 0 ? '+' : '';

    const badgesHtml = (item.rationale || []).map(r => {
      const isWarn = r.toLowerCase().includes('risky') || r.toLowerCase().includes('penalty') || r.toLowerCase().includes('negative') || r.toLowerCase().includes('trap') || r.toLowerCase().includes('warning') || r.toLowerCase().includes('unfavorable');
      return `<span class="rec-badge ${isWarn ? 'badge-warn' : 'badge-good'}">${r}</span>`;
    }).join('');

    card.innerHTML = `
      <div class="rec-card-top">
        <span class="${rankClass}">${rankLabel}</span>
        <div class="rec-avatar">
          <img src="${item.icon}" alt="${item.name}" loading="lazy">
        </div>
        <div class="rec-champ-info">
          <span class="rec-champ-name">${item.name}</span>
          <span class="${scoreClass}">${item.composite_score}%</span>
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

// Start application
window.addEventListener('DOMContentLoaded', init);
