/**
 * frontend/app.js
 * ────────────────
 * Clean, modern frontend controller for Zomato Dining Discovery.
 * Handles user interactions, instant filtering, clean card rendering, and reservations.
 */

// Base API URL: Uses relative path '/api' by default (works with Vercel rewrites & local server).
// Can be overridden via window.API_BASE_URL if needed.
const API_BASE = window.API_BASE_URL || "";

// Application State
const state = {
  location: "Indiranagar",
  cuisines: ["Italian"],
  budget: "medium",
  minRating: 4.0,
  notes: "Looking for a lively yet intimate dinner spot with craft sourdough pizza or handmade pasta, pet-friendly outdoor courtyard, artisanal tiramisu, and great cocktails.",
  availableLocations: [],
  presets: [],
  isLoading: false,
};

// Available Popular Cuisines for Quick Toggle
const POPULAR_CUISINES = [
  "Italian",
  "North Indian",
  "South Indian",
  "Continental",
  "Pan-Asian",
  "Cafe",
  "Chinese",
  "Bakery",
  "Biryani",
  "Desserts",
];

// Document Ready Initialization
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  loadLocations();
  loadPresets();
  renderCuisineChips();

  // Perform initial discovery fetch
  fetchRecommendations();
});

function initEventListeners() {
  // Budget Buttons (Low, Medium, High)
  document.querySelectorAll("[data-budget]").forEach((btn) => {
    btn.addEventListener("click", () => {
      setBudget(btn.getAttribute("data-budget"));
    });
  });

  // Rating Range Slider
  const ratingSlider = document.getElementById("rating-slider");
  if (ratingSlider) {
    ratingSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      state.minRating = val;
      updateRatingDisplay(val);
    });
  }

  // Preferences / Notes
  const promptTextarea = document.getElementById("prompt-textarea");
  if (promptTextarea) {
    promptTextarea.addEventListener("input", (e) => {
      state.notes = e.target.value;
    });
  }

  // Single Primary Find Restaurants Button (At bottom of search form)
  const generateBtn = document.getElementById("generate-btn");
  if (generateBtn) {
    generateBtn.addEventListener("click", () => {
      fetchRecommendations();
    });
  }

  // Reset Filters Button
  const resetBtn = document.getElementById("reset-filters-btn");
  if (resetBtn) {
    resetBtn.addEventListener("click", resetFilters);
  }

  // Location Dropdown
  const locationSelect = document.getElementById("location-select");
  if (locationSelect) {
    locationSelect.addEventListener("change", (e) => {
      setLocation(e.target.value);
    });
  }

  // Keyboard shortcut: Ctrl+K or Cmd+K
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      if (promptTextarea) {
        promptTextarea.focus();
        showToast("Search prompt focused (⌘K)", "search");
      }
    }
  });

  // Modal Closers
  document.querySelectorAll("[data-close-modal]").forEach((el) => {
    el.addEventListener("click", () => {
      closeAllModals();
    });
  });

  // Reservation Form Submission
  const bookingForm = document.getElementById("booking-form");
  if (bookingForm) {
    bookingForm.addEventListener("submit", handleBookingSubmit);
  }
}

// ── API Fetchers ─────────────────────────────────────────────────────────────

async function loadLocations() {
  try {
    const res = await fetch(`${API_BASE}/api/locations`);
    const data = await res.json();
    if (data.locations) {
      state.availableLocations = data.locations;
      populateLocationDropdown(data.locations);
      populateAreaPills(data.locations.slice(0, 6));
    }
  } catch (err) {
    console.error("Failed to load locations:", err);
  }
}

async function loadPresets() {
  try {
    const res = await fetch(`${API_BASE}/api/presets`);
    const data = await res.json();
    if (data.presets) {
      state.presets = data.presets;
      renderPresets(data.presets);
    }
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function populateLocationDropdown(locations) {
  const select = document.getElementById("location-select");
  if (!select) return;

  select.innerHTML = "";
  locations.forEach((loc) => {
    const opt = document.createElement("option");
    opt.value = loc;
    opt.textContent = loc;
    if (loc.toLowerCase() === state.location.toLowerCase()) {
      opt.selected = true;
    }
    select.appendChild(opt);
  });
}

function populateAreaPills(topLocations) {
  const container = document.getElementById("quick-area-pills");
  if (!container) return;

  container.innerHTML = "";
  topLocations.forEach((loc) => {
    const isSelected = loc.toLowerCase() === state.location.toLowerCase();
    const pill = document.createElement("span");
    pill.className = `px-2.5 py-1 rounded-lg text-xs font-medium cursor-pointer transition-colors ${
      isSelected
        ? "bg-primary text-white font-semibold shadow-sm shadow-primary/20"
        : "bg-surface-card text-text-muted hover:text-white hover:bg-surface-highlight border border-border-subtle"
    }`;
    pill.textContent = loc;
    pill.addEventListener("click", () => {
      setLocation(loc);
    });
    container.appendChild(pill);
  });
}

function renderPresets(presets) {
  const container = document.getElementById("presets-carousel");
  if (!container) return;

  container.innerHTML = "";
  presets.forEach((preset, index) => {
    const btn = document.createElement("button");
    const isActive = index === 0;
    btn.className = `shrink-0 flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all border ${
      isActive
        ? "bg-primary text-white border-primary shadow-md shadow-primary/25"
        : "bg-surface-low hover:bg-surface-card text-text-muted hover:text-white border-border-subtle"
    }`;
    btn.innerHTML = `<span>${preset.emoji}</span> <span>${preset.label.replace(preset.emoji, "").trim()}</span>`;
    
    btn.addEventListener("click", () => {
      // Highlight active preset
      container.querySelectorAll("button").forEach((b) => {
        b.className =
          "shrink-0 flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-surface-low hover:bg-surface-card text-text-muted hover:text-white border border-border-subtle transition-all";
      });
      btn.className =
        "shrink-0 flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-primary text-white border-primary shadow-md shadow-primary/25 transition-all";

      applyPreset(preset);
    });

    container.appendChild(btn);
  });
}

function applyPreset(preset) {
  state.location = preset.location;
  state.cuisines = preset.cuisine ? [preset.cuisine] : [];
  state.budget = preset.budget;
  state.minRating = preset.rating;
  state.notes = preset.notes;

  // Sync Form Controls
  const locSelect = document.getElementById("location-select");
  if (locSelect) locSelect.value = preset.location;

  setBudget(preset.budget);

  const slider = document.getElementById("rating-slider");
  if (slider) {
    slider.value = preset.rating;
    updateRatingDisplay(preset.rating);
  }

  const promptTextarea = document.getElementById("prompt-textarea");
  if (promptTextarea) {
    promptTextarea.value = preset.notes;
  }

  renderCuisineChips();
  populateAreaPills(state.availableLocations.slice(0, 6));
  updateLocationHeader();

  showToast(`Applied preset: ${preset.label}`, "check");
  fetchRecommendations();
}

// ── UI Mutators ──────────────────────────────────────────────────────────────

function setLocation(loc) {
  state.location = loc;
  const select = document.getElementById("location-select");
  if (select) select.value = loc;

  populateAreaPills(state.availableLocations.slice(0, 6));
  updateLocationHeader();
  showToast(`Location set to ${loc}`, "location_on");
}

function updateLocationHeader() {
  const headerLabel = document.getElementById("header-location-label");
  if (headerLabel) {
    headerLabel.textContent = state.location;
  }
  const noticeTitle = document.getElementById("traffic-notice-title");
  if (noticeTitle) {
    noticeTitle.textContent = `${state.location} Dining Note`;
  }
}

function setBudget(tier) {
  state.budget = tier;
  document.querySelectorAll("[data-budget]").forEach((b) => {
    if (b.getAttribute("data-budget") === tier) {
      b.className =
        "py-2 text-center rounded-lg bg-primary text-white font-semibold text-xs shadow-md shadow-primary/20 transition-all";
    } else {
      b.className =
        "py-2 text-center rounded-lg text-xs font-medium text-text-muted hover:text-white transition-all";
    }
  });

  const priceAvg = document.getElementById("price-tier-avg");
  if (priceAvg) {
    if (tier === "low") priceAvg.textContent = "≤ ₹300 for two";
    else if (tier === "medium") priceAvg.textContent = "₹301 – ₹700 avg";
    else priceAvg.textContent = "> ₹700 premium";
  }
}

function updateRatingDisplay(rating) {
  const badge = document.getElementById("rating-badge");
  if (badge) {
    badge.textContent = `⭐ ${rating.toFixed(1)}+`;
  }
}

function renderCuisineChips() {
  const container = document.getElementById("cuisine-chips-container");
  if (!container) return;

  container.innerHTML = "";

  // Selected cuisines
  state.cuisines.forEach((c) => {
    const chip = document.createElement("span");
    chip.className =
      "px-3 py-1 rounded-full text-xs font-semibold bg-primary text-white flex items-center gap-1.5 cursor-pointer shadow-sm shadow-primary/20 transition-all hover:opacity-90";
    chip.innerHTML = `${c} <span class="material-symbols-outlined text-[13px]">close</span>`;
    chip.addEventListener("click", () => {
      removeCuisine(c);
    });
    container.appendChild(chip);
  });

  // Top unselected cuisines
  POPULAR_CUISINES.filter((c) => !state.cuisines.includes(c)).slice(0, 5).forEach((c) => {
    const chip = document.createElement("span");
    chip.className =
      "px-3 py-1 rounded-full text-xs font-medium bg-surface-card hover:bg-surface-highlight text-text-muted hover:text-white border border-border-subtle cursor-pointer transition-colors";
    chip.textContent = `+ ${c}`;
    chip.addEventListener("click", () => {
      addCuisine(c);
    });
    container.appendChild(chip);
  });
}

function addCuisine(c) {
  if (!state.cuisines.includes(c)) {
    state.cuisines.push(c);
    renderCuisineChips();
  }
}

function removeCuisine(c) {
  state.cuisines = state.cuisines.filter((x) => x !== c);
  renderCuisineChips();
}

function resetFilters() {
  state.location = "Indiranagar";
  state.cuisines = ["Italian"];
  state.budget = "medium";
  state.minRating = 4.0;
  state.notes = "";

  const locSelect = document.getElementById("location-select");
  if (locSelect) locSelect.value = "Indiranagar";

  setBudget("medium");

  const slider = document.getElementById("rating-slider");
  if (slider) {
    slider.value = 4.0;
    updateRatingDisplay(4.0);
  }

  const promptTextarea = document.getElementById("prompt-textarea");
  if (promptTextarea) {
    promptTextarea.value = "";
  }

  renderCuisineChips();
  populateAreaPills(state.availableLocations.slice(0, 6));
  updateLocationHeader();
  showToast("Filters reset", "restart_alt");
}

// ── Recommendation Execution ─────────────────────────────────────────────────

async function fetchRecommendations() {
  if (state.isLoading) return;
  state.isLoading = true;

  const generateBtn = document.getElementById("generate-btn");
  if (generateBtn) {
    generateBtn.disabled = true;
    generateBtn.innerHTML = `
      <span class="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
      <span>Finding the best spots...</span>
    `;
  }

  // Render clean loading skeletons
  renderLoadingSkeletons();

  const payload = {
    location: state.location,
    cuisine: state.cuisines.join(", "),
    budget: state.budget,
    min_rating: state.minRating,
    notes: state.notes,
    top_n: 15,
  };

  try {
    const res = await fetch(`${API_BASE}/api/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Failed to retrieve recommendations");
    }

    renderRecommendations(data);
  } catch (err) {
    console.error("Error fetching recommendations:", err);
    renderErrorState(err.message);
  } finally {
    state.isLoading = false;
    if (generateBtn) {
      generateBtn.disabled = false;
      generateBtn.innerHTML = `
        <span class="material-symbols-outlined text-[18px]">restaurant</span>
        <span>Find Restaurants</span>
      `;
    }
  }
}

// ── Clean Rendering Functions ────────────────────────────────────────────────

function renderLoadingSkeletons() {
  const container = document.getElementById("recommendations-feed");
  if (!container) return;

  container.innerHTML = `
    <!-- Shimmering Summary Card -->
    <div class="p-6 rounded-2xl bg-surface-low border border-border-subtle shadow-lg space-y-3">
      <div class="h-6 w-56 skeleton"></div>
      <div class="h-4 w-full skeleton"></div>
      <div class="h-4 w-4/5 skeleton"></div>
    </div>

    <!-- Shimmering Restaurant Cards -->
    ${[1, 2, 3]
      .map(
        () => `
      <div class="rounded-2xl bg-surface-low border border-border-subtle overflow-hidden shadow-lg">
        <div class="h-56 w-full skeleton"></div>
        <div class="p-6 space-y-4">
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div class="h-10 skeleton"></div>
            <div class="h-10 skeleton"></div>
            <div class="h-10 skeleton"></div>
            <div class="h-10 skeleton"></div>
          </div>
          <div class="h-20 skeleton rounded-xl"></div>
          <div class="flex justify-between items-center pt-2">
            <div class="h-10 w-36 rounded-xl skeleton"></div>
            <div class="h-10 w-28 rounded-xl skeleton"></div>
          </div>
        </div>
      </div>
    `
      )
      .join("")}
  `;
}

function renderRecommendations(data) {
  const container = document.getElementById("recommendations-feed");
  if (!container) return;

  if (!data.recommendations || data.recommendations.length === 0) {
    container.innerHTML = `
      <div class="p-12 rounded-2xl bg-surface-low border border-border-subtle text-center flex flex-col items-center gap-3">
        <span class="material-symbols-outlined text-[44px] text-text-subtle">restaurant_menu</span>
        <h3 class="font-display font-bold text-lg text-white">No matching restaurants found</h3>
        <p class="text-sm text-text-muted max-w-md">${escapeHtml(data.summary || "Try lowering the minimum rating or clearing some cuisine filters to see more recommendations.")}</p>
        <button onclick="resetFilters()" class="mt-2 px-4 py-2 rounded-xl bg-primary text-white text-xs font-semibold shadow-md">Reset Filters</button>
      </div>
    `;
    return;
  }

  const queryTitle = `${data.query.location} ${data.query.cuisine !== "All Cuisines" ? data.query.cuisine : ""} Dining`.trim();

  // Clean Overview Banner
  let html = `
    <div class="p-6 rounded-2xl bg-surface-low border border-border-subtle shadow-lg flex flex-col gap-3">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <h2 class="font-display font-bold text-lg text-white flex items-center gap-2">
          <span>Curated Picks for "${escapeHtml(queryTitle)}"</span>
        </h2>
        <span class="text-xs font-medium text-emerald bg-emerald/10 border border-emerald/20 px-2.5 py-1 rounded-full">
          ${data.recommendations.length} Recommended Places
        </span>
      </div>
      <p class="text-sm text-text-muted leading-relaxed">
        ${escapeHtml(data.summary)}
      </p>
      <div class="flex flex-wrap gap-2 pt-1">
        <span class="px-2.5 py-1 rounded-lg bg-surface-card border border-border-subtle text-text-muted text-xs font-medium flex items-center gap-1.5">
          🌿 Courtyard Dining
        </span>
        <span class="px-2.5 py-1 rounded-lg bg-surface-card border border-border-subtle text-text-muted text-xs font-medium flex items-center gap-1.5">
          🍕 Wood-Fired Preparation
        </span>
        <span class="px-2.5 py-1 rounded-lg bg-surface-card border border-border-subtle text-text-muted text-xs font-medium flex items-center gap-1.5">
          🐶 Pet Friendly
        </span>
        <span class="px-2.5 py-1 rounded-lg bg-surface-card border border-border-subtle text-text-muted text-xs font-medium flex items-center gap-1.5">
          🍸 Craft Cocktails
        </span>
      </div>
    </div>
  `;

  // Clean Restaurant Cards
  data.recommendations.forEach((rec, idx) => {
    const isGold = idx === 0;
    const isSilver = idx === 1;
    const isBronze = idx === 2;

    const rankBadgeClass = isGold
      ? "bg-amber text-black font-bold"
      : isSilver
      ? "bg-slate-200 text-black font-bold"
      : isBronze
      ? "bg-[#d97706] text-white font-bold"
      : "bg-surface-card text-text-main font-semibold border border-border-subtle";

    const rankLabel = isGold ? "🏆 #1 Top Pick" : isSilver ? "🥈 #2 Top Pick" : isBronze ? "🥉 #3 Top Pick" : `#${rec.rank} Recommended`;

    html += `
      <article class="flex flex-col rounded-2xl bg-surface-low border border-border-subtle overflow-hidden shadow-lg hover:border-text-subtle/50 transition-all duration-200 group">
        <!-- Media Banner -->
        <div class="relative h-60 sm:h-64 w-full overflow-hidden">
          <img class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
               src="${rec.image_url}"
               alt="${escapeHtml(rec.name)}"
               onerror="this.src='https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=1200&q=80'" />
          <div class="absolute inset-0 bg-gradient-to-t from-surface-low via-surface-low/30 to-transparent"></div>
          
          <!-- Top Floating Badges -->
          <div class="absolute top-4 left-4 flex items-center gap-2">
            <span class="px-3 py-1 rounded-full text-xs shadow-md ${rankBadgeClass}">
              ${rankLabel}
            </span>
            ${
              rec.traits && rec.traits[0]
                ? `<span class="px-2.5 py-1 rounded-full bg-surface/80 backdrop-blur-md text-emerald text-xs font-semibold border border-border-subtle">${rec.traits[0]}</span>`
                : ""
            }
          </div>

          <div class="absolute top-4 right-4">
            <div class="px-2.5 py-1 rounded-full bg-emerald text-white text-xs font-bold shadow-md flex items-center gap-1">
              <span>⭐ ${rec.rating.toFixed(1)}</span>
              <span class="text-[10px] font-normal opacity-90">${rec.votes_label || "(Verified)"}</span>
            </div>
          </div>

          <!-- Bottom Title Overlay -->
          <div class="absolute bottom-4 left-4 right-4">
            <div class="flex items-center gap-1.5">
              <h3 class="font-display font-extrabold text-xl text-white drop-shadow-sm">
                ${escapeHtml(rec.name)}
              </h3>
              <span class="material-symbols-outlined text-emerald text-[20px]" title="Verified Kitchen">verified</span>
            </div>
            <p class="text-xs text-text-muted mt-0.5">
              ${escapeHtml(rec.cuisine)} • ${escapeHtml(rec.rest_type || "Casual Dining")}
            </p>
          </div>
        </div>

        <!-- Card Content Body -->
        <div class="p-5 sm:p-6 flex flex-col gap-4">
          <!-- 4 Quick Facts Pills -->
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5 p-3 rounded-xl bg-surface-card border border-border-subtle text-xs">
            <div>
              <span class="text-text-subtle text-[11px] block">Cost for Two</span>
              <span class="font-semibold text-text-main">${rec.cost_label}</span>
            </div>
            <div>
              <span class="text-text-subtle text-[11px] block">Location</span>
              <span class="font-semibold text-text-main truncate block">${rec.distance_label}</span>
            </div>
            <div>
              <span class="text-text-subtle text-[11px] block">Hours</span>
              <span class="font-semibold text-emerald">${rec.timing}</span>
            </div>
            <div>
              <span class="text-text-subtle text-[11px] block">Ambience</span>
              <span class="font-semibold text-cyan truncate block">${rec.acoustics}</span>
            </div>
          </div>

          <!-- Clean Recommendation Box -->
          <div class="rounded-xl p-4 bg-surface-card/60 border border-border-subtle flex flex-col gap-2">
            <div class="flex items-center gap-1.5 text-xs font-semibold text-cyan uppercase tracking-wider">
              <span class="material-symbols-outlined text-[15px]">tips_and_updates</span>
              Why We Recommend This
            </div>
            <p class="text-sm text-text-muted leading-relaxed">
              ${escapeHtml(rec.explanation)}
            </p>
            
            <!-- Must Try Dish Highlights -->
            ${
              rec.curated_dishes && rec.curated_dishes.length
                ? `
              <div class="mt-1 pt-2 border-t border-border-subtle flex flex-wrap items-center gap-1.5">
                <span class="text-xs font-medium text-text-subtle">Must Try:</span>
                ${rec.curated_dishes
                  .map(
                    (d) =>
                      `<span class="text-xs px-2.5 py-0.5 rounded-md bg-surface-highlight text-text-main border border-border-subtle">${escapeHtml(
                        d
                      )}</span>`
                  )
                  .join("")}
              </div>
            `
                : ""
            }
          </div>

          <!-- Action Buttons Bar -->
          <div class="flex flex-wrap items-center justify-between gap-3 pt-1">
            <div class="flex items-center gap-2">
              <button onclick="openBookingModal('${escapeJsString(rec.name)}')"
                      class="px-4 py-2 rounded-xl bg-primary hover:bg-primary-hover text-white text-xs font-semibold shadow-md shadow-primary/20 hover:scale-[1.02] active:scale-95 transition-all flex items-center gap-1.5">
                <span class="material-symbols-outlined text-[16px]">table_restaurant</span>
                Reserve Table
              </button>
              <button onclick="openMenuModal('${escapeJsString(rec.name)}', '${escapeJsString(rec.cuisine)}')"
                      class="px-4 py-2 rounded-xl bg-surface-card hover:bg-surface-highlight text-text-main text-xs font-medium border border-border-subtle transition-colors flex items-center gap-1.5">
                <span class="material-symbols-outlined text-[16px]">restaurant_menu</span>
                View Menu
              </button>
            </div>
            
            <div class="flex items-center gap-2">
              <a href="https://www.google.com/maps/search/${encodeURIComponent(rec.name + ' ' + rec.location + ' Bangalore')}"
                 target="_blank" rel="noopener noreferrer"
                 class="p-2 rounded-xl bg-surface-card hover:bg-surface-highlight text-text-muted hover:text-white border border-border-subtle transition-colors"
                 title="View on Google Maps">
                <span class="material-symbols-outlined text-[16px]">directions</span>
              </a>
              <button onclick="shareRestaurant('${escapeJsString(rec.name)}', '${escapeJsString(rec.location)}')"
                      class="p-2 rounded-xl bg-surface-card hover:bg-surface-highlight text-text-muted hover:text-white border border-border-subtle transition-colors"
                      title="Share Restaurant">
                <span class="material-symbols-outlined text-[16px]">share</span>
              </button>
            </div>
          </div>
        </div>
      </article>
    `;
  });

  container.innerHTML = html;
}

function renderErrorState(errorMessage) {
  const container = document.getElementById("recommendations-feed");
  if (!container) return;

  container.innerHTML = `
    <div class="p-8 rounded-2xl bg-surface-low border border-border-subtle text-center flex flex-col items-center gap-3">
      <span class="material-symbols-outlined text-primary text-[36px]">error</span>
      <h3 class="font-display font-bold text-base text-white">Unable to retrieve recommendations</h3>
      <p class="text-xs text-text-muted max-w-md">${escapeHtml(errorMessage)}</p>
      <button onclick="fetchRecommendations()" class="mt-2 px-4 py-2 rounded-xl bg-primary text-white text-xs font-semibold">Try Again</button>
    </div>
  `;
}

// ── Modals & Actions ─────────────────────────────────────────────────────────

function openBookingModal(restaurantName) {
  const modal = document.getElementById("booking-modal");
  const modalTitle = document.getElementById("booking-modal-title");
  const restaurantInput = document.getElementById("booking-restaurant-name");

  if (modalTitle) modalTitle.textContent = `Reserve Table at ${restaurantName}`;
  if (restaurantInput) restaurantInput.value = restaurantName;

  if (modal) {
    modal.classList.remove("hidden");
    modal.classList.add("flex");
  }
}

async function handleBookingSubmit(e) {
  e.preventDefault();
  const restaurant = document.getElementById("booking-restaurant-name").value;
  const guests = document.getElementById("booking-guests").value;
  const time = document.getElementById("booking-time").value;
  const date = document.getElementById("booking-date").value;

  try {
    const res = await fetch(`${API_BASE}/api/book`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ restaurant, guests, time, date }),
    });
    const data = await res.json();

    closeAllModals();
    showToast(`Table reserved at ${restaurant}! Confirmation #${data.confirmation_id}`, "check_circle");
  } catch (err) {
    showToast("Booking failed, please try again", "error");
  }
}

function openMenuModal(restaurantName, cuisine) {
  const modal = document.getElementById("menu-modal");
  const modalTitle = document.getElementById("menu-modal-title");
  const modalSubtitle = document.getElementById("menu-modal-subtitle");

  if (modalTitle) modalTitle.textContent = `${restaurantName} — Menu`;
  if (modalSubtitle) modalSubtitle.textContent = `Signature Items • ${cuisine}`;

  if (modal) {
    modal.classList.remove("hidden");
    modal.classList.add("flex");
  }
}

function closeAllModals() {
  document.querySelectorAll(".modal-root").forEach((m) => {
    m.classList.add("hidden");
    m.classList.remove("flex");
  });
}

function shareRestaurant(name, location) {
  const url = window.location.href;
  const shareText = `Check out ${name} in ${location} on Zomato Dining! ${url}`;
  if (navigator.clipboard) {
    navigator.clipboard.writeText(shareText);
    showToast("Link copied to clipboard!", "content_copy");
  } else {
    showToast(`Shared: ${name}`, "share");
  }
}

// ── Toast Utility ────────────────────────────────────────────────────────────

function showToast(message, icon = "info") {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `
    <span class="material-symbols-outlined text-primary text-[16px]">${icon}</span>
    <span>${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}

// ── Helper Utilities ─────────────────────────────────────────────────────────

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeJsString(str) {
  if (!str) return "";
  return String(str).replace(/'/g, "\\'").replace(/"/g, '\\"');
}
