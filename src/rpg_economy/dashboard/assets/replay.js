/* Replay player. Reads DATA and STR (both injected above); touches nothing else.
   Frames are complete states, so seeking is an array index, never a re-fold. */
(function () {
  "use strict";
  var D = window.DATA, S = window.STR;
  var horizon = D.horizon, tracks = D.tracks;
  var tick = 0, playing = false, speed = 4, lastFrame = 0, acc = 0;
  var good = null;
  var F = {};  /* agent field index by name */
  (tracks[0] ? tracks[0].agentFields : []).forEach(function (n, i) { F[n] = i; });

  /* ---------- small helpers ---------- */
  var $ = function (s, r) { return (r || document).querySelector(s); };
  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt !== undefined && txt !== null) n.textContent = txt;
    return n;
  }
  function add(parent, child) { parent.appendChild(child); return child; }

  /* ---------- prompts and replies: decompressed on first use ---------- */
  function loadTexts(tr) {
    if (tr._texts) return tr._texts;
    if (!tr.promptTexts) { tr._texts = Promise.resolve(null); return tr._texts; }
    if (typeof DecompressionStream === "undefined") { tr._texts = Promise.resolve("unsupported"); return tr._texts; }
    var bin = atob(tr.promptTexts), bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    tr._texts = new Response(stream).text().then(function (s) { return JSON.parse(s); })
      .catch(function () { return "error"; });
    return tr._texts;
  }
  function promptSection(box, title, text, open) {
    var det = add(box, el("details", "psec"));
    if (open) det.open = true;
    var sum = add(det, el("summary", "", title));
    if (text === null || text === undefined) {
      add(det, el("pre", "ptext", tpl("replay.prompt.no_reply")));
      return;
    }
    var copy = add(sum, el("button", "pcopy", tpl("replay.prompt.copy")));
    copy.type = "button";
    copy.onclick = function (e) {
      e.preventDefault(); e.stopPropagation();
      function flash(key) {
        copy.textContent = tpl(key);
        setTimeout(function () { copy.textContent = tpl("replay.prompt.copy"); }, 1500);
      }
      if (!navigator.clipboard) { flash("replay.prompt.copy_failed"); return; }
      navigator.clipboard.writeText(text || "").then(
        function () { flash("replay.prompt.copied"); },
        function () { flash("replay.prompt.copy_failed"); });
    };
    add(det, el("pre", "ptext", text));
  }
  var promptDialog = null;
  function showPrompt(tr, title, ids) {
    if (!promptDialog) {
      promptDialog = document.createElement("dialog");
      promptDialog.className = "pdialog";
      document.body.appendChild(promptDialog);
      // Closes on a click on the backdrop, not at the end of a text selection
      // that started inside and was released outside.
      var downOnBackdrop = false;
      promptDialog.addEventListener("mousedown", function (e) { downOnBackdrop = e.target === promptDialog; });
      promptDialog.addEventListener("click", function (e) {
        if (e.target === promptDialog && downOnBackdrop) promptDialog.close();
        downOnBackdrop = false;
      });
    }
    // The replay stands still while a prompt is read.
    pause();
    var dlg = promptDialog, opener = document.activeElement;
    var openerKey = opener && opener.getAttribute ? opener.getAttribute("data-key") : null;
    dlg.textContent = "";
    var head = add(dlg, el("div", "phead"));
    add(head, el("strong", "", title));
    var close = add(head, el("button", "icon", "×"));
    close.type = "button";
    close.setAttribute("aria-label", tpl("replay.prompt.close"));
    close.onclick = function () { dlg.close(); };
    var body = add(dlg, el("div", "pbody"));
    add(body, el("p", "muted", tpl("replay.prompt.loading")));
    dlg.onclose = function () {
      // A re-render may have replaced the button that opened the dialog.
      var back = opener && opener.isConnected ? opener :
        (openerKey ? document.querySelector('[data-key="' + openerKey.replace(/["\\]/g, "\\$&") + '"]') : null);
      if (back && back.focus) back.focus();
    };
    dlg.showModal();
    loadTexts(tr).then(function (texts) {
      body.textContent = "";
      if (!texts) { add(body, el("p", "muted", tpl("replay.prompt.unavailable"))); return; }
      if (texts === "unsupported") { add(body, el("p", "muted", tpl("replay.prompt.unsupported"))); return; }
      if (texts === "error") { add(body, el("p", "muted", tpl("replay.prompt.error"))); return; }
      function at(i) { return i === null || i === undefined ? null : texts[i]; }
      if (ids[0] !== null && ids[0] !== undefined) promptSection(body, tpl("replay.prompt.system"), at(ids[0]), false);
      promptSection(body, tpl("replay.prompt.user"), at(ids[1]), true);
      if (ids.length > 2) promptSection(body, tpl("replay.prompt.reply"), at(ids[2]), true);
    });
  }
  function promptButton(parent, tr, title, ids, key) {
    if (!ids || ids[1] === null || ids[1] === undefined) return;
    var b = add(parent, el("button", "pbtn", tpl("replay.prompt.open")));
    b.type = "button";
    // Stable across re-renders, so focus can return to it.
    b.setAttribute("data-key", tr.condition + ":" + key);
    b.onclick = function (e) { e.preventDefault(); e.stopPropagation(); showPrompt(tr, title, ids); };
  }
  function num(v, d) {
    if (v === null || v === undefined || v === "") return "—";
    var x = typeof v === "number" ? v : parseFloat(v);
    if (isNaN(x)) return String(v);
    // "always": Spanish otherwise leaves four-digit numbers ungrouped (2100 vs 12.100).
    return x.toLocaleString(S.locale, { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0, useGrouping: "always" });
  }
  function money(v) { return num(v, 2); }
  /* The smallest 1, 2 or 5 times a power of ten that is at least raw. */
  function niceStep(raw) {
    if (!(raw > 0)) return 1;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var f = raw / mag;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag;
  }
  function tpl(key, vals) {
    var s = S.tpl[key] || key;
    return s.replace(/\{(\w+)\}/g, function (m, k) {
      var v = vals ? vals[k] : undefined;
      return v === undefined || v === null || v === "" ? "—" : String(v);
    });
  }
  function goodL(g) { return (g && S.good[g]) || g || "—"; }
  function actL(a) { return (a && S.action[a]) || a || "—"; }
  function roleOf(tr, id) {
    var spec = tr.agentSpecs[id];
    return (spec && S.spec[spec]) || S.role[tr.agentRoles[id]] || "";
  }
  function frameAt(tr, t) { return tr.frames[Math.min(t, tr.frames.length - 1)]; }
  function trigShort(name) { return String(name || "").split("_")[0]; }
  function statusL(s) { return S.tpl["replay.status." + s] ? tpl("replay.status." + s) : s; }
  /* An engine code in the page's language; free text the engine wrote is shown as is. */
  function reasonL(r) { return (r && S.rejection && S.rejection[r]) || r || "—"; }
  function itemsText(obj) {
    var keys = Object.keys(obj || {}).filter(function (k) { return obj[k]; });
    if (!keys.length) return tpl("replay.ret.nothing");
    return keys.sort().map(function (k) { return goodL(k) + " × " + obj[k]; }).join(", ");
  }
  /* Keyboard access for a clickable element that is not a button. The label
     names it; describedBy, when given, points at the text that explains it. */
  function activatable(node, label, describedBy) {
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "button");
    node.setAttribute("aria-label", label);
    if (describedBy) node.setAttribute("aria-describedby", describedBy);
    node.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        node.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      }
    });
  }
  function pct(v) { var x = parseFloat(v); return isNaN(x) ? "—" : Math.round(x * 100) + "%"; }
  /* What an adventurer set out with and what each item added, as the engine
     fixed it at departure (the expedition entry's gear bonus). Durability comes
     from the observation the agent had just been given. */
  function expeditionOf(s) {
    var found = null;
    (s.e || []).forEach(function (e) { if (e[0] === "expedition") found = e; });
    return found;
  }
  function gearText(bonus, eq) {
    var keys = Object.keys(bonus || {}).sort();
    if (!keys.length) return tpl("replay.gear.none");
    return tpl("replay.gear.carried", { items: keys.map(function (k) {
      var d = (eq || {})[k];
      return goodL(k) + " +" + pct(bonus[k]) + (d ? " (" + tpl("replay.gear.durability", { durability: d }) + ")" : "");
    }).join(", ") });
  }

  /* ---------- per-track precomputation ---------- */
  tracks.forEach(function (tr) {
    var msgs = 0, acts = 0, wasted = 0;
    tr.cum = tr.frames.map(function (f) {
      msgs += (f.m || []).length;
      (f.st || []).forEach(function (s) {
        acts += 1;
        if ((s.e || []).some(function (e) { return e[0] === "wasted"; })) wasted += 1;
      });
      return { msgs: msgs, acts: acts, wasted: wasted };
    });
    tr.ui = { tab: "agents", focus: null, open: {} };
  });

  /* ---------- the shock, in words ---------- */
  function shockText(tr) {
    var s = tr.shock;
    if (!s) return null;
    var from = s.applied_tick, to = tr.reversionTick === null ? horizon : tr.reversionTick - 1;
    var span = { from: from, to: to };
    switch (s.scenario) {
      case "resource_scarcity":
        return tpl("replay.shock.resource_scarcity", Object.assign({ resource: goodL(s.target_resource), pct: pct(s.spawn_multiplier) }, span));
      case "demand_surge":
        return tpl("replay.shock.demand_surge", Object.assign({ product: goodL(s.target_product), times: num(s.consumption_multiplier, 1) }, span));
      case "productivity_drop":
        return tpl("replay.shock.productivity_drop", Object.assign({ specialization: S.spec[s.target_specialization] || s.target_specialization, pct: pct(s.capacity_multiplier) }, span));
      case "liquidity_loss":
        var hit = s.affected_agents || [];
        var everyone = tr.agentIds.length > 0 && tr.agentIds.every(function (id) { return hit.indexOf(id) >= 0; });
        return everyone
          ? tpl("replay.shock.liquidity_loss_all", Object.assign({ pct: pct(s.frozen_gold_fraction) }, span))
          : tpl("replay.shock.liquidity_loss", Object.assign({ agents: hit.join(", "), pct: pct(s.frozen_gold_fraction) }, span));
      default:
        return tpl("replay.shock.generic", Object.assign({ scenario: s.scenario }, span));
    }
  }

  /* ---------- phases ---------- */
  var head = tracks[0];
  function phaseOf(t) {
    if (head.shockTick === null || t < head.shockTick) return "base";
    if (head.reversionTick === null || t < head.reversionTick) return "shock";
    return "recovery";
  }

  /* ---------- charts: every condition on one set of axes ---------- */
  var cv = $("#chart"), cv2 = $("#stockchart");
  var colour = {};
  function readColours() {
    var cs = getComputedStyle(document.documentElement);
    ["a", "b", "c", "ink", "ink-3", "rule", "rule-soft", "shock", "shock-band", "recovery", "recovery-band", "panel", "tax"]
      .forEach(function (k) { colour[k] = cs.getPropertyValue("--" + k).trim(); });
  }
  function seriesOf(tr, valueAt) {
    var out = [];
    for (var t = 0; t <= horizon; t++) {
      var v = valueAt(frameAt(tr, t));
      out.push(v === undefined || v === null ? null : v);
    }
    return out;
  }
  /* Wealth-tax collections, in order. The schedule comes from the world rules
     when the configuration is known, so a collection nobody paid still shows;
     without it, only the collections somebody paid in can be seen. */
  var wealthRules = D.rules && D.rules.wealth ? D.rules.wealth : null;
  var allTaxTicks = (function () {
    var seen = {};
    if (wealthRules) (wealthRules.ticks || []).forEach(function (t) { if (t <= horizon) seen[t] = true; });
    tracks.forEach(function (tr) { (tr.wealthTaxTicks || []).forEach(function (t) { seen[t] = true; }); });
    return Object.keys(seen).map(Number).sort(function (a, b) { return a - b; });
  })();
  function isTaxTick(t) { return allTaxTicks.indexOf(t) >= 0; }
  function exemptionText() {
    if (!wealthRules) return null;
    return parseFloat(wealthRules.exemption) % 1 ? money(wealthRules.exemption) : num(wealthRules.exemption);
  }
  function nextTaxTick() {
    for (var i = 0; i < allTaxTicks.length; i++) if (allTaxTicks[i] > tick) return allTaxTicks[i];
    return null;
  }
  function quoteAt(f, side) { var q = (f.q || {})[good]; return q ? q[side] : null; }
  /* Market price: the last trade, pulled up to a standing bid above it or down
     to a standing ask below it. With buyers offering more and nobody selling,
     the last trade alone would show a price nobody accepts any more. A crossed
     book (an agent's own bid at or above its own ask, which never match) says
     nothing about the price, so the last trade stands. */
  function priceAt(f) {
    var m = f.p[good], bid = quoteAt(f, 0), ask = quoteAt(f, 1);
    if (m === undefined) m = null;
    var has = function (v) { return v !== null && v !== undefined; };
    if (has(bid) && has(ask) && bid >= ask) return m;
    if (bid !== null && bid !== undefined && (m === null || bid > m)) m = bid;
    if (ask !== null && ask !== undefined && (m === null || ask < m)) m = ask;
    return m;
  }
  function stockAt(f) { return f.ws && good in f.ws ? f.ws[good] : (f.ws && Object.keys(f.ws).length ? 0 : null); }
  function zoneAt(f) {
    if (!f.zs) return null;
    var total = 0, seen = false;
    Object.keys(f.zs).forEach(function (z) { if (good in f.zs[z]) { total += f.zs[z][good]; seen = true; } });
    return seen ? total : null;
  }
  var geom = {};
  /* lines: [{values, color, dash, width, dot}]; dot marks the current value. */
  function plot(canvas, lines, opts) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight || 200;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    var g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    var padL = 48, padR = 14, padT = opts.labels ? 22 : 10, padB = 26;
    var iw = w - padL - padR, ih = h - padT - padB;
    var X = function (t) { return padL + (t / Math.max(1, horizon)) * iw; };
    geom[canvas.id] = { padL: padL, iw: iw };

    var lo = Infinity, hi = -Infinity;
    lines.forEach(function (l) { l.values.forEach(function (v) { if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); });

    var st = head.shockTick, rt = head.reversionTick;
    g.font = "12px system-ui, sans-serif"; g.textBaseline = "alphabetic";
    function band(a, b, fill, label, ink) {
      if (fill) { g.fillStyle = fill; g.fillRect(X(a), padT, Math.max(1, X(b) - X(a)), ih); }
      // A label wider than its band (a short phase, a narrow screen) is left out
      // rather than run into the next one.
      if (opts.labels && g.measureText(label).width <= X(b) - X(a) - 8) {
        g.fillStyle = ink; g.textAlign = "left"; g.fillText(label, X(a) + 5, padT - 7);
      }
    }
    band(0, st === null ? horizon : st, null, S.phase.base, colour["ink-3"]);
    if (st !== null) band(st, rt === null ? horizon : rt, colour["shock-band"], S.phase.shock, colour.shock);
    if (rt !== null) band(rt, horizon, colour["recovery-band"], S.phase.recovery, colour.recovery);
    // Where a condition ran out of token budget: from there on only mechanical
    // ticks, no decisions.
    tracks.forEach(function (tr) {
      if (tr.exhaustionTick === null || tr.exhaustionTick === undefined) return;
      g.strokeStyle = colour[tr.condition.toLowerCase()] || colour.ink; g.lineWidth = 1.5; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(X(tr.exhaustionTick) + .5, padT); g.lineTo(X(tr.exhaustionTick) + .5, padT + ih); g.stroke();
      g.setLineDash([]);
    });

    // Wealth-tax collections: a faint dotted line each.
    g.strokeStyle = colour.tax || colour["ink-3"]; g.globalAlpha = .55; g.lineWidth = 1; g.setLineDash([1, 3]);
    allTaxTicks.forEach(function (t) {
      g.beginPath(); g.moveTo(X(t) + .5, padT); g.lineTo(X(t) + .5, padT + ih); g.stroke();
    });
    g.setLineDash([]); g.globalAlpha = 1;

    g.strokeStyle = colour.rule; g.lineWidth = 1;
    g.beginPath(); g.moveTo(padL, padT + ih + .5); g.lineTo(w - padR, padT + ih + .5); g.stroke();
    g.fillStyle = colour["ink-3"]; g.textAlign = "center";
    var step = horizon > 200 ? 50 : horizon > 80 ? 20 : 10;
    for (var xt = 0; xt <= horizon; xt += step) g.fillText(String(xt), X(xt), padT + ih + 17);

    if (lo === Infinity) {
      g.fillStyle = colour["ink-3"]; g.textAlign = "center";
      g.fillText(opts.empty, padL + iw / 2, padT + ih / 2);
      drawPlayhead(g, X, padT, ih);
      return;
    }
    if (opts.fromZero) lo = Math.min(lo, 0);
    if (hi - lo < 1e-9) { hi = lo + 1; lo = Math.max(0, lo - 1); }
    var pad = (hi - lo) * .1; hi += pad; if (!opts.fromZero) lo -= pad;
    // Gridlines sit on a "nice" step (1, 2 or 5 times a power of ten), so every
    // label is a round value the line actually marks. A count axis never steps
    // below 1.
    var gridStep = niceStep((hi - lo) / 4);
    if (opts.integer) gridStep = Math.max(1, gridStep);
    lo = Math.floor(lo / gridStep + 1e-9) * gridStep;
    hi = Math.ceil(hi / gridStep - 1e-9) * gridStep;
    var gridLines = Math.round((hi - lo) / gridStep);
    var decimals = gridStep >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(gridStep) - 1e-9));
    var Y = function (v) { return padT + ih - ((v - lo) / (hi - lo)) * ih; };

    g.textAlign = "right"; g.strokeStyle = colour["rule-soft"];
    for (var i = 0; i <= gridLines; i++) {
      var v = lo + gridStep * i, y = Y(v);
      g.beginPath(); g.moveTo(padL, y + .5); g.lineTo(w - padR, y + .5); g.stroke();
      g.fillStyle = colour["ink-3"]; g.fillText(num(v, decimals), padL - 6, y + 4);
    }

    lines.forEach(function (l) {
      g.strokeStyle = l.color; g.lineWidth = l.width || 2.2; g.lineJoin = "round"; g.setLineDash(l.dash || []);
      g.beginPath();
      var started = false, prev = null;
      for (var t = 0; t <= Math.min(tick, horizon); t++) {
        var v2 = l.values[t];
        if (v2 === null) continue;
        /* a level holds until it changes, so draw steps, not slopes */
        if (!started) { g.moveTo(X(t), Y(v2)); started = true; }
        else { g.lineTo(X(t), Y(prev)); g.lineTo(X(t), Y(v2)); }
        prev = v2;
      }
      if (started) g.lineTo(X(Math.min(tick, horizon)), Y(prev));
      g.stroke(); g.setLineDash([]);
      var cur = l.values[Math.min(tick, horizon)];
      if (cur !== null && cur !== undefined && l.dot) {
        g.fillStyle = l.color;
        g.beginPath(); g.arc(X(tick), Y(cur), 4, 0, 6.2832); g.fill();
      }
    });
    drawPlayhead(g, X, padT, ih);
  }
  function drawChart() {
    var palette = { A: colour.a, B: colour.b, C: colour.c };
    var three = tracks.length === 3;
    plot(cv, tracks.map(function (tr, k) {
      return { values: seriesOf(tr, priceAt), color: palette[tr.condition] || colour.a, dash: k === 2 && three ? [6, 3] : null, dot: true };
    }), { labels: true, empty: tpl("replay.no_trades") });

    var stockLines = tracks.map(function (tr) {
      return { values: seriesOf(tr, stockAt), color: palette[tr.condition] || colour.a, dot: true };
    });
    tracks.forEach(function (tr) {
      var z = seriesOf(tr, zoneAt);
      if (z.some(function (v) { return v !== null; })) {
        stockLines.push({ values: z, color: palette[tr.condition] || colour.a, dash: [2, 3], width: 1.6 });
      }
    });
    plot(cv2, stockLines, { labels: false, empty: tpl("replay.stock.none"), fromZero: true, integer: true });
  }
  function drawPlayhead(g, X, padT, ih) {
    g.strokeStyle = colour.ink; g.globalAlpha = .45; g.lineWidth = 1;
    g.beginPath(); g.moveTo(X(tick) + .5, padT); g.lineTo(X(tick) + .5, padT + ih); g.stroke();
    g.globalAlpha = 1;
  }
  [cv, cv2].forEach(function (canvas) {
    canvas.addEventListener("click", function (e) {
      var gm = geom[canvas.id];
      if (!gm) return;
      var r = canvas.getBoundingClientRect();
      pause(); setTick(Math.round(((e.clientX - r.left - gm.padL) / gm.iw) * horizon));
    });
  });

  /* ---------- "at this tick" comparison ---------- */
  function totals(tr, f) {
    var gold = 0, eff = 0, all = 0, seen = 0;
    tr.agentIds.forEach(function (id) {
      var a = f.a[id]; if (!a || !a[F.observed]) return;
      gold += a[F.gold]; eff += a[F.effective]; all += a[F.activations]; seen += 1;
    });
    return { gold: gold, eff: all ? eff / all : null, all: all, seen: seen, total: tr.agentIds.length };
  }
  function renderNow() {
    $("#nowtitle").textContent = tpl("replay.now.title", { tick: tick });
    var table = $("#now"); table.textContent = "";
    var thead = add(table, el("thead")), hr = add(thead, el("tr"));
    add(hr, el("th", "", ""));
    tracks.forEach(function (tr) {
      var th = add(hr, el("th"));
      add(th, el("span", "swatch sw-" + tr.condition));
      th.appendChild(document.createTextNode(D.labelByModel ? tr.model : tr.condition));
      th.title = S.condition[tr.condition];
    });
    var rows = [
      [tpl("replay.now.price", { good: goodL(good) }), function (tr, f) { return money(priceAt(f)); }],
      [tpl("replay.now.last_trade"), function (tr, f) { return money(f.p[good]); }],
      [tpl("replay.now.best_bid"), function (tr, f) { return money(quoteAt(f, 0)); }],
      [tpl("replay.now.best_ask"), function (tr, f) { return money(quoteAt(f, 1)); }],
      [tpl("replay.now.stock", { good: goodL(good) }), function (tr, f) { return num(stockAt(f)); }],
      [tpl("replay.now.on_sale", { good: goodL(good) }), function (tr, f) { return num((f.os || {})[good] || 0); }],
      [tpl("replay.now.in_zones", { good: goodL(good) }), function (tr, f) { var z = zoneAt(f); return z === null ? "—" : num(z); }],
      [tpl("replay.now.trades"), function (tr, f) { return num(f.tradesTotal); }],
      // Reconstructed from the log for every agent at every tick, so the three
      // conditions compare like with like.
      [tpl("replay.now.gold_adventurers"), function (tr, f) { return num((f.rg || {}).adventurer); }],
      [tpl("replay.now.gold_artisans"), function (tr, f) { return num((f.rg || {}).artisan); }],
      // Gold that has left the economy so far, per charge.
      [tpl("replay.now.sales_tax"), function (tr, f) { return money((f.tx || {}).sales || 0); }],
      [tpl("replay.now.wealth_tax"), function (tr, f) { return money((f.tx || {}).wealth || 0); }],
      [tpl("replay.now.storage"), function (tr, f) { return money((f.tx || {}).storage || 0); }],
      [tpl("replay.now.wealth_last"), function (tr) {
        var last = null;
        allTaxTicks.forEach(function (t) { if (t <= tick) last = t; });
        if (last === null) return "—";
        // The collection itself; arrears settled at the same tick are not part of it.
        var wt = (frameAt(tr, last).wt || []).filter(function (r) { return !r[3]; });
        var paid = wt.reduce(function (s, r) { return s + r[2]; }, 0);
        return tpl("replay.wealth.last_value", { tick: last, amount: money(paid) });
      }],
      [tpl("replay.now.wealth_next"), function () { var n = nextTaxTick(); return n === null ? "—" : "t" + n; }],
      [tpl("replay.now.activations"), function (tr, f, c) { return num(c.acts); }],
      [tpl("replay.now.efficiency"), function (tr, f) {
        var e = totals(tr, f).eff; return e === null ? "—" : Math.round(e * 100) + "%";
      }],
      [tpl("replay.now.messages"), function (tr, f, c) { return num(c.msgs); }],
      [tpl("replay.now.tokens"), function (tr, f) { return num(f.tokTotal); }]
    ];
    var tb = add(table, el("tbody"));
    rows.forEach(function (row) {
      var r = add(tb, el("tr"));
      add(r, el("td", "", row[0]));
      tracks.forEach(function (tr) {
        add(r, el("td", "", row[1](tr, frameAt(tr, tick), tr.cum[Math.min(tick, tr.cum.length - 1)])));
      });
    });
  }

  /* ---------- steps ---------- */
  function decisionText(d) {
    if (!d) return tpl("replay.dec.none");
    if (d.os) {
      return actL(d.ac) + ": " + d.os.map(function (o) {
        if (o[0] === "cancel_order") return actL(o[0]) + " #" + o[2];
        return actL(o[0]) + " " + goodL(o[1]) + " × " + o[2] + (o[3] ? " @ " + money(o[3]) : "");
      }).join("; ");
    }
    var parts = [actL(d.ac)];
    if (d.ac === "send_message" && d.mt) parts.push(S.message[d.mt] || d.mt);
    if (d.tg) parts.push(goodL(d.tg));
    if (d.ac === "cancel_order" && d.q !== null && d.q !== undefined) parts.push("#" + d.q);
    else if (d.q !== null && d.q !== undefined && d.ac !== "wait") {
      parts.push("× " + d.q + (d.p ? " @ " + money(d.p) : ""));
    }
    if (d.r) parts.push(tpl("replay.dec.recipient", { agent: d.r }));
    return parts.join(" ");
  }
  function engineLine(e) {
    switch (e[0]) {
      case "order": return tpl("replay.eng.order", { side: tpl("replay.side." + e[1]), qty: e[3], good: goodL(e[2]), price: money(e[4]), expires: e[6], id: e[5] });
      case "trade": return tpl(e[1] === "buy" ? "replay.eng.trade_buy" : "replay.eng.trade_sell", { qty: e[3], good: goodL(e[2]), price: money(e[4]), other: e[5] });
      case "expedition": return tpl("replay.eng.expedition", { zone: e[1], risk: tpl("replay.risk." + e[2]), probability: Math.round(parseFloat(e[3]) * 100) + "%", back: e[4] });
      case "produce": return tpl("replay.eng.produce", { qty: e[2], product: goodL(e[1]) });
      case "noproduce": return tpl("replay.eng.noproduce", { product: goodL(e[1]), reason: reasonL(e[2]) });
      case "equip": return tpl("replay.eng.equip", { product: goodL(e[1]), durability: e[2] });
      case "msg": return tpl("replay.eng.message", { type: S.message[e[1]] || e[1], to: e[2] });
      case "cancel": return tpl("replay.eng.cancel", { id: e[1] });
      case "rejected":
        if (e[2] !== null && e[2] !== undefined) return tpl("replay.eng.rejected_order", { n: e[2] + 1, reason: reasonL(e[1]) });
        return tpl("replay.eng.rejected", { reason: reasonL(e[1]) });
      case "wasted": return tpl("replay.eng.wasted", { reason: reasonL(e[1]) });
      default: return e.join(" ");
    }
  }
  function marksFor(s, tr) {
    var out = [];
    var kinds = (s.e || []).map(function (e) { return e[0]; });
    var wasted = kinds.indexOf("wasted") >= 0;
    // A place_orders with some orders refused and others taken is partial.
    if (kinds.indexOf("rejected") >= 0) out.push(s.ok ? ["waste", tpl("replay.mark.partial")] : ["bad", tpl("replay.mark.rejected")]);
    else if (s.ok && !wasted) out.push(["ok", tpl("replay.mark.ok")]);
    if (wasted) out.push(["waste", tpl("replay.mark.wasted")]);
    if ((s.x || []).length) out.push(["bad", "↻ " + s.x.length]);
    var ex = expeditionOf(s);
    if (ex) {
      var bonus = ex[5] || {}, n = Object.keys(bonus).length, sum = 0;
      Object.keys(bonus).forEach(function (k) { sum += parseFloat(bonus[k]); });
      out.push([n ? "info" : "waste", n ? "⚔ +" + Math.round(sum * 100) + "%" : tpl("replay.gear.none_short")]);
    }
    (s.e || []).forEach(function (e) {
      if (e[0] === "trade") out.push(["info", (e[1] === "buy" ? "+" : "−") + e[3] + " " + goodL(e[2]) + " @ " + money(e[4])]);
      if (e[0] === "msg") out.push(["info", "✉ " + (S.message[e[1]] || e[1]) + " → " + e[2]]);
    });
    if (s.r) out.push([s.r.st === "SUCCESS" ? "ok" : "waste", tpl("replay.result_status", { status: statusL(s.r.st) })]);
    return out;
  }

  function stepItem(tr, s, t, showTick) {
    var key = t + ":" + s.n;
    var li = el("li", "step" + (tr.ui.open[key] ? " open" : ""));
    var b = add(li, el("button", "summary"));
    b.type = "button";
    // Stable across re-renders, so focus can return to it (see renderColumn).
    b.setAttribute("data-key", "step:" + key);
    b.setAttribute("aria-expanded", tr.ui.open[key] ? "true" : "false");
    add(b, el("span", "num", String(s.n)));
    var body = add(b, el("span", ""));
    var who = add(body, el("span", "who"));
    add(who, el("span", "agent", s.a));
    add(who, el("span", "role", roleOf(tr, s.a)));
    var w = s.w || {};
    if (w.task) {
      var c = add(who, el("span", "chip task", S.task[w.task.ty] || w.task.ty));
      c.title = tpl("replay.why.task") + " · " + (S.priority[w.task.pr] || w.task.pr || "");
    } else if (w.tr) {
      w.tr.forEach(function (name) {
        var ch = add(who, el("span", "chip", trigShort(name)));
        ch.title = S.trigger[name] || name;
      });
    }
    if (showTick) add(who, el("span", "tickref", "t" + t));
    add(body, el("span", "decision", decisionText(s.d)));
    if (s.d && s.d.why) add(body, el("span", "reason", "“" + s.d.why + "”"));
    var marks = add(body, el("span", "marks"));
    marksFor(s, tr).forEach(function (m) { add(marks, el("span", "mark " + m[0], m[1])); });

    b.onclick = function () {
      tr.ui.open[key] = !tr.ui.open[key];
      renderColumn(tr.col, tr);
    };
    if (tr.ui.open[key]) li.appendChild(stepDetail(tr, s, t));
    return li;
  }

  function returnItem(tr, r, t, showTick) {
    var li = el("li", "ret " + (r.ok ? "ok" : "bad"));
    var who = add(li, el("div", "who"));
    add(who, el("span", "agent", r.a));
    add(who, el("span", "role", roleOf(tr, r.a)));
    add(who, el("span", "mark " + (r.ok ? "ok" : "bad"), tpl("replay.outcome." + (r.ok ? "success" : "failure"))));
    if (showTick) add(who, el("span", "tickref", "t" + t));
    add(li, el("div", "", tpl("replay.ret.line", { zone: r.z, risk: tpl("replay.risk." + r.r), probability: pct(r.pr) })));
    var got = Object.keys(r.y || {}).some(function (k) { return r.y[k]; });
    var gainKey = r.ok ? (got ? "replay.ret.success" : "replay.ret.success_empty") : (got ? "replay.ret.failure" : "replay.ret.failure_empty");
    add(li, el("div", "gain", tpl(gainKey, { items: itemsText(r.y) })));
    if (parseFloat(r.gd) > 0) add(li, el("div", "gain", tpl("replay.ret.gold", { gold: money(r.gd) })));
    // Every return is followed by a rest of rest_ticks; a failure adds
    // recovery_ticks_on_failure for the risk taken.
    if (r.rec > t) add(li, el("div", "muted", tpl("replay.ret.recovering", { tick: r.rec })));
    if ((r.br || []).length) add(li, el("div", "broken", tpl("replay.ret.broken", { items: r.br.map(goodL).join(", ") })));
    return li;
  }

  function kvList(parent, obj, label) {
    var keys = Object.keys(obj || {});
    if (!keys.length) { add(parent, el("p", "muted", tpl("replay.d.none"))); return; }
    var box = add(parent, el("div", "kv"));
    keys.sort().forEach(function (k) { add(box, el("span", "", label(k) + ": " + obj[k])); });
  }
  function smallTable(parent, headers, rows, numeric) {
    var t = add(parent, el("table"));
    var hr = add(add(t, el("thead")), el("tr"));
    headers.forEach(function (h, i) { add(hr, el("th", numeric[i] ? "n" : "", h)); });
    var tb = add(t, el("tbody"));
    rows.forEach(function (r) {
      var row = add(tb, el("tr"));
      r.forEach(function (v, i) { add(row, el("td", numeric[i] ? "n" : "", v)); });
    });
  }
  function jsonish(items) {
    return (items || []).map(function (it) {
      if (typeof it !== "object" || it === null) return String(it);
      return Object.keys(it).map(function (k) {
        var v = it[k];
        if (k === "good" || k === "resource" || k === "product") v = goodL(v);
        return k + ": " + v;
      }).join(", ");
    });
  }

  function stepDetail(tr, s, t) {
    var d = el("div", "detail");
    var w = s.w || {}, o = s.o || {};

    add(d, el("h4", "", tpl("replay.d.why")));
    if (w.task) {
      var tk = w.task;
      add(d, el("p", "", tpl("replay.why.task") + ": " + (S.task[tk.ty] || tk.ty) + " · " +
        tpl("replay.dec.priority", { priority: S.priority[tk.pr] || tk.pr }) + " · " + (tk.id || "")));
      if (tk.p && Object.keys(tk.p).length) kvList(d, tk.p, function (k) { return k; });
      if (tk.ctx && tk.ctx.length) add(d, el("p", "muted", tpl("replay.why.context", { results: tk.ctx.join(", ") })));
      if (w.dc) {
        var dp = add(d, el("p", "muted", tpl("replay.why.director_call", { n: w.dc.n, input: num(w.dc.it), output: num(w.dc.ot), remaining: w.dc.rem })));
        promptButton(dp, tr, tpl("replay.prompt.title_director", { tick: t, n: w.dc.n }), [w.dc.ps, w.dc.pu, w.dc.pr], "pd:" + t + ":" + s.n);
      }
    } else if (w.tr) {
      add(d, el("p", "", tpl("replay.why.triggers") + ": " + w.tr.map(function (n) { return S.trigger[n] || n; }).join("; ")));
      if (w.wt) add(d, el("p", "muted", tpl(w.wt === 1 ? "replay.why.waited_one" : "replay.why.waited", { ticks: w.wt })));
    } else {
      add(d, el("p", "muted", tpl("replay.why.unknown")));
    }

    add(d, el("h4", "", tpl("replay.d.observed")));
    add(d, el("p", "", tpl("replay.d.gold", { gold: money(o.g), available: money(o.av) })));
    if (o.away) add(d, el("p", "muted", tpl("replay.d.away")));
    if (tr.agentRoles[s.a] === "artisan") add(d, el("p", "muted", tpl("replay.d.capacity", { capacity: o.cap })));
    add(d, el("p", "muted", tpl("replay.d.inventory")));
    kvList(d, o.inv, goodL);
    if (o.eq && Object.keys(o.eq).length) {
      add(d, el("p", "muted", tpl("replay.d.equipment")));
      kvList(d, o.eq, goodL);
    }
    if ((o.ord || []).length) {
      add(d, el("p", "muted", tpl("replay.d.orders")));
      smallTable(d, [tpl("replay.col.order"), tpl("replay.col.side"), tpl("replay.col.good"), tpl("replay.col.qty"), tpl("replay.col.price"), tpl("replay.col.expires")],
        o.ord.map(function (r) { return [r[5] !== null && r[5] !== undefined ? "#" + r[5] : "—", tpl("replay.side." + r[0]), goodL(r[1]), r[2], money(r[3]), "t" + r[4]]; }),
        [false, false, false, true, true, true]);
    }
    if ((o.mk || []).length) {
      add(d, el("p", "muted", tpl("replay.d.market")));
      smallTable(d, [tpl("replay.col.good"), tpl("replay.col.bid"), tpl("replay.col.ask"), tpl("replay.col.last")],
        o.mk.map(function (r) { return [goodL(r[0]), money(r[1]), money(r[2]), money(r[3])]; }),
        [false, true, true, true]);
    }
    if ((o.in || []).length) {
      add(d, el("p", "muted", tpl("replay.d.inbox")));
      var ul = add(d, el("ul"));
      o.in.forEach(function (m) {
        var what = [S.message[m[0]] || m[0], "← " + m[1]];
        if (m[2]) what.push(S.message[m[2]] || goodL(m[2]));
        if (m[3]) what.push("× " + m[3] + (m[4] ? " @ " + money(m[4]) : ""));
        if (m[5] !== null && m[5] !== undefined) what.push("(t" + m[5] + ")");
        add(ul, el("li", "", what.join(" ")));
      });
    }
    if (o.bul) add(d, el("p", "", tpl("replay.d.bulletin")));

    add(d, el("h4", "", tpl("replay.d.model")));
    if (!(s.l || []).length) add(d, el("p", "muted", tpl("replay.llm.none")));
    // Calls and invalid outputs in attempt order: attempt 1's failure before
    // attempt 2's call.
    var lines = (s.l || []).map(function (c) { return ["l", c[0], c]; })
      .concat((s.x || []).map(function (x) { return ["x", x[0], x]; }))
      .sort(function (a, b) { return a[1] - b[1] || (a[0] === "x" ? -1 : 1); });
    lines.forEach(function (entry) { if (entry[0] === "l") callLine(entry[2]); else invalidLine(entry[2]); });
    function callLine(c) {
      var line = tpl("replay.llm.call", { attempt: c[0], input: num(c[1]), output: num(c[2]) });
      var bk = Object.keys(c[4] || {}).map(function (k) { return (S.bucket[k] || k) + " " + num(c[4][k]); }).join(", ");
      var p = add(d, el("p", "", line));
      promptButton(p, tr, tpl("replay.prompt.title_agent", { agent: s.a, tick: t, attempt: c[0] }), c[5], "pc:" + t + ":" + s.n + ":" + c[0]);
      // Cut off by the output cap: the model ran out of room.
      if (c[3] === "length") add(d, el("p", "muted", tpl("replay.llm.truncated")));
      if (bk) add(d, el("p", "muted", tpl("replay.llm.buckets", { buckets: bk })));
    }
    function invalidLine(x) {
      var label = x[3] ? tpl("replay.invalid." + x[3]) : x[1];
      var p = add(d, el("p", "", tpl("replay.llm.invalid", { attempt: x[0], reason: label })));
      // The parser's own words, in English, for whoever wants the detail.
      if (x[3] && x[1]) p.title = x[1];
      // A call that failed outright: its prompt, and no reply.
      if (x[2]) promptButton(p, tr, tpl("replay.prompt.title_agent", { agent: s.a, tick: t, attempt: x[0] }), [x[2][0], x[2][1], null], "px:" + t + ":" + s.n + ":" + x[0]);
    }

    add(d, el("h4", "", tpl("replay.d.decided")));
    add(d, el("p", "", decisionText(s.d)));
    if (s.d && s.d.pr) add(d, el("p", "muted", tpl("replay.dec.priority", { priority: S.priority[s.d.pr] || s.d.pr })));
    if (s.d && s.d.why) add(d, el("p", "reason", "“" + s.d.why + "”"));

    add(d, el("h4", "", tpl("replay.d.engine")));
    if (!(s.e || []).length) add(d, el("p", "muted", tpl("replay.eng.nothing")));
    else {
      var eu = add(d, el("ul"));
      s.e.forEach(function (e) {
        add(eu, el("li", "", engineLine(e)));
        if (e[0] === "expedition") add(eu, el("li", "muted", gearText(e[5], o.eq)));
      });
    }

    if (s.r) {
      add(d, el("h4", "", tpl("replay.d.result")));
      add(d, el("p", "", s.r.id ? tpl("replay.result_status_id", { id: s.r.id, status: statusL(s.r.st) })
        : tpl("replay.result_status", { status: statusL(s.r.st) })));
      [["needs", "replay.result.needs"], ["offers", "replay.result.offers"], ["obs", "replay.result.observations"]].forEach(function (p) {
        var items = jsonish(s.r[p[0]]);
        add(d, el("p", "muted", tpl(p[1]) + ": " + (items.length ? items.join(" · ") : tpl("replay.d.none"))));
      });
    }
    return d;
  }

  function coordinationText(tr, f) {
    var sc = f.sc || {};
    var lines = [];
    if (tr.condition === "A") {
      var calls = (sc.calls || []), end = sc.end || {};
      var cap = calls.length ? calls[0].cap : tr.directorCap;
      var tasks = (f.st || []).length;
      lines.push([tpl("replay.sched.a", {
        used: calls.length, cap: cap, tasks: tasks, retries: end.director_retries || 0
      }), ""]);
      var closure = end.closed_by_decision ? tpl("replay.sched.closed_director")
        : end.closed_by_cap ? tpl("replay.sched.closed_cap")
        : end.closed_by_budget ? tpl("replay.sched.closed_budget")
        : cap ? tpl("replay.sched.closed_k", { k: cap - 1 }) : "";
      if (closure) lines.push([closure, ""]);
      if ((sc.errors || []).length) lines.push([tpl("replay.sched.errors", { count: sc.errors.length }), "warn"]);
    } else {
      var end = sc.end || {};
      lines.push([tpl("replay.sched.bc", {
        requested: sc.requested || 0, served: (sc.served || []).length,
        deferred: (sc.deferred || []).length, k: end.k
      }), ""]);
      if ((sc.deferred || []).length) {
        lines.push([tpl("replay.sched.waiting", { agents: sc.deferred.map(function (x) { return x[0] + " (" + x[1].map(trigShort).join(", ") + ")"; }).join(", ") }), ""]);
      }
    }
    if (f.x && !(f.st || []).length) lines.push([tpl("replay.sched.exhausted"), "warn"]);
    return lines;
  }

  /* ---------- tabs ---------- */
  function everOwed(tr) {
    if (tr.everOwed === undefined) {
      tr.everOwed = F.owed !== undefined && tr.frames.some(function (fr) {
        return Object.keys(fr.a || {}).some(function (id) { return fr.a[id][F.owed] > 0; });
      });
    }
    return tr.everOwed;
  }
  function renderTab(tr, box, f) {
    box.textContent = "";
    var tab = tr.ui.tab;
    if (tab === "agents") {
      add(box, el("p", "hint", tpl("replay.agents.hint")));
      var acted = {};
      (f.st || []).forEach(function (s) { acted[s.a] = true; });
      var t = add(box, el("table", "agents"));
      var hr = add(add(t, el("thead")), el("tr"));
      // Wealth tax still owed: a column only in a run where somebody ever owed.
      var owes = everOwed(tr);
      var heads = [["replay.col.agent", ""], ["replay.col.gold", "n"], ["replay.col.items", "n"], ["replay.col.taxes", "n"]];
      if (owes) heads.push(["replay.col.tax_owed", "n"]);
      heads.concat([["replay.col.last_action", ""], ["replay.col.effective_wasted", "n"]])
        .forEach(function (h) { add(hr, el("th", h[1], tpl(h[0]))); });
      var tb = add(t, el("tbody"));
      tr.agentIds.forEach(function (id, k) {
        var a = f.a[id] || [0, 0, "", "", 0, 0, 0, 0, 0, 0];
        var seen = a[F.observed];
        // Reconstructed from the log for every tick; the observed gold is only
        // what the agent was last shown, and goes stale between activations.
        var gold = f.ag ? f.ag[k] : (seen ? a[F.gold] : null);
        // Items likewise; without final holdings, the count the agent was last shown.
        var items = f.ai ? f.ai[k] : (seen ? a[F.items] : null);
        var known = f.ag && f.ai ? true : seen;
        var row = add(tb, el("tr", (acted[id] ? "acted " : "") + (known ? "" : "unseen ") + (tr.ui.focus === id ? "focus" : "")));
        var name = add(row, el("td", a[F.away] ? "away" : ""));
        var pick = add(name, el("button", "agentbtn", id));
        pick.type = "button";
        pick.title = roleOf(tr, id);
        pick.setAttribute("data-key", "agent:" + id);
        pick.setAttribute("aria-pressed", tr.ui.focus === id ? "true" : "false");
        pick.setAttribute("aria-label", tpl("replay.aria.follow_agent", { agent: id }));
        add(row, el("td", "n", gold === null || gold === undefined ? "—" : num(gold)));
        add(row, el("td", "n", items === null || items === undefined ? "—" : num(items)));
        add(row, el("td", "n", a[F.taxes] ? money(a[F.taxes]) : "—"));
        if (owes) add(row, el("td", "n", a[F.owed] ? money(a[F.owed]) : "—"));
        var cell = add(row, el("td"));
        add(cell, el("span", "pill", a[F.action] ? actL(a[F.action]) : "—"));
        add(row, el("td", "n", a[F.effective] + " / " + a[F.activations]));
        // The whole row answers a click; the button in its first cell gives
        // keyboard access without breaking the table's semantics.
        row.onclick = function () {
          tr.ui.focus = tr.ui.focus === id ? null : id;
          renderColumn(tr.col, tr);
          writeHash(tr);
        };
      });
    } else if (tab === "feed") {
      var n = 0;
      for (var t2 = tick; t2 >= 0 && n < 60; t2--) {
        (frameAt(tr, t2).f || []).forEach(function (e) {
          if (n >= 60) return;
          n += 1;
          var hi = e.e === "shock_applied" || e.e === "shock_reverted" || e.e === "budget_exhausted" || e.e === "director_error";
          if (e.e === "more") {
            var mrow = add(box, el("div", "feedrow"));
            add(mrow, el("span", "t", "t" + t2));
            add(mrow, el("span", "muted", tpl("replay.feed.more", { n: e.q })));
            return;
          }
          if (e.e === "tax_paid") {
            var trow = add(box, el("div", "feedrow tax"));
            add(trow, el("span", "t", "t" + t2));
            // A tick that settled only arrears had no collection, so it is not
            // described as one; arrears are never counted as collected.
            var arrearsTxt = tpl("replay.wealth.arrears", { amount: money(e.ar), payers: e.aq });
            var collected = tpl("replay.feed.wealth_tax", { amount: money(e.v), agents: e.q });
            if (e.c && !e.q) {
              // A collection nobody paid is said so, not as "collected 0".
              var owers = (frameAt(tr, t2).wt || []).filter(function (r) { return !r[3] && !(r[2] > 0); }).length;
              collected = owers
                ? tpl(owers === 1 ? "replay.wealth.nobody_unpaid_one" : "replay.wealth.nobody_unpaid", { unpaid: owers })
                : tpl("replay.wealth.nobody" + (wealthRules ? "" : "_bare"), { exemption: exemptionText() });
            }
            add(trow, el("span", "", e.c
              ? collected + (e.aq ? " · " + arrearsTxt : "")
              : arrearsTxt));
            return;
          }
          var exp = e.e === "expedition_resolved" ? (e.g === "success" ? " ok" : " bad") : "";
          var row = add(box, el("div", "feedrow" + (hi ? " hi" : "") + exp));
          add(row, el("span", "t", "t" + t2));
          var txt = S.event[e.e] || e.e;
          if (e.e === "expedition_resolved") txt += " · " + tpl("replay.outcome." + e.g) + (e.q ? " × " + e.q : "");
          else {
            if (e.g) txt += " · " + goodL(e.g);
            if (e.s) txt += " · " + (S.scenario[e.s] || e.s);
            if (e.v) txt += " @ " + money(e.v);
            if (e.q) txt += " × " + e.q;
            if (e.tx) txt += " · " + tpl("replay.feed.sales_tax", { amount: money(e.tx) });
          }
          if (e.a) txt += " (" + e.a + ")";
          add(row, el("span", "", txt));
        });
      }
      if (!n) add(box, el("p", "empty", tpl("replay.feed.empty")));
    } else if (tab === "messages") {
      var m = 0;
      for (var t3 = tick; t3 >= 0 && m < 60; t3--) {
        (frameAt(tr, t3).m || []).forEach(function (msg) {
          if (m >= 60) return;
          m += 1;
          var row = add(box, el("div", "feedrow"));
          add(row, el("span", "t", "t" + t3));
          var txt = (S.message[msg.ty] || msg.ty) + " · " + msg.from + " → " + msg.to;
          if (msg.g) txt += " · " + goodL(msg.g);
          if (msg.q) txt += " · × " + msg.q + (msg.p ? " @ " + money(msg.p) : "");
          add(row, el("span", "", txt));
        });
      }
      if (!m) add(box, el("p", "empty", tpl(tr.condition === "A" ? "replay.messages.none_in_a" : "replay.messages.empty")));
    } else if (tab === "tokens") {
      var max = 1;
      Object.keys(f.tok).forEach(function (k) { if (f.tok[k] > max) max = f.tok[k]; });
      ["agent", "director", "p2p_content", "bulletin"].forEach(function (name) {
        var v = f.tok[name] || 0;
        var bar = add(box, el("div", "bar " + name));
        var lbl = add(bar, el("div", "lbl"));
        add(lbl, el("span", "", S.bucket[name]));
        add(lbl, el("b", "", num(v)));
        add(add(bar, el("div", "track")), el("div", "fill")).style.width = (v / max * 100).toFixed(1) + "%";
      });
      add(box, el("p", "", tpl("replay.tokens.total", { total: num(f.tokTotal) })));
      if (tr.budgetTotal) {
        var share = Math.min(100, f.tokTotal / tr.budgetTotal * 100);
        var bb = add(box, el("div", "bar budget" + (f.x ? " exhausted" : "")));
        var bl = add(bb, el("div", "lbl"));
        add(bl, el("span", "", tpl("replay.budget")));
        add(bl, el("b", "", num(f.tokTotal) + " / " + num(tr.budgetTotal)));
        add(add(bb, el("div", "track")), el("div", "fill")).style.width = share.toFixed(1) + "%";
        if (f.x) add(box, el("p", "exhaust", tpl("replay.exhausted_at", { tick: tr.exhaustionTick })));
      }
    } else if (tab === "stock") {
      add(box, el("p", "hint", tpl("replay.stock.observer")));
      var prevF = tick > 0 ? frameAt(tr, tick - 1) : null;
      var goods = Object.keys(f.ws || {}).sort(function (x, y) { return goodL(x).localeCompare(goodL(y), S.locale); });
      if (!goods.length) { add(box, el("p", "empty", tpl("replay.stock.none"))); return; }
      smallTable(box, [tpl("replay.col.good"), tpl("replay.col.stock"), tpl("replay.col.on_sale"), tpl("replay.col.equipped"), tpl("replay.col.change")],
        goods.map(function (gd) {
          var now = f.ws[gd], before = prevF && prevF.ws ? (prevF.ws[gd] || 0) : now, d = now - before;
          return [goodL(gd), num(now), num((f.os || {})[gd] || 0), (f.eqc || {})[gd] ? num(f.eqc[gd]) : "—", d ? (d > 0 ? "+" : "−") + Math.abs(d) : ""];
        }), [false, true, true, true, true]);
      if (f.zs) {
        add(box, el("p", "muted", tpl("replay.stock.zones")));
        var zones = Object.keys(f.zs), resources = {};
        zones.forEach(function (z) { Object.keys(f.zs[z]).forEach(function (r) { resources[r] = true; }); });
        var res = Object.keys(resources).sort();
        smallTable(box, [tpl("replay.col.zone")].concat(res.map(goodL)),
          zones.map(function (z) { return [z].concat(res.map(function (r) { return r in f.zs[z] ? num(f.zs[z][r]) : "—"; })); }),
          [false].concat(res.map(function () { return true; })));
      } else {
        add(box, el("p", "muted", tpl("replay.stock.no_zones")));
      }
    } else if (tab === "bulletin") {
      var bu = null, at = null;
      for (var t4 = tick; t4 >= 0; t4--) { var fb = frameAt(tr, t4); if (fb.bu) { bu = fb.bu; at = t4; break; } }
      if (!bu) { add(box, el("p", "empty", tpl("replay.bulletin.empty"))); return; }
      add(box, el("p", "hint", "t" + at));
      smallTable(box, [tpl("replay.col.good"), tpl("replay.col.mean_price"), tpl("replay.col.trades"), tpl("replay.col.supply"), tpl("replay.col.unmet")],
        bu.map(function (r) { return [goodL(r[0]), money(r[1]), r[2], r[3], r[4]]; }), [false, true, true, true, true]);
    }
  }

  /* ---------- one column ---------- */
  function renderColumn(col, tr) {
    // Rebuilding the lists would drop keyboard focus to the page; remember the
    // focused control by its stable key and focus its replacement afterwards.
    var active = document.activeElement;
    var focusKey = active && col.el.contains(active) ? active.getAttribute("data-key") : null;
    renderColumnBody(col, tr);
    if (focusKey) {
      var again = col.el.querySelector('[data-key="' + focusKey.replace(/["\\]/g, "\\$&") + '"]');
      if (again && !again.closest("[hidden]")) again.focus();
    }
  }
  function renderColumnBody(col, tr) {
    var f = frameAt(tr, tick);
    col.coord.textContent = "";
    coordinationText(tr, f).forEach(function (l) { add(col.coord, el("div", l[1], l[0])); });
    // Every Director call of the tick, including those that closed the cycle
    // or failed: each one's prompt and reply.
    var dcalls = ((f.sc || {}).calls || []).filter(function (c) { return c.pu !== null && c.pu !== undefined; });
    if (dcalls.length) {
      var row = add(col.coord, el("div", "pcalls"));
      add(row, el("span", "", tpl("replay.prompt.director_calls")));
      dcalls.forEach(function (c) {
        // A call whose output the Director protocol rejected says so.
        var kindKey = !c.err ? "replay.prompt.call_n"
          : (c.pr === null || c.pr === undefined) ? "replay.prompt.call_n_noreply" : "replay.prompt.call_n_invalid";
        var b = add(row, el("button", "pbtn" + (c.err ? " bad" : ""), tpl(kindKey, { n: c.n })));
        b.type = "button";
        if (c.err) b.title = String(c.err);
        b.setAttribute("data-key", tr.condition + ":dc:" + tick + ":" + c.n);
        b.onclick = function () { showPrompt(tr, tpl("replay.prompt.title_director", { tick: tick, n: c.n }), [c.ps, c.pu, c.pr]); };
      });
    }

    var list = col.steps; list.textContent = "";
    col.returns.textContent = "";
    col.stepsBack.hidden = !tr.ui.focus;
    if (!tr.ui.focus && tr.shock && (tick === tr.shockTick || tick === tr.reversionTick)) {
      var starting = tick === tr.shockTick;
      var card = add(col.returns, el("div", "shockcard " + (starting ? "on" : "off")));
      add(card, el("strong", "", tpl(starting ? "replay.shock.starts" : "replay.shock.ends")));
      if (starting) add(card, el("div", "", shockText(tr)));
    }
    if (!tr.ui.focus && ((f.wt || []).length || isTaxTick(tick))) {
      var wt = f.wt || [];
      var sum = function (rows) { return rows.reduce(function (s, r) { return s + r[2]; }, 0); };
      // Paying: assessed at this collection and paid something now.
      var current = wt.filter(function (r) { return !r[3] && r[2] > 0; });
      var arrears = wt.filter(function (r) { return r[3]; });
      // Assessed at this collection but paid nothing: the amount is now owed.
      var unpaid = wt.filter(function (r) { return !r[3] && !(r[2] > 0); });
      // Below the exempt amount: every agent with no row at this tick. An agent
      // that only settled arrears, or that was assessed and could not pay, is
      // not below it.
      var listed = {};
      // Assessed at this collection (paying or now owing); settling earlier
      // arrears is not an assessment, so those agents count with the rest.
      wt.forEach(function (r) { if (!r[3]) listed[r[0]] = true; });
      var below = tr.agentIds.filter(function (a) { return !listed[a]; }).length;
      var tc = add(col.returns, el("div", "taxcard"));
      add(tc, el("strong", "", tpl(isTaxTick(tick) ? "replay.wealth.title" : "replay.wealth.arrears_title")));
      if (isTaxTick(tick)) {
        // Not paying is not the same as exempt: an agent whose gold sits frozen,
        // or who cannot pay now, owes it as arrears. The minimum is shown only
        // when it comes from a configuration proven to be the run's.
        var bare = wealthRules ? "" : "_bare";
        var key = current.length ? "replay.wealth.summary" + bare
          : unpaid.length ? (unpaid.length === 1 ? "replay.wealth.nobody_unpaid_one" : "replay.wealth.nobody_unpaid")
          : "replay.wealth.nobody" + bare;
        var owingTxt = current.length && unpaid.length
          ? " " + tpl("replay.wealth.summary_unpaid", { unpaid: unpaid.length })
          : "";
        add(tc, el("div", "", tpl(key, {
          unpaid: unpaid.length,
          amount: money(sum(current)), payers: current.length, others: below,
          exemption: exemptionText()
        }) + owingTxt));
      }
      if (arrears.length) {
        add(tc, el("div", "", tpl("replay.wealth.arrears", { amount: money(sum(arrears)), payers: arrears.length })));
      }
      if (wt.length) {
        // Only a collection that left somebody owing gets the column.
        var owing = wt.some(function (r) { return r[4] > 0; });
        var heads = [tpl("replay.col.agent"), tpl("replay.wealth.taxable"), tpl("replay.wealth.paid")];
        if (owing) heads.push(tpl("replay.wealth.owed"));
        smallTable(tc, heads, wt.map(function (r) {
          var row = [r[0], r[1] === null || r[1] === undefined ? "—" : money(r[1]),
            money(r[2]) + (r[3] ? " · " + tpl("replay.wealth.arrears_tag") : "")];
          if (owing) row.push(r[4] === null || r[4] === undefined ? "—" : money(r[4]));
          return row;
        }), owing ? [false, true, true, true] : [false, true, true]);
      }
    }
    if (!tr.ui.focus && (f.ex || []).length) {
      add(col.returns, el("h3", "", tpl("replay.ret.title")));
      var rl = add(col.returns, el("ul", "rets"));
      f.ex.forEach(function (r) { rl.appendChild(returnItem(tr, r, tick, false)); });
    }
    if (tr.ui.focus) {
      col.stepsTitle.textContent = tpl("replay.steps.history", { agent: tr.ui.focus, tick: tick });
      var found = 0;
      for (var t = Math.min(tick, tr.frames.length - 1); t >= 0 && found < 40; t--) {
        (tr.frames[t].st || []).slice().reverse().forEach(function (s) {
          if (s.a !== tr.ui.focus || found >= 40) return;
          found += 1;
          list.appendChild(stepItem(tr, s, t, true));
        });
        (tr.frames[t].ex || []).forEach(function (r) {
          if (r.a !== tr.ui.focus || found >= 40) return;
          found += 1;
          list.appendChild(returnItem(tr, r, t, true));
        });
      }
      if (!found) add(list, el("li", "empty", tpl("replay.steps.history_empty", { agent: tr.ui.focus })));
    } else {
      col.stepsTitle.textContent = tpl("replay.steps.title", { tick: tick });
      (f.st || []).forEach(function (s) { list.appendChild(stepItem(tr, s, tick, false)); });
      if (!(f.st || []).length) add(list, el("li", "empty", tpl("replay.steps.empty")));
    }

    col.tabButtons.forEach(function (b) {
      var on = b.dataset.tab === tr.ui.tab;
      b.classList.toggle("on", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.setAttribute("tabindex", on ? "0" : "-1");
      if (on) col.tabBody.setAttribute("aria-labelledby", b.id);
    });
    renderTab(tr, col.tabBody, f);
  }

  /* ---------- build DOM ---------- */
  /* Every tab a column can show; the bulletin only in Condition C. */
  var TABS = [["agents", "replay.tab.agents"], ["stock", "replay.tab.stock"], ["feed", "replay.tab.feed"], ["messages", "replay.tab.messages"], ["tokens", "replay.tab.tokens"], ["bulletin", "replay.tab.bulletin"]];
  var grid = $("#grid");
  grid.style.setProperty("--cols", tracks.length);
  tracks.forEach(function (tr, ti) {
    var c = add(grid, el("article", "col " + tr.condition));
    var h = add(c, el("header", "col-head"));
    add(h, el("h2", "", D.labelByModel ? S.condition[tr.condition] + " · " + tr.model : S.condition[tr.condition]));
    add(h, el("p", "", tpl("replay.desc." + tr.condition)));
    if (tr.shock) add(c, el("p", "seen", tpl("replay.shock.seen." + tr.condition)));
    var coord = add(c, el("div", "coord"));
    var rets = add(c, el("div", "returns"));
    var sh = add(c, el("div", "steps-head"));
    var stitle = add(sh, el("h3"));
    var back = add(sh, el("button", "", tpl("replay.steps.back")));
    back.type = "button";
    back.onclick = function () {
      var was = tr.ui.focus;
      tr.ui.focus = null; renderColumn(tr.col, tr); writeHash(tr);
      // The button hides itself; hand focus to the agent that was followed.
      var btn = was && c.querySelector('[data-key="agent:' + was + '"]');
      if (btn) btn.focus();
    };
    var steps = add(c, el("ol", "steps"));
    var tabs = add(c, el("div", "tabs"));
    tabs.setAttribute("role", "tablist");
    var defs = TABS.filter(function (d) { return d[0] !== "bulletin" || tr.condition === "C"; });
    var panelId = "panel-" + ti;
    var buttons = defs.map(function (d) {
      var b = add(tabs, el("button", "", tpl(d[1])));
      b.type = "button"; b.dataset.tab = d[0]; b.setAttribute("role", "tab");
      b.id = "tab-" + ti + "-" + d[0];
      b.setAttribute("aria-controls", panelId);
      b.onclick = function () { tr.ui.tab = d[0]; renderColumn(tr.col, tr); writeHash(tr); };
      return b;
    });
    /* Arrow keys move between tabs, Home and End to the first and last. */
    tabs.addEventListener("keydown", function (e) {
      var i = buttons.indexOf(e.target);
      if (i < 0) return;
      var j = e.key === "ArrowRight" ? (i + 1) % buttons.length
        : e.key === "ArrowLeft" ? (i - 1 + buttons.length) % buttons.length
        : e.key === "Home" ? 0 : e.key === "End" ? buttons.length - 1 : -1;
      if (j < 0) return;
      e.preventDefault(); e.stopPropagation();
      tr.ui.tab = buttons[j].dataset.tab; renderColumn(tr.col, tr);
      buttons[j].focus(); writeHash(tr);
    });
    var tabBody = add(c, el("div", "tabbody"));
    tabBody.id = panelId; tabBody.setAttribute("role", "tabpanel");
    tr.col = { el: c, coord: coord, returns: rets, steps: steps, stepsTitle: stitle, stepsBack: back, tabButtons: buttons, tabBody: tabBody };
  });

  /* goods selector */
  var sel = $("#good"), allGoods = [];
  tracks.forEach(function (tr) { tr.goods.forEach(function (g) { if (allGoods.indexOf(g) < 0) allGoods.push(g); }); });
  allGoods.sort(function (x, y) { return goodL(x).localeCompare(goodL(y), S.locale); });
  allGoods.forEach(function (g) { var o = add(sel, el("option", "", goodL(g))); o.value = g; });
  // Open on the good the shock targets, the one the scenario is about.
  var shockGood = head.shock ? (head.shock.target_resource || head.shock.target_product) : null;
  good = allGoods.indexOf(D.focusGood) >= 0 ? D.focusGood
    : allGoods.indexOf(shockGood) >= 0 ? shockGood : (allGoods[0] || null);
  sel.value = good;
  sel.onchange = function () { good = sel.value; render(); };

  /* phase strip under the scrubber */
  (function () {
    var marks = $("#marks"), st = head.shockTick, rt = head.reversionTick;
    function seg(a, b, cls) {
      var s = add(marks, el("span", ""));
      s.style.left = (a / horizon * 100) + "%"; s.style.width = ((b - a) / horizon * 100) + "%";
      s.style.background = cls;
    }
    if (st !== null) seg(st, rt === null ? horizon : rt, "var(--shock)");
    if (rt !== null) seg(rt, horizon, "var(--recovery)");
    allTaxTicks.forEach(function (t) {
      var m = add(marks, el("i", "tax-mark"));
      m.style.left = (t / horizon * 100) + "%";
      m.title = tpl("replay.wealth_tax_at", { tick: t });
    });
    tracks.forEach(function (tr) {
      if (tr.exhaustionTick === null || tr.exhaustionTick === undefined) return;
      var m = add(marks, el("i", "exhaust-mark sw-" + tr.condition));
      m.style.left = (tr.exhaustionTick / horizon * 100) + "%";
      m.title = tpl("replay.exhausted_at", { tick: tr.exhaustionTick });
    });
    marks.style.background = "var(--rule)";
  })();

  (function () {
    var box = $("#shockbox"), text = shockText(head);
    if (!text) return;
    box.hidden = false;
    add(box, el("strong", "", tpl("replay.shock.title", { scenario: S.scenario[head.shock.scenario] || head.shock.scenario })));
    add(box, el("span", "", text));
  })();

  /* ---------- crafting guide (static reference, not tied to the tick) ---------- */
  var SVGNS = "http://www.w3.org/2000/svg";
  var RES_ORDER = ["iron", "wood", "leather", "herbs", "arcane_crystal"];
  var PROD_ORDER = ["melee_weapon", "armor", "bow_gear", "staff_focus", "health_potion", "mana_potion"];
  var RES_COLOUR = { iron: "#7d8a9a", wood: "#a8703f", leather: "#c68a52", herbs: "#4f9f45", arcane_crystal: "#8b6cf0" };

  function svg(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }
  function icon(goodId, cls) {
    var s = svg("svg", { "class": "gi " + (cls || ""), "aria-hidden": "true" });
    s.appendChild(svg("use", { href: "#g-" + goodId }));
    return s;
  }
  function decimal(v, d) { return num(parseFloat(v), d); }
  function specName(s) { return S.spec[s] || s; }
  function ordered(list, order) {
    return list.slice().sort(function (a, b) {
      var ia = order.indexOf(a), ib = order.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
  }
  function focusGood(g) {
    if (allGoods.indexOf(g) < 0) return;
    good = g; sel.value = g; render();
    $(".overview").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function flowDiagram(R, resources, products) {
    var rowR = 76, rowP = 62, top = 24;
    var h = Math.max(resources.length * rowR, products.length * rowP) + top;
    var W = 820, xr = 30, xp = W - 250;
    var box = svg("svg", { viewBox: "0 0 " + W + " " + h, "class": "flow", role: "img" });
    var yR = {}, yP = {};
    resources.forEach(function (r, i) { yR[r] = top + i * rowR + (h - top - resources.length * rowR) / 2 + 20; });
    products.forEach(function (p, j) { yP[p] = top + j * rowP + (h - top - products.length * rowP) / 2 + 20; });
    var paths = [];
    products.forEach(function (p) {
      var inputs = ordered(Object.keys(R.recipes[p].in), RES_ORDER).filter(function (r) { return r in yR; });
      inputs.forEach(function (r, k) {
        /* Inputs of one product arrive side by side, so their labels never collide. */
        var x1 = xr + 200, y1 = yR[r], x2 = xp - 12, y2 = yP[p] + (k - (inputs.length - 1) / 2) * 14;
        var mx = (x1 + x2) / 2;
        var path = svg("path", { d: "M" + x1 + " " + y1 + " C" + mx + " " + y1 + " " + mx + " " + y2 + " " + x2 + " " + y2,
          stroke: RES_COLOUR[r] || "#888", "class": "edge", "data-r": r, "data-p": p });
        box.appendChild(path);
        /* The quantity sits on its own curve, three quarters of the way along. */
        var s = 0.78, u = 1 - s;
        var lx = u * u * u * x1 + 3 * u * u * s * mx + 3 * u * s * s * mx + s * s * s * x2;
        var ly = u * u * u * y1 + 3 * u * u * s * y1 + 3 * u * s * s * y2 + s * s * s * y2;
        var badge = svg("g", { "class": "qty", "data-r": r, "data-p": p });
        badge.appendChild(svg("rect", { x: lx - 15, y: ly - 10, width: 30, height: 19, rx: 9.5, fill: RES_COLOUR[r] || "#888" }));
        var label = svg("text", { x: lx, y: ly + 4.5, "text-anchor": "middle" });
        label.textContent = "×" + R.recipes[p].in[r];
        badge.appendChild(label);
        box.appendChild(badge);
        paths.push(path, badge);
      });
    });
    function node(g, x, y, isRes) {
      var grp = svg("g", { "class": "node", tabindex: "0" });
      grp.appendChild(svg("use", { href: "#g-" + g, x: x, y: y - 20, width: 40, height: 40 }));
      var txt = svg("text", { x: x + 48, y: y + 5, "class": "name" });
      txt.textContent = goodL(g);
      grp.appendChild(txt);
      function on() {
        box.classList.add("focus");
        paths.forEach(function (e) { e.classList.toggle("on", e.getAttribute(isRes ? "data-r" : "data-p") === g); });
      }
      function off() { box.classList.remove("focus"); paths.forEach(function (e) { e.classList.remove("on"); }); }
      grp.addEventListener("mouseenter", on); grp.addEventListener("focus", on);
      grp.addEventListener("mouseleave", off); grp.addEventListener("blur", off);
      grp.addEventListener("click", function () { focusGood(g); });
      activatable(grp, goodL(g));
      box.appendChild(grp);
    }
    resources.forEach(function (r) { node(r, xr, yR[r], true); });
    products.forEach(function (p) { node(p, xp, yP[p], false); });
    return box;
  }

  /* A card's text, apart from its name, in one element the card can point at. */
  function cardBody(card, id) {
    var body = add(card, el("div", "gbody"));
    body.id = "gdesc-" + id;
    return body;
  }
  function resourceCard(R, r, usedIn) {
    var outer = el("article", "gcard");
    var head = add(outer, el("header", ""));
    head.appendChild(icon(r, "big"));
    add(head, el("h4", "", goodL(r)));
    var card = cardBody(outer, r);
    add(card, el("h5", "", tpl("replay.guide.used_for")));
    var uses = usedIn[r] || [];
    if (!uses.length) add(card, el("p", "muted", tpl("replay.guide.unused")));
    else {
      var ul = add(card, el("ul", "uses"));
      uses.forEach(function (u) {
        var li = add(ul, el("li", ""));
        li.appendChild(icon(u[0], "small"));
        // Per recipe run: one run can make more than one unit.
        add(li, el("span", "", u[2] > 1
          ? tpl("replay.guide.per_batch", { qty: u[1], out: u[2], product: goodL(u[0]) })
          : tpl("replay.guide.per_unit", { qty: u[1], product: goodL(u[0]) })));
      });
    }
    add(card, el("h5", "", tpl("replay.guide.found_in")));
    var zones = R.zones.filter(function (z) { return parseFloat(z.spawn[r] || 0) > 0; })
      .sort(function (a, b) { return parseFloat(b.spawn[r]) - parseFloat(a.spawn[r]); });
    var zl = add(card, el("ul", "plain"));
    zones.forEach(function (z) {
      add(zl, el("li", "", tpl("replay.guide.zone_rate", { zone: z.id, risk: tpl("replay.risk." + z.risk), rate: decimal(z.spawn[r], 1) })));
    });
    var best = Object.keys(R.affinity).filter(function (s) { return parseFloat(R.affinity[s][r] || 1) > 1; });
    if (best.length) {
      add(card, el("h5", "", tpl("replay.guide.best_gatherer")));
      add(card, el("p", "", best.map(function (s) { return specName(s) + " ×" + decimal(R.affinity[s][r], 1); }).join(" · ")));
    }
    outer.addEventListener("click", function () { focusGood(r); });
    activatable(outer, goodL(r), card.id);
    return outer;
  }

})();
