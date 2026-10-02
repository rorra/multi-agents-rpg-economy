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

})();
