// 画面。数字はすべて API（決定論コア）が出したものをそのまま描く。
// **画面では計算しない。** kL への換算だけは表示の都合でここで行う。

const kl = (l) => (l / 1000).toFixed(1);
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

// API が返す日本語には **強調** が入る。画面では太字にして出す（記号のまま見せない）。
function rich(text) {
  const frag = document.createDocumentFragment();
  String(text).split("**").forEach((part, i) => {
    if (!part) return;
    frag.append(i % 2 ? el("strong", null, part) : document.createTextNode(part));
  });
  return frag;
}
const elRich = (tag, cls, text) => { const n = el(tag, cls); n.append(rich(text)); return n; };

// --- 上部のナビ -------------------------------------------------------------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.removeAttribute("aria-current"));
    tab.setAttribute("aria-current", "true");
    document.querySelectorAll(".panel").forEach((p) => { p.hidden = true; });
    $("#" + tab.dataset.panel).hidden = false;
  });
});

// --- 待たせている間、何を待っているかを出す ---------------------------------
function waiting(node, what) {
  node.hidden = false;
  node.className = "status";
  const started = Date.now();
  node.textContent = `${what}…`;
  const timer = setInterval(() => {
    const s = Math.round((Date.now() - started) / 1000);
    node.textContent = `${what}… ${s}秒`;
  }, 1000);
  return {
    done: (text) => { clearInterval(timer); node.textContent = text; },
    failed: (text) => { clearInterval(timer); node.className = "status error"; node.textContent = text; },
  };
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

// 画面に出ている計画を組んだときの条件。差し込みも持ち戻りも、**この計画の上で**
// 起きる話なので、あとで上のフォームをいじっても、組んだときの条件のまま送る。
let planBasis = null;

function planParams(form) {
  return {
    seed: Number(form.seed.value),
    date: form.date ? form.date.value : $("#plan-form").date.value,
    season: form.season ? form.season.value : $("#plan-form").season.value,
    mode: form.mode ? form.mode.value : $("#plan-form").mode.value,
  };
}

// --- 今日の配車 -------------------------------------------------------------
// 持ち戻りは「いまの計画の立ち寄り」に対して起きるので、計画から選ばせる。
// ID を手で打たせると、計画に無い組み合わせを選べてしまう。
function fillReturnStops(days) {
  const sel = $("[name=stop]");
  sel.replaceChildren();
  let count = 0;
  for (const vd of days) {
    for (const tr of vd.trips) {
      for (const s of tr.stops) {
        const opt = el("option", null,
          `${vd.vehicle_name.split("（")[0]} — ${s.customer_name}（${s.arrive}着 ${kl(s.delivered_l)}kL）`);
        opt.value = `${vd.vehicle_id}|${s.customer_id}`;
        sel.append(opt);
        count += 1;
      }
    }
  }
  if (!count) sel.append(el("option", null, "計画に立ち寄りがありません"));
}

function drawVehicles(days) {
  const box = $("#plan-vehicles");
  box.replaceChildren();
  for (const vd of days) {
    const card = el("div", "vehicle");
    card.append(el("h3", null, `${vd.vehicle_name}（${vd.driver_name}）`));
    card.append(el("div", "duty",
      `拘束 ${vd.duty_start}〜${vd.duty_end}（${Math.floor(vd.duty_min / 60)}時間${vd.duty_min % 60}分）`
      + `／走行 ${Math.floor(vd.drive_min / 60)}時間${vd.drive_min % 60}分`
      + (vd.break_min ? `／休憩 ${vd.break_min}分` : "")
      + `／${vd.trips.length}便で ${kl(vd.load_l)}kL`));

    for (const tr of vd.trips) {
      card.append(el("div", "trip-head",
        `‹${tr.trip_index + 1}便› 積込 ${tr.load_start}／出庫 ${tr.depart}／帰庫 ${tr["return"]}`
        + `　積載 ${kl(tr.load_l)} / ${kl(tr.capacity_l)}kL`));
      const table = el("table");
      const head = el("tr");
      ["着", "降ろす先", "量", "受入枠", "余裕"].forEach((h) => head.append(el("th", null, h)));
      table.append(head);
      for (const s of tr.stops) {
        const row = el("tr");
        row.append(el("td", null, s.arrive));
        row.append(el("td", null, s.customer_name));
        row.append(el("td", "num", `${kl(s.delivered_l)}kL`));
        row.append(el("td", null, s.window));
        row.append(el("td", "num", `${s.slack_min}分`));
        table.append(row);
      }
      card.append(table);
    }
    box.append(card);
  }
}

$("#plan-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const wait = waiting($("#plan-status"), "計画を組んでいます");
  form.querySelector(".go").disabled = true;
  $("#plan-summary").hidden = true;
  $("#plan-unserved").hidden = true;
  $("#plan-vehicles").replaceChildren();
  try {
    const basis = planParams(form);
    const body = await post("/api/plan", basis);
    planBasis = basis;
    wait.done(`${body.status === "OPTIMAL" ? "最適な計画が出ました" : "計画が出ました"}`
      + `（求解 ${body.wall_time_s}秒）`);

    const sum = $("#plan-summary");
    sum.replaceChildren(el("strong", null, body.summary_ja));
    sum.append(el("div", "meta",
      `走行 ${Math.floor(body.total_drive_min / 60)}時間${body.total_drive_min % 60}分`
      + `／${body.diagnostics.solve_mode === "reproducible" ? "再現モード" : "即応モード"}`
      + `／同じ入力で同じ計画が返るか: ${body.diagnostics.reproducible ? "はい" : "いいえ（時間で打ち切った）"}`));
    sum.hidden = false;

    drawVehicles(body.vehicle_days);
    fillReturnStops(body.vehicle_days);

    if (body.unserved.length) {
      const box = $("#plan-unserved");
      box.replaceChildren(el("h3", null, "今日は回れない先"));
      const ul = el("ul", "lines");
      for (const u of body.unserved) {
        ul.append(el("li", u.mandatory ? "check" : null,
          `${u.customer_name}　必要 ${kl(u.demand_l)}kL`
          + (u.days_to_empty !== null ? `／在庫あと ${u.days_to_empty}日` : "")
          + (u.mandatory ? "　【要手配】" : "")));
      }
      box.append(ul);
      box.hidden = false;
    }
  } catch (err) {
    wait.failed(`組めませんでした: ${err.message}`);
  } finally {
    form.querySelector(".go").disabled = false;
  }
});

// --- 当日の差し込み ---------------------------------------------------------
async function fileToBase64(file) {
  const buf = new Uint8Array(await file.arrayBuffer());
  let bin = "";
  for (const b of buf) bin += String.fromCharCode(b);
  return btoa(bin);
}

function drawReplan(r) {
  const box = $("#intake-result");
  box.replaceChildren(el("h3", null, "組み直し"));
  box.append(elRich("p", null, r.summary_ja));

  if (r.shortfalls && r.shortfalls.length) {
    box.append(el("h3", null, "確認が要るもの"));
    const ul = el("ul", "lines");
    r.shortfalls.forEach((x) => ul.append(elRich("li", "check", x)));
    box.append(ul);
  }
  if (r.changes && r.changes.length) {
    box.append(el("h3", null, "変わるところ"));
    const ul = el("ul", "lines");
    r.changes.forEach((x) => ul.append(elRich("li", null, x)));
    box.append(ul);
  }
  if (r.escapes && r.escapes.length) {
    box.append(el("h3", null, "応えられなかった先"));
    const ul = el("ul", "lines");
    r.escapes.forEach((e) => ul.append(el("li", "check", `${e.customer_name}　${kl(e.volume_l)}kL`)));
    box.append(ul);
  }
  if (r.sold_more_now && r.sold_more_now.length) {
    box.append(el("div", "kv", "当てた事象: " + r.sold_more_now.join("、")));
  }
  box.hidden = false;
}

$("#intake-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const wait = waiting($("#intake-status"), "FAXを読み取って組み直しています");
  form.querySelector(".go").disabled = true;
  $("#intake-read").hidden = true;
  $("#intake-result").hidden = true;
  try {
    let image;
    if (form.sample.checked || !form.fax.files.length) {
      const res = await fetch("/demo/fax_order.png");
      if (!res.ok) throw new Error("サンプルのFAXが見つかりません");
      image = await fileToBase64(await res.blob());
    } else {
      image = await fileToBase64(form.fax.files[0]);
    }

    const ev_on = $("[name=ev-on]").checked;
    const body = {
      ...(planBasis || planParams($("#plan-form"))),
      now: form.now.value,
      image_base64: image,
      mime: "image/png",
      spot_standby: Number(form.spot.value),
      sold_more_now: ev_on ? [{
        customer_id: $("[name=ev-customer]").value.trim(),
        extra_days: Number($("[name=ev-days]").value),
        ...($("[name=ev-product]").value ? { product: $("[name=ev-product]").value } : {}),
      }] : [],
    };
    const data = await post("/api/intake/fax", body);
    wait.done(`読み取りと組み直しが終わりました（モデル費用 ${(data.model_usd * 150).toFixed(2)}円ほど）`);

    const read = $("#intake-read");
    read.replaceChildren(el("h3", null, "FAXから読み取った発注"));
    if (data.orders.length) {
      const ul = el("ul", "lines");
      const ja = { regular: "レギュラー", premium: "ハイオク", gasoil: "軽油", kerosene: "灯油" };
      data.orders.forEach((o) => ul.append(el("li", "ok",
        `${o.customer_id}　${ja[o.product] || o.product}　${kl(o.volume_l)}kL`
        + `（${o.source === "fax" ? "FAX" : o.source}・証拠 ${o.evidence_ref}）`)));
      read.append(ul);
    }
    if (data.issues && data.issues.length) {
      read.append(el("h3", null, "読み取れなかったもの"));
      const ul = el("ul", "lines");
      data.issues.forEach((x) => ul.append(elRich("li", "check", x)));
      read.append(ul);
    }
    if (data.action === "confirm") {
      read.append(el("p", "kv", "確認が要るものがあるので、組み直していません。"));
    }
    if (data.notice) {
      read.append(el("p", "kv", "電話などの連絡は予告として扱い、車は動かしません（試算のみ）。"));
    }
    read.hidden = false;

    if (data.replan) drawReplan(data.replan);
  } catch (err) {
    wait.failed(`読み取れませんでした: ${err.message}`);
  } finally {
    form.querySelector(".go").disabled = false;
  }
});

// --- 持ち戻り ---------------------------------------------------------------
$("#ret-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const wait = waiting($("#ret-status"), "手を探しています");
  form.querySelector(".go").disabled = true;
  $("#ret-result").hidden = true;
  try {
    const [vehicle_id, customer_id] = (form.stop.value || "").split("|");
    if (!vehicle_id) throw new Error("先に「今日の配車」で計画を組んでください");
    const data = await post("/api/return", {
      ...(planBasis || planParams($("#plan-form"))),
      vehicle_id,
      customer_id,
      product: form.product.value,
      volume_l: Number(form.volume.value),
    });
    wait.done("手が出ました");

    const box = $("#ret-result");
    box.replaceChildren(el("h3", null, "持ち戻りへの手"));
    box.append(elRich("p", null, data.summary_ja));
    box.append(elRich("p", "kv", data.responsibility_ja));

    const table = el("table");
    const head = el("tr");
    ["", "手", "行き先", "着", "寄り道", "理由"].forEach((h) => head.append(el("th", null, h)));
    table.append(head);
    const label = { redeliver: "近くの降ろし先", cushion: "クッション", buyback: "積場へ持ち戻り" };
    for (const o of data.options) {
      const row = el("tr");
      const chosen = o.kind === data.chosen.kind && o.destination === data.chosen.destination;
      row.append(el("td", null, chosen ? "◎" : (o.feasible ? "○" : "×")));
      row.append(el("td", null, label[o.kind] || o.kind));
      row.append(el("td", null, o.destination));
      row.append(el("td", null, chosen ? (data.chosen.arrive || "") : ""));
      row.append(el("td", "num", o.extra_min ? `+${o.extra_min}分` : ""));
      row.append(elRich("td", null, o.reason));
      table.append(row);
    }
    box.append(table);
    if (data.chosen.handoff && data.chosen.handoff.length) {
      box.append(el("p", "kv", "他のローリーで持つ先: " + data.chosen.handoff.join("、")));
    }
    box.hidden = false;
  } catch (err) {
    wait.failed(`手を出せませんでした: ${err.message}`);
  } finally {
    form.querySelector(".go").disabled = false;
  }
});
