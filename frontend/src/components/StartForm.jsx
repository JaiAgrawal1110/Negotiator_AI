import { useState } from "react";

const ARCHETYPES = [
  { value: "lowballer", label: "Lowballer" },
  { value: "ghoster", label: "Ghoster" },
  { value: "friendly_crusher", label: "Friendly Crusher" },
  { value: "deadline_rusher", label: "Deadline Rusher" },
  { value: "scope_creeper", label: "Scope Creeper" },
];

const initial = {
  client_id: "",
  archetype: "lowballer",
  floor: "",
  target: "",
  current_offer: "",
  project_description: "",
  project_category: "",
  max_turns: 8,
};

export default function StartForm({ onStart, loading, error }) {
  const [form, setForm] = useState(initial);
  const [advanced, setAdvanced] = useState({ leverage_score: 0.5, deadline_urgency: 0.5 });

  const set = (key) => (e) =>
    setForm((f) => ({ ...f, [key]: e.target.value }));

  const submit = (e) => {
    e.preventDefault();
    onStart({
      client_id: form.client_id.trim(),
      archetype: form.archetype,
      floor: Number(form.floor),
      target: Number(form.target),
      current_offer: Number(form.current_offer),
      project_description: form.project_description.trim(),
      project_category: form.project_category.trim() || null,
      max_turns: Number(form.max_turns) || 8,
      leverage_score: Number(advanced.leverage_score),
      deadline_urgency: Number(advanced.deadline_urgency),
    });
  };

  const valid =
    form.client_id.trim() &&
    form.floor !== "" &&
    form.target !== "" &&
    form.current_offer !== "" &&
    form.project_description.trim() &&
    Number(form.target) > Number(form.floor);

  return (
    <form className="panel" onSubmit={submit}>
      <div className="panel-title">
        <span className="eyebrow">01</span> Open a case
      </div>

      {error && <div className="error-box">{error}</div>}

      <div className="field-row">
        <div className="field">
          <label htmlFor="client_id">Client ID</label>
          <input
            id="client_id"
            placeholder="e.g. acme-corp-jane"
            value={form.client_id}
            onChange={set("client_id")}
            required
          />
        </div>
        <div className="field">
          <label htmlFor="archetype">Archetype</label>
          <select id="archetype" value={form.archetype} onChange={set("archetype")}>
            {ARCHETYPES.map((a) => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="field-row">
        <div className="field">
          <label htmlFor="floor">Your floor ($)</label>
          <input
            id="floor"
            type="number"
            min="0"
            placeholder="1800"
            value={form.floor}
            onChange={set("floor")}
            required
          />
        </div>
        <div className="field">
          <label htmlFor="target">Your target ($)</label>
          <input
            id="target"
            type="number"
            min="0"
            placeholder="2500"
            value={form.target}
            onChange={set("target")}
            required
          />
        </div>
      </div>

      <div className="field-row">
        <div className="field">
          <label htmlFor="current_offer">Client's opening offer ($)</label>
          <input
            id="current_offer"
            type="number"
            min="0"
            placeholder="1200"
            value={form.current_offer}
            onChange={set("current_offer")}
            required
          />
        </div>
        <div className="field">
          <label htmlFor="max_turns">Max turns</label>
          <input
            id="max_turns"
            type="number"
            min="1"
            max="20"
            value={form.max_turns}
            onChange={set("max_turns")}
          />
        </div>
      </div>

      <div className="field">
        <label htmlFor="project_description">Project description</label>
        <textarea
          id="project_description"
          rows={2}
          placeholder="a 3-week backend API build for a scheduling app"
          value={form.project_description}
          onChange={set("project_description")}
          required
        />
      </div>

      <div className="field">
        <label htmlFor="project_category">Project category (optional)</label>
        <input
          id="project_category"
          placeholder="Web Development"
          value={form.project_category}
          onChange={set("project_category")}
        />
      </div>

      <details>
        <summary>Advanced — leverage &amp; urgency</summary>
        <div className="field-row">
          <div className="field">
            <label htmlFor="leverage_score">Your leverage (0–1)</label>
            <input
              id="leverage_score"
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={advanced.leverage_score}
              onChange={(e) =>
                setAdvanced((a) => ({ ...a, leverage_score: e.target.value }))
              }
            />
          </div>
          <div className="field">
            <label htmlFor="deadline_urgency">Client's deadline urgency (0–1)</label>
            <input
              id="deadline_urgency"
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={advanced.deadline_urgency}
              onChange={(e) =>
                setAdvanced((a) => ({ ...a, deadline_urgency: e.target.value }))
              }
            />
          </div>
        </div>
      </details>

      <button className="btn btn-primary btn-block" type="submit" disabled={!valid || loading}>
        {loading ? "Opening case…" : "Open case & get first script"}
      </button>
    </form>
  );
}
