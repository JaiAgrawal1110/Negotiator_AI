import { useState } from "react";
import DealGauge from "./DealGauge.jsx";

function fmtMoney(n) {
  if (n === null || n === undefined) return "—";
  return `$${Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function ActionLabel(name) {
  return (name || "").replace(/_/g, " ");
}

export default function SessionView({
  clientId,
  state,
  clientHistory,
  transcript,
  onStep,
  onEnd,
  stepLoading,
  stepError,
  ended,
  endSummary,
}) {
  const [clientMessage, setClientMessage] = useState("");
  const [clientOffer, setClientOffer] = useState("");
  const [nVariants, setNVariants] = useState(1);
  const [finalDeal, setFinalDeal] = useState("");
  const [showEndForm, setShowEndForm] = useState(false);

  const submitStep = (e) => {
    e.preventDefault();
    if (!clientMessage.trim()) return;
    onStep({
      client_message: clientMessage.trim(),
      client_offer: clientOffer === "" ? null : Number(clientOffer),
      n_variants: Number(nVariants),
    });
    setClientMessage("");
    setClientOffer("");
  };

  const submitEnd = (e) => {
    e.preventDefault();
    onEnd({ final_deal: finalDeal === "" ? null : Number(finalDeal) });
  };

  return (
    <>
      <div className="panel">
        <div className="panel-title">
          <span className="eyebrow">02</span> Case file — {clientId}
        </div>

        <div className="stat-row">
          <div className="stat">
            <div className="stat-label">Archetype</div>
            <div className="stat-value">{state.archetype}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Turn</div>
            <div className="stat-value">
              {state.turn} / {state.max_turns}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Relationship</div>
            <div className="stat-value">{state.relationship_score}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Status</div>
            <div className="stat-value">{state.done ? "closed" : "active"}</div>
          </div>
        </div>

        <DealGauge floor={state.floor} target={state.target} currentOffer={state.current_offer} />

        {clientHistory ? (
          <div className="history-strip" style={{ marginTop: 20 }}>
            <div className="history-item">
              <span className="k">Past negotiations</span>
              {clientHistory.past_negotiation_count}
            </div>
            <div className="history-item">
              <span className="k">Closed / lost</span>
              {clientHistory.deals_closed} / {clientHistory.deals_lost}
            </div>
            <div className="history-item">
              <span className="k">Avg % of target</span>
              {clientHistory.avg_pct_of_target_on_closed_deals ?? "—"}%
            </div>
            <div className="history-item">
              <span className="k">Last outcome</span>
              {clientHistory.last_outcome}
            </div>
          </div>
        ) : (
          <div className="empty-note" style={{ marginTop: 18 }}>
            No prior history — this is a new client.
          </div>
        )}
      </div>

      <div className="panel">
        <div className="panel-title">
          <span className="eyebrow">03</span> Transcript
        </div>

        {transcript.length === 0 && (
          <div className="empty-note">
            Log the client's first message below to get your opening script.
          </div>
        )}

        {transcript.map((t) => (
          <div className="turn" key={t.turnNumber}>
            <div className="turn-number">TURN {t.turnNumber}</div>
            <div className="bubble bubble-client">
              <div className="bubble-who">Client{t.clientOffer != null ? ` · ${fmtMoney(t.clientOffer)}` : ""}</div>
              {t.clientMessage}
            </div>
            <div className="bubble bubble-agent">
              <span className="action-stamp">{ActionLabel(t.actionName)}</span>
              <div className="bubble-who">Your script</div>
              {t.scripts.map((s, i) => (
                <div className="variant" key={i}>
                  {t.scripts.length > 1 && <div className="variant-tag">variant {i + 1}</div>}
                  {s}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {!ended && (
        <div className="panel">
          <div className="panel-title">
            <span className="eyebrow">04</span> Log client's next move
          </div>

          {stepError && <div className="error-box">{stepError}</div>}

          <form onSubmit={submitStep}>
            <div className="field">
              <label htmlFor="client_message">What the client said, verbatim</label>
              <textarea
                id="client_message"
                rows={2}
                value={clientMessage}
                onChange={(e) => setClientMessage(e.target.value)}
                placeholder="Ok fine, I could maybe do 1500."
                required
              />
            </div>
            <div className="field-row">
              <div className="field">
                <label htmlFor="client_offer">New offer, if any ($)</label>
                <input
                  id="client_offer"
                  type="number"
                  placeholder="leave blank if unchanged"
                  value={clientOffer}
                  onChange={(e) => setClientOffer(e.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="n_variants">Script variants</label>
                <select
                  id="n_variants"
                  value={nVariants}
                  onChange={(e) => setNVariants(e.target.value)}
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                </select>
              </div>
            </div>
            <button className="btn btn-primary btn-block" type="submit" disabled={stepLoading}>
              {stepLoading ? "Thinking…" : "Get next script"}
            </button>
          </form>

          <div style={{ marginTop: 18, paddingTop: 18, borderTop: "1px solid var(--border-soft)" }}>
            {!showEndForm ? (
              <button className="btn btn-danger btn-block" onClick={() => setShowEndForm(true)}>
                End negotiation
              </button>
            ) : (
              <form onSubmit={submitEnd}>
                <div className="field">
                  <label htmlFor="final_deal">Final agreed price ($, leave blank if no deal)</label>
                  <input
                    id="final_deal"
                    type="number"
                    value={finalDeal}
                    onChange={(e) => setFinalDeal(e.target.value)}
                    placeholder="2100"
                  />
                </div>
                <button className="btn btn-danger btn-block" type="submit">
                  Confirm & save to memory
                </button>
              </form>
            )}
          </div>
        </div>
      )}

      {ended && endSummary && (
        <div className="panel">
          <div className="panel-title">
            <span className="eyebrow">05</span> Case closed
          </div>
          <p className="small muted">
            {endSummary.deal_closed
              ? `Deal saved at ${fmtMoney(endSummary.final_deal)} for client "${endSummary.client_id}".`
              : `No deal reached. Outcome saved for client "${endSummary.client_id}".`}
          </p>
        </div>
      )}
    </>
  );
}
