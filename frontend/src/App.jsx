import { useState } from "react";
import { api } from "./api.js";
import StartForm from "./components/StartForm.jsx";
import SessionView from "./components/SessionView.jsx";

export default function App() {
  const [session, setSession] = useState(null); // { sessionId, clientId, state, clientHistory }
  const [transcript, setTranscript] = useState([]);
  const [ended, setEnded] = useState(false);
  const [endSummary, setEndSummary] = useState(null);

  const [startLoading, setStartLoading] = useState(false);
  const [startError, setStartError] = useState(null);

  const [stepLoading, setStepLoading] = useState(false);
  const [stepError, setStepError] = useState(null);

  const handleStart = async (payload) => {
    setStartLoading(true);
    setStartError(null);
    try {
      const res = await api.startNegotiation(payload);
      setSession({
        sessionId: res.session_id,
        clientId: payload.client_id,
        state: res.state,
        clientHistory: res.client_history,
      });
      setTranscript([]);
      setEnded(false);
      setEndSummary(null);
    } catch (err) {
      setStartError(err.message);
    } finally {
      setStartLoading(false);
    }
  };

  const handleStep = async (payload) => {
    if (!session) return;
    setStepLoading(true);
    setStepError(null);
    try {
      const res = await api.step(session.sessionId, payload);
      setTranscript((t) => [
        ...t,
        {
          turnNumber: res.turn,
          clientMessage: payload.client_message,
          clientOffer: payload.client_offer,
          actionName: res.action_name,
          scripts: res.scripts,
        },
      ]);
      setSession((s) => ({ ...s, state: res.state }));
      if (res.state.done) setEnded(true);
    } catch (err) {
      setStepError(err.message);
    } finally {
      setStepLoading(false);
    }
  };

  const handleEnd = async (payload) => {
    if (!session) return;
    setStepLoading(true);
    setStepError(null);
    try {
      const res = await api.end(session.sessionId, payload);
      setEndSummary(res);
      setEnded(true);
    } catch (err) {
      setStepError(err.message);
    } finally {
      setStepLoading(false);
    }
  };

  const startNewCase = () => {
    setSession(null);
    setTranscript([]);
    setEnded(false);
    setEndSummary(null);
    setStartError(null);
    setStepError(null);
  };

  return (
    <div className="shell">
      <div className="masthead">
        <div className="masthead-title">
          <span className="mark">Negotiator</span> — Case File
        </div>
        <div className="masthead-sub">
          {session ? `Session ${session.sessionId.slice(0, 8)}` : "No active case"}
        </div>
      </div>

      {!session && (
        <StartForm onStart={handleStart} loading={startLoading} error={startError} />
      )}

      {session && (
        <>
          <SessionView
            clientId={session.clientId}
            state={session.state}
            clientHistory={session.clientHistory}
            transcript={transcript}
            onStep={handleStep}
            onEnd={handleEnd}
            stepLoading={stepLoading}
            stepError={stepError}
            ended={ended}
            endSummary={endSummary}
          />
          {ended && (
            <button className="btn btn-block" style={{ marginTop: 20 }} onClick={startNewCase}>
              Open a new case
            </button>
          )}
        </>
      )}
    </div>
  );
}
