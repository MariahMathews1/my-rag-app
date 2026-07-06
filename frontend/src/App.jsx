// =========================================================================
// App.jsx
//
// A styled chat interface for our drone-knowledge RAG app. Functionally
// identical to the earlier basic version - state, submit handler, fetch
// call - just with real visual design applied on top, and one added
// feature: each AI response shows which document chunks it pulled from,
// so you can visually confirm retrieval is working as you use the app.
// =========================================================================

import { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // A ref lets us grab a direct handle to a DOM element - here, the
  // bottom of the message list - so we can scroll to it automatically
  // whenever a new message arrives.
  const bottomRef = useRef(null);

  // These refs track the "typewriter" effect state. Refs are like state,
  // but updating them does NOT trigger a re-render - useful here because
  // we update them extremely often (many times per second) and only want
  // React to actually re-render on a controlled schedule, not every time.
  const targetTextRef = useRef("");     // the full text received so far
  const displayedLengthRef = useRef(0); // how many characters we've shown
  const typingIntervalRef = useRef(null);
  const streamDoneRef = useRef(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Reveals text a few characters at a time, on a fixed timer, regardless
  // of how fast (or in what size chunks) the network actually delivers
  // text. This is what creates a natural, readable "typing" pace instead
  // of the whole answer appearing in one instant burst.
  function startTypingEffect(assistantMessageIndex, sources) {
    // Clear any previous interval first, just in case one is still running.
    if (typingIntervalRef.current) clearInterval(typingIntervalRef.current);

    typingIntervalRef.current = setInterval(() => {
      const target = targetTextRef.current;
      const currentLength = displayedLengthRef.current;

      if (currentLength < target.length) {
        // Reveal a few more characters. Bumping this number (e.g. to 3-5)
        // makes it type faster; lowering it makes it type slower.
        const charsPerTick = 1;
        displayedLengthRef.current = Math.min(currentLength + charsPerTick, target.length);

        setMessages((prev) => {
          const updated = [...prev];
          updated[assistantMessageIndex] = {
            role: "assistant",
            text: target.slice(0, displayedLengthRef.current),
            sources,
          };
          return updated;
        });
      } else if (streamDoneRef.current) {
        // We've displayed everything AND the network stream is finished -
        // nothing left to do, so stop the timer.
        clearInterval(typingIntervalRef.current);
      }
    }, 20); // one tick every 20ms - adjust this to change overall speed
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!question.trim()) return;

    const userText = question;
    setMessages((prev) => [...prev, { role: "user", text: userText }]);
    setQuestion("");
    setIsLoading(true);

    // Reset the typewriter state for this new message.
    targetTextRef.current = "";
    displayedLengthRef.current = 0;
    streamDoneRef.current = false;

    let assistantMessageIndex;
    setMessages((prev) => {
      assistantMessageIndex = prev.length;
      return [...prev, { role: "assistant", text: "", sources: [] }];
    });

    try {
      const response = await fetch("http://localhost:8000/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userText }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      let sourcesParsed = false;
      let sources = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        if (!sourcesParsed) {
          const separator = "\n---STREAM-START---\n";
          const separatorIndex = buffer.indexOf(separator);

          if (separatorIndex !== -1) {
            const sourcesJson = buffer.slice(0, separatorIndex);
            sources = JSON.parse(sourcesJson).sources;
            sourcesParsed = true;

            // Whatever came after the separator is real answer text -
            // feed it into the typewriter's target and start the effect.
            buffer = buffer.slice(separatorIndex + separator.length);
            targetTextRef.current = buffer;
            startTypingEffect(assistantMessageIndex, sources);
          }
        } else {
          // More answer text arrived - just extend the typewriter's
          // target. We do NOT touch message state here directly; the
          // running interval is the only thing that updates the screen,
          // at its own controlled pace.
          targetTextRef.current = buffer;
        }
      }

      // The network stream is done, but the typewriter may still be
      // catching up on already-received text - let it finish naturally.
      streamDoneRef.current = true;
    } catch (error) {
      if (typingIntervalRef.current) clearInterval(typingIntervalRef.current);
      setMessages((prev) => {
        const updated = [...prev];
        updated[assistantMessageIndex] = {
          role: "assistant",
          text: "Transmission failed: " + error.message,
          sources: [],
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="console">
      {/* Header styled like a ground-control status bar */}
      <header className="console-header">
        <div className="header-left">
          <span className="status-dot" />
          <span className="header-title">DRONE KNOWLEDGE BASE</span>
        </div>
        <span className="header-meta">RAG.SYS // ONLINE</span>
      </header>

      {/* Message log */}
      <main className="log">
        {messages.length === 0 && (
          <div className="empty-state">
            <p className="empty-title">No transmissions yet</p>
            <p className="empty-sub">Ask about flight mechanics, FAA rules, cameras, or maintenance.</p>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`entry entry-${message.role}`}>
            <div className="entry-label">
              {message.role === "user" ? "YOU" : "SYSTEM"}
            </div>
            <div className="entry-bubble">
              <p className="entry-text">{message.text}</p>

              {/* Only assistant messages have retrieved sources to show */}
              {message.role === "assistant" && message.sources?.length > 0 && (
                <details className="sources">
                  <summary>retrieved context ({message.sources.length})</summary>
                  {message.sources.map((source, i) => (
                    <p key={i} className="source-chunk">{source}</p>
                  ))}
                </details>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="entry entry-assistant">
            <div className="entry-label">SYSTEM</div>
            <div className="entry-bubble">
              <p className="entry-text pending">
                <span className="pulse-dot" /> retrieving context and generating response
              </p>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* Input bar */}
      <form className="input-bar" onSubmit={handleSubmit}>
        <span className="prompt-caret">&gt;</span>
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="transmit a question..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          SEND
        </button>
      </form>
    </div>
  );
}

export default App;