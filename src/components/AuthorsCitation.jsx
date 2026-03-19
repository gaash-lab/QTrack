
import { useState } from "react";
import "../styles/authorsCitation.css";

const citationText = `@inproceedings{Ashraf2026QTrackQR,
  title={QTrack: Query-Driven Reasoning for Multi-modal MOT},
  author={Tajamul Ashraf and Tavaheed Tariq and Sonia Yadav and Abrar Ul Riyaz and Wasif Tak and Moloud Abdar and Janibul Bashir},
  year={2026},
  url={https://api.semanticscholar.org/CorpusID:286568599}
}`;

function AuthorsCitation() {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(citationText);
      setCopied(true);

      setTimeout(() => {
        setCopied(false);
      }, 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
      setError(true);

      setTimeout(() => {
        setError(false);
      }, 2000);
    }
  };

  return (
    <section id="authors-citation" className="content-section">
      <div className="section-container">
        <h2 className="section-title">Citation</h2>

        <div className="citation-section">
          <p>
            If you find QTrack useful in your research, we would appreciate it if
            you consider citing our work:
          </p>

          <div className="citation-block">
            <div className="code-container">

              <button
                className={`copy-button-corner ${
                  copied ? "copied" : error ? "copy-error" : ""
                }`}
                onClick={handleCopy}
                aria-label="Copy citation"
              >
                {copied ? (
                  <svg
                    className="check-icon"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path d="M9 16.2l-3.5-3.5-1.4 1.4L9 19 20.3 7.7l-1.4-1.4z" />
                  </svg>
                ) : (
                  <svg
                    className="copy-icon"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path d="M16 1H4c-1.1 0-2 .9-2 2v12h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14h13c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z" />
                  </svg>
                )}
              </button>

              <pre>
                <code>{citationText}</code>
              </pre>

            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default AuthorsCitation;


