
import { useState } from "react";
import "../styles/authorsCitation.css";

const citationText = `@misc{ragen,
  title={RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning}, 
  author={Zihan Wang and Kangrui Wang and Qineng Wang and Pingyue Zhang and Linjie Li and Zhengyuan Yang and Xing Jin and Kefan Yu and Minh Nhat Nguyen and Licheng Liu and Eli Gottlieb and Yiping Lu and Kyunghyun Cho and Jiajun Wu and Li Fei-Fei and Lijuan Wang and Yejin Choi and Manling Li},
  year={2025},
  eprint={2504.20073},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2504.20073}, 
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
            If you find RAGEN useful in your research, we would appreciate it if
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


