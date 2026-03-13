import { useEffect, useState } from "react";
import "../styles/hero.css";
import "../styles/section.css";
import "../styles/global.css";
import "../styles/resources.css";

function Hero() {
  const [stars, setStars] = useState("1.4k");
  const [forks, setForks] = useState("105");

  const animatedText = "Query-Driven Reasoning for multimodal MOT ";
  const motText = "Multiple Object Tracking";

  useEffect(() => {
    async function fetchGitHubStats() {
      try {
        const response = await fetch(
          "https://api.github.com/repos/RAGEN-AI/RAGEN",
        );

        if (!response.ok) {
          throw new Error("GitHub API error");
        }

        const data = await response.json();

        const formatNumber = (num) =>
          num > 999 ? (num / 1000).toFixed(1) + "k" : num;

        if (data.stargazers_count !== undefined) {
          setStars(formatNumber(data.stargazers_count));
        }

        if (data.forks_count !== undefined) {
          setForks(formatNumber(data.forks_count));
        }
      } catch (error) {
        console.error("Error fetching GitHub stats:", error);
      }
    }

    fetchGitHubStats();
  }, []);

  return (
    <section id="hero" className="hero-section">
      <div className="title-wrapper">
        <img src="/icon.png" alt="MOT Icon" className="title-icon" />

        <h1 className="main-title">
          {motText.split("").map((char, index) => (
            <span
              key={index}
              className="reveal-char-title"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {char === " " ? "\u00A0" : char}
            </span>
          ))}
        </h1>
      </div>

      <p className="subtitle">
        {animatedText.split("").map((char, index) => (
          <span
            key={index}
            className="reveal-char"
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            {char === " " ? "\u00A0" : char}
          </span>
        ))}
      </p>

      <p className="description">
        Deep Learning + Computer Vision <br />
        to track multiple objects simultaneously in dynamic video environments.
        <br />
        <a
          href="https://vagen-ai.github.io/"
          className="vagen-announcement"
          target="_blank"
          rel="noopener noreferrer"
        >
          Announcing <b>TrackNet</b> for MOT Systems &gt;
        </a>
      </p>

      {/* GitHub Button */}

      <a
        href="https://github.com/ya-sonia/MOT"
        className="cta-button"
        target="_blank"
        rel="noopener noreferrer"
      >
        <svg height="22" viewBox="0 0 16 16" fill="white" aria-hidden="true">
          <path
            d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 
  0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
  -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 
  2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 
  0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 
  0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 
  1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82 
  .44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 
  0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 
  0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 
  8.013 0 0016 8c0-4.42-3.58-8-8-8z"
          />
        </svg>

        <span className="button-text">Get Started</span>

        <div className="repo-stats">
          <div className="stat-item">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path
                d="M12 17.27L18.18 21 16.54 13.97 
               22 9.24 14.81 8.63 
               12 2 9.19 8.63 
               2 9.24 7.46 13.97 
               5.82 21 12 17.27Z"
              />
            </svg>
            {stars}
          </div>
          <div className="stat-item">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-.878a2.25 2.25 0 1 1 1.5 0v.878a2.25 2.25 0 0 1-2.25 2.25h-1.5v2.128a2.251 2.251 0 1 1-1.5 0V8.5h-1.5A2.25 2.25 0 0 1 3.5 6.25v-.878a2.25 2.25 0 1 1 1.5 0ZM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Zm6.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm-3 8.75a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z" />
            </svg>

            {forks}
          </div>
        </div>
      </a>

      {/* Resources Section */}

      {/* <section className="resources-section">
        <h3 className="resources-title">Resources</h3>
        <div className="resources-buttons">
          <a
            href="https://arxiv.org/abs/2101.02702"
            className="resource-button"
            target="_blank"
            rel="noopener noreferrer"
          >
            <svg viewBox="0 0 24 24" className="resource-icon">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
            <span>Paper</span>
          </a>

          <a
            href="https://ragen-doc.readthedocs.io/"
            className="resource-button"
            target="_blank"
            rel="noopener noreferrer"
          >
            <svg viewBox="0 0 24 24" className="resource-icon">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            <span>Documentation</span>
          </a>

          <a
            href="https://x.com/wzihanw/status/1915052871474712858"
            className="resource-button"
            target="_blank"
            rel="noopener noreferrer"
          >
            <svg viewBox="0 0 24 24" className="resource-icon">
              <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
            </svg>
            <span>tl;dr</span>
          </a>

          <a
            href="https://api.wandb.ai/links/zihanwang-ai-northwestern-university/a8er8l7b"
            className="resource-button"
            target="_blank"
            rel="noopener noreferrer"
          >
            <svg viewBox="0 0 24 24" className="resource-icon">
              <path d="M21 3v18M3 9h18M3 15h18M3 21h18" />
              <path d="M3 3h18" />
              <path d="M9 21V9" />
              <path d="M15 21V9" />
            </svg>
            <span>Logs</span>
          </a>

          <a
            href="https://mll-lab-nu.github.io/joinus/"
            className="resource-button"
            target="_blank"
            rel="noopener noreferrer"
          >
            <svg viewBox="0 0 24 24" className="resource-icon">
              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
            <span>Collaborate with us →</span>
          </a>
        </div>
      </section> */}

      {/* Diagram Section */}
      {/* <div className="diagram-placeholder">
        <img src="/teaser.jpeg" alt="MOT Diagram" />
        <p style={{ textAlign: "center", color: "#888" }}>
          Comparison between traditional MOT and ours work.
        </p>
      </div> */}

      {/* Authors Section */}
      {/* <div className="authors-section">
        <h3 className="section-title">Authors</h3>
        <div className="authors-list">
          <p>
            <a href="https://zihanwang314.github.io/">Zihan Wang</a>
            <sup className="affiliation-marker">*1</sup>,
            <a href="https://jameskrw.github.io/">Kangrui Wang</a>
            <sup className="affiliation-marker">*1</sup>,
            <a href="https://qinengwang-aiden.github.io/">Qineng Wang</a>
            <sup className="affiliation-marker">*1</sup>,
            <a href="https://williamzhangsjtu.github.io/">Pingyue Zhang</a>
            <sup className="affiliation-marker">*1</sup>,
            <a href="https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en">
              Linjie Li
            </a>
            <sup className="affiliation-marker">*2</sup>
          </p>

          <p>
            <a href="https://zyang-ur.github.io/">Zhengyuan Yang</a>
            <sup className="affiliation-marker">4</sup>,<a href="">Xing Jin</a>
            <sup className="affiliation-marker">6</sup>,
            <a href="https://www.linkedin.com/in/kefan-yu-22723a25b/en/">
              Kefan Yu
            </a>
            <sup className="affiliation-marker">1</sup>,
            <a href="https://www.linkedin.com/in/menhguin/?originalSubdomain=sg">
              Minh Nhat Nguyen
            </a>
            <sup className="affiliation-marker">7</sup>,
            <a href="">Licheng Liu</a>
            <sup className="affiliation-marker">1</sup>,
            <a href="https://www.linkedin.com/in/eli-gottlieb1/">
              Eli Gottlieb
            </a>
            <sup className="affiliation-marker">1</sup>,
          </p>

          <p>
            <a href="https://2prime.github.io/">Yiping Lu</a>
            <sup className="affiliation-marker">1</sup>,
            <a href="https://kyunghyuncho.me/">Kyunghyun Cho</a>
            <sup className="affiliation-marker">5</sup>,
            <a href="https://jiajunwu.com/">Jiajun Wu</a>
            <sup className="affiliation-marker">3</sup>,
            <a href="https://profiles.stanford.edu/fei-fei-li">Li Fei-Fei</a>
            <sup className="affiliation-marker">3</sup>,
            <a href="https://www.microsoft.com/en-us/research/people/lijuanw/">
              Lijuan Wang
            </a>
            <sup className="affiliation-marker">4</sup>,
            <a href="https://homes.cs.washington.edu/~yejin/">Yejin Choi</a>
            <sup className="affiliation-marker">3</sup>,
            <a href="https://limanling.github.io/">Manling Li</a>
            <sup className="affiliation-marker">1</sup>
          </p>

          <p className="equal-contribution">
            <sup className="affiliation-marker">*</sup> Equal Contribution
          </p>

          <p className="affiliations">
            <sup className="affiliation-marker">1</sup> Northwestern University
            &nbsp;&nbsp;
            <sup className="affiliation-marker">2</sup> University of Washington
            &nbsp;&nbsp;
            <sup className="affiliation-marker">3</sup> Stanford University
            &nbsp;&nbsp;
            <sup className="affiliation-marker">4</sup> Microsoft &nbsp;&nbsp;
            <br />
            <sup className="affiliation-marker">5</sup> New York University
            &nbsp;&nbsp;
            <sup className="affiliation-marker">6</sup> University of British
            Columbia &nbsp;&nbsp;
            <sup className="affiliation-marker">7</sup> Singapore Management
            University &nbsp;&nbsp;
          </p>
        </div>
      </div> */}
    </section>
  );
}

export default Hero;
