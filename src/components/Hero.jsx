import { useEffect, useState } from "react";
import "../styles/hero.css";
import "../styles/section.css";
import "../styles/global.css";
import "../styles/resources.css";

function Hero() {
  const [stars, setStars] = useState("0");
  const [forks, setForks] = useState("0");

  const animatedText = "Query-Driven Reasoning for multimodal MOT";
  const motText = "QTrack";

  useEffect(() => {
    async function fetchGitHubStats() {
      try {
        const response = await fetch(
          "https://api.github.com/repos/gaash-lab/QTrack"
        );

        const data = await response.json();

        if (data.stargazers_count !== undefined) {
          setStars(data.stargazers_count);
        }

        if (data.forks_count !== undefined) {
          setForks(data.forks_count);
        }

      } catch (error) {
        console.error("GitHub fetch error:", error);
      }
    }

    fetchGitHubStats();
  }, []);

  return (
    <section id="hero" className="hero-section">
      <div className="title-wrapper">
        <img src="QTrack-logo.png" alt="MOT Icon" className="title-icon" />

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
        <strong>TLDR; </strong> This work introduces a query-driven tracking paradigm that formulates tracking as a spatiotemporal reasoning problem conditioned on natural language queries, and presents QTrack, an end-to-end vision-language model that integrates multimodal reasoning with tracking-oriented localization.
      </p>


      {/* Authors Section */}
      <div className="authors-section">
        <h3 className="section-title">Authors</h3>
        <div className="authors-list">
          <p>
            <a href="https://www.tajamulashraf.com/">Tajamul Ashraf</a>
            <sup className="affiliation-marker">1,4*</sup>,
            <a href="https://tavaheed.netlify.app/">Tavaheed Tariq</a>
            <sup className="affiliation-marker">4†</sup>,
            <a href="https://sonia-yadav.netlify.app/">Sonia Yadav</a>
            <sup className="affiliation-marker">4†</sup>,
            <a href="https://abrarulriyaz.vercel.app/">Abrar Ul Riyaz</a>
            <sup className="affiliation-marker">4†</sup>,
            <a href="">Wasif Tak</a>
            <sup className="affiliation-marker">2</sup>,
            <a href="https://scholar.google.com/citations?user=PwgggdIAAAAJ&hl=en">Moloud Abdar</a>
            <sup className="affiliation-marker">3</sup>,
            <a href="https://www.janibbashir.com/">Janibul Bashir</a>
            <sup className="affiliation-marker">4</sup>
          </p>

          <p className="equal-contribution">
            <sup className="affiliation-marker">*</sup> Corresponding author
            &nbsp;&nbsp;
            <sup className="affiliation-marker">†</sup> Equal Contribution
          </p>

          <p className="affiliations">
            <sup className="affiliation-marker">1</sup> King Abdullah University of Science and Technology (KAUST), Saudi Arabia
            &nbsp;&nbsp;
            <br />
            <sup className="affiliation-marker">2</sup> Thapar Institute of Engineering and Technology, India
            &nbsp;&nbsp;
            <br />
            <sup className="affiliation-marker">3</sup> The University of Queensland, Australia
            &nbsp;&nbsp;
            <br />
            <sup className="affiliation-marker">4</sup> Gaash Research Lab, National Institute of Technology Srinagar, India
          </p>
        </div>
      </div>

      {/* GitHub Button */}

      <a
        href="https://github.com/gaash-lab/QTrack"
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

      <section className="resources-section">
        <h3 className="resources-title">Resources</h3>
        <div className="resources-buttons">
          <a
            href="https://arxiv.org/abs/2603.13759"
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
            <span>Arxiv Paper</span>
          </a>

          <a
            href="https://github.com/gaash-lab/QTrack"
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
            <span>Github Repo</span>
          </a>


          <a
            href="https://huggingface.co/datasets/GAASH-Lab/RMOT26"
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
            <span>HuggingFace Data</span>
          </a>


          <a
            href="https://gaash.nitsri.ac.in/"
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
      </section>

      

      {/* Diagram Section */}
      <div className="diagram-placeholder">
        <img src="teasor.jpg" alt="MOT Diagram" />
        <p style={{ textAlign: "center", color: "#888" }}>
          <strong>Comparison of tracking paradigms.</strong> (a) Traditional MOT follows a tracking-by-detection paradigm, tracking all objects from predefined categories regardless of user intent. (b) QTrack enables reasoning-aware, query-conditioned tracking: given a video and natural language query, it selectively identifies and tracks only the specified targets, shifting from all-object tracking to semantic, user-driven tracking.
        </p>
      </div>

    </section>
  );
}

export default Hero;
