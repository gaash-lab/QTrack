import { useEffect, useState } from 'react';
import '../styles/algorithm.css';

function Algorithm() {
  return (
    <section id="algorithm">
      <div className="section-container">

        {/* Section Title */}
        <h2 className="section-title">Architecture</h2>

        {/* Architecture Figure */}
        <div className="algorithm-main-image">
          <img
            src="RMOT.jpg"
            alt="QTrack Framework Architecture"
            className="main-algorithm-img"
          />
          <p className="image-caption">
            <strong>Fig. 2:</strong> Overview of the QTrack framework. Given a video sequence
            and a natural language query, QTrack generates a chain-of-thought reasoning trace
            and directly predicts bounding box trajectories for queried targets across all frames,
            performing joint spatial grounding and temporal association.
          </p>
        </div>

        {/* TAPO Algorithm Block */}
        <div className="algorithm-components">
          <div className="component-card">
            <div className="component-header">
              <h3 className="component-title">
                Temporal Perception-Aware Policy Optimization (TAPO)
              </h3>
            </div>
            <div className="component-content active">

              <p style={{ textAlign: "center", color: "#494949" }}>
                Standard GRPO improves spatial reasoning but tends to over-rely on static
                appearance cues, ignoring motion dynamics and causing identity drift under
                occlusion. TAPO introduces a temporal corruption signal  a frozen-frame
                version of the input  and penalises the policy via KL divergence when its
                outputs remain invariant to motion. The final objective combines both terms:
                <strong> J<sub>TAPO</sub> = J<sub>GRPO</sub> + Y · L<sub>track</sub></strong>.
              </p>

              {/* Algorithm Box */}
              <div className="phase-card" style={{ marginTop: '1.5rem' }}>
                <h4>Algorithm 1: TAPO</h4>
                <div className="phase-content">
                  <div className="code-snippet">
                    <code>
                      <span style={{ color: '#a0c4ff' }}>Input:</span> Video F = (F₀, …, F<sub>T</sub>), query q, policy π<sub>θ</sub>, weight γ<br />
                      <br />
                      1: Generate rollout  <span style={{ color: '#90ee90' }}>o ~ π<sub>θ</sub>(o | q, F)</span><br />
                      2: Construct corrupted sequence  <span style={{ color: '#ffd700' }}>F<sup>mask</sup></span>  by freezing frames:  F̃<sub>t</sub> = F₀<br />
                      3: Compute log-probabilities:<br />
                      &nbsp;&nbsp;&nbsp;&nbsp;log p      = log π<sub>θ</sub>(o | q, F)<br />
                      &nbsp;&nbsp;&nbsp;&nbsp;log p<sup>mask</sup> = log π<sub>θ</sub>(o | q, F<sup>mask</sup>)<br />
                      4: Compute temporal loss:<br />
                      &nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: '#ffd700' }}>L<sub>track</sub> = log p − log p<sup>mask</sup></span><br />
                      5: Combine objectives:<br />
                      &nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: '#90ee90' }}>J = J<sub>GRPO</sub> + γ · L<sub>track</sub></span><br />
                      6: Update θ using J
                    </code>
                  </div>
                </div>
              </div>

            </div>
          </div>
        </div>

      </div>
    </section>
  );
}

export default Algorithm;