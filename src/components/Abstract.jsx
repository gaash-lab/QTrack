import '../styles/abstract.css';

const keywords = [
  'Multi-Object Tracking',
  'Vision-Language Models',
  'Spatiotemporal Reasoning',
  'Reinforcement Learning',
];

function Abstract() {
  return (
    <section id="abstract">
      <div className="section-container">

        <h2 className="section-title">Abstract</h2>

        <div className="abstract-card">

          {/* Decorative bracket left */}
          <div className="abstract-bracket left" aria-hidden="true">
            <span /><span /><span />
          </div>

          <div className="abstract-body">

            {/* Highlighted lead sentence */}
            <p className="abstract-lead">
              We introduce a <em>query-driven tracking paradigm</em> that formulates
              multi-object tracking as a spatiotemporal reasoning problem conditioned
              on natural language.
            </p>

            {/* Full text split into readable paragraphs */}
            <p className="abstract-text">
              Multi-object tracking (MOT) has traditionally focused on estimating
              trajectories of all objects in a video, without selectively reasoning
              about user-specified targets under semantic instructions. Given a
              reference frame, a video sequence, and a textual query, the goal is to
              localize and track only the target(s) specified in the query while
              maintaining temporal coherence and identity consistency.
            </p>

            <p className="abstract-text">
              To support this setting, we construct{' '}
              <strong className="highlight-term">RMOT26</strong>, a large-scale
              benchmark with grounded queries and sequence-level splits to prevent
              identity leakage and enable robust generalization evaluation. We further
              present{' '}
              <strong className="highlight-term">QTrack</strong>, an end-to-end
              vision–language model that integrates multi-modal reasoning with
              tracking-oriented localization. Additionally, we introduce a{' '}
              <strong className="highlight-term">
                Temporal Perception-Aware Policy Optimization (TAPO)
              </strong>{' '}
              strategy with structured rewards to encourage motion-aware reasoning.
              Extensive experiments demonstrate the effectiveness of our approach for
              reasoning-centric, language-guided tracking.
            </p>

            {/* Keywords */}
            <div className="abstract-keywords">
              <span className="keywords-label">Keywords</span>
              <div className="keywords-list">
                {keywords.map(kw => (
                  <span key={kw} className="keyword-tag">{kw}</span>
                ))}
              </div>
            </div>

          </div>

          {/* Decorative bracket right */}
          <div className="abstract-bracket right" aria-hidden="true">
            <span /><span /><span />
          </div>

        </div>

      </div>
    </section>
  );
}

export default Abstract;