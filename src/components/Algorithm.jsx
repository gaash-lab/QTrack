import { useEffect, useState } from 'react';
import '../styles/algorithm.css';

function Algorithm() {
  const [mdpExpanded, setMdpExpanded] = useState(true);
  const [starpoExpanded, setStarpoExpanded] = useState(true);
  const [activeStrategy, setActiveStrategy] = useState('ppo');
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  // Math formulas object
  const mathFormulas = {
    state: "\\(s_t\\)",
    action: "\\(a_t\\)",
    nextState: "\\(s_{t+1}\\)",
    transitionFunction: "\\(P(s_{t+1} | s_t, a_t)\\)",
    policy: "\\(\\pi(a_t | s_t)\\)",
    expectedRewards: "\\(\\mathbb{E}_\\pi[\\sum_t \\gamma^t r_t]\\)",
    ppoFormula: "\\[J_{\\text{PPO}}(\\theta) = \\frac{1}{G} \\sum_{i=1}^G \\frac{1}{|\\tau_i|} \\sum_{t=1}^{|\\tau_i|} \\min \\left( \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\text{old}}(a_t|s_t)} A_{i,t}, \\text{clip}\\left(\\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\text{old}}(a_t|s_t)}, 1-\\epsilon, 1+\\epsilon\\right) A_{i,t} \\right)\\]",
    grpoFormula: "\\[\\hat{A}_{i,t} = \\frac{R(\\tau_i) - \\text{mean}(\\{R(\\tau_1), \\ldots, R(\\tau_G)\\})}{\\text{std}(\\{R(\\tau_1), \\ldots, R(\\tau_G)\\})}\\] \\[J_{\\text{GRPO}}(\\theta) = \\frac{1}{G} \\sum_{i=1}^G \\frac{1}{|\\tau_i|} \\sum_{t=1}^{|\\tau_i|} \\min \\left( \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\text{old}}(a_t|s_t)} \\hat{A}_{i,t}, \\text{clip}\\left(\\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\text{old}}(a_t|s_t)}, 1-\\epsilon, 1+\\epsilon\\right) \\hat{A}_{i,t} \\right)\\]"
  };

  // Load MathJax
  useEffect(() => {
    // Configure MathJax
    window.MathJax = {
      tex: {
        inlineMath: [['\\(', '\\)']],
        displayMath: [['\\[', '\\]']]
      },
      svg: {
        fontCache: 'global'
      }
    };

    // Load polyfill script
    const polyfillScript = document.createElement('script');
    polyfillScript.src = 'https://polyfill.io/v3/polyfill.min.js?features=es6';
    document.head.appendChild(polyfillScript);

    // Load MathJax script
    const mathjaxScript = document.createElement('script');
    mathjaxScript.id = 'MathJax-script';
    mathjaxScript.async = true;
    mathjaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
    document.head.appendChild(mathjaxScript);

    return () => {
      // Cleanup scripts on unmount
      const scripts = document.querySelectorAll('script[src*="mathjax"], script[src*="polyfill"]');
      scripts.forEach(script => script.remove());
    };
  }, []);

  // Typeset math when content changes
  useEffect(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise().catch((err) => console.error('MathJax typeset failed:', err));
    }
  }, [activeStrategy, mdpExpanded, starpoExpanded]);

  // Rollout animation
  const animateRollout = () => {
    const agentThinking = document.getElementById('agent-thinking');
    const agentAction = document.getElementById('agent-action');
    const envFeedback = document.getElementById('env-feedback');

    // Reset
    [agentThinking, agentAction, envFeedback].forEach(el => {
      if (el) {
        el.style.opacity = '0';
        el.style.transform = 'translateY(10px)';
      }
    });

    // Animate thinking
    setTimeout(() => {
      if (agentThinking) {
        agentThinking.style.opacity = '1';
        agentThinking.style.transform = 'translateY(0)';
      }
    }, 500);

    // Animate action
    setTimeout(() => {
      if (agentAction) {
        agentAction.style.opacity = '1';
        agentAction.style.transform = 'translateY(0)';
      }
    }, 2000);

    // Animate feedback
    setTimeout(() => {
      if (envFeedback) {
        envFeedback.style.opacity = '1';
        envFeedback.style.transform = 'translateY(0)';
      }
    }, 3500);
  };

  // Initial animation on mount
  useEffect(() => {
    const timer = setTimeout(animateRollout, 1000);
    return () => clearTimeout(timer);
  }, []);

  // Flow step highlighting
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStepIndex((prev) => (prev + 1) % 4); // 4 steps in the flow
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const toggleMdp = () => {
    setMdpExpanded(!mdpExpanded);
  };

  const toggleStarpo = () => {
    setStarpoExpanded(!starpoExpanded);
  };

  return (
    <section id="starpo" className="content-section alt-background">
      <div className="section-container">
        <h2 className="section-title">
          TrackFlow (<b>T</b>rack-<b>R</b>epresent-<b>A</b>ssociate-<b>C</b>onfirm-<b>K</b>eep <b>Flow</b> Framework)
        </h2>

        {/* Algorithm Flowchart */}
        <div className="algorithm-visualization">
          <div className="visual-container">
            <div className={`flow-step ${currentStepIndex === 0 ? 'active' : ''}`} id="step-initial">
              <div className="step-icon">🔍</div>
              <div className="step-content">
                <h4>Detect Objects</h4>
              </div>
            </div>

            <div className="flow-arrow-wrapper">
              <svg width="50" height="20" viewBox="0 0 50 20" className="flow-arrow">
                <defs>
                  <marker id="arrow-marker" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#2a2f6c" />
                  </marker>
                </defs>
                <path d="M 0 10 L 40 10" stroke="#2a2f6c" strokeWidth="2" markerEnd="url(#arrow-marker)" />
              </svg>
            </div>

            <div className={`flow-step ${currentStepIndex === 1 ? 'active' : ''}`} id="step-reasoning">
              <div className="step-icon">🎯</div>
              <div className="step-content">
                <h4>Track</h4>
              </div>
            </div>

            <div className="flow-arrow-wrapper">
              <svg width="50" height="20" viewBox="0 0 50 20" className="flow-arrow">
                <path d="M 0 10 L 40 10" stroke="#2a2f6c" strokeWidth="2" markerEnd="url(#arrow-marker)" />
              </svg>
            </div>

            <div className={`flow-step ${currentStepIndex === 2 ? 'active' : ''}`} id="step-action">
              <div className="step-icon">🆔</div>
              <div className="step-content">
                <h4>Manage IDs</h4>
              </div>
            </div>

            <div className="flow-arrow-wrapper">
              <svg width="50" height="20" viewBox="0 0 50 20" className="flow-arrow">
                <path d="M 0 10 L 40 10" stroke="#2a2f6c" strokeWidth="2" markerEnd="url(#arrow-marker)" />
              </svg>
            </div>

            <div className={`flow-step ${currentStepIndex === 3 ? 'active' : ''}`} id="step-reward">
              <div className="step-icon">🔄</div>
              <div className="step-content">
                <h4>Update</h4>
              </div>
            </div>

            {/* Repeat Arrow */}
            <div className="repeat-arrow-container">
              <svg className="repeat-arrow" width="100%" height="50" viewBox="0 0 500 50">
                <line x1="-50" y1="5" x2="-50" y2="30" stroke="#2a2f6c" strokeWidth="2" />
                <line x1="-50" y1="30" x2="560" y2="30" stroke="#2a2f6c" strokeWidth="2" />
                <line x1="560" y1="5" x2="560" y2="30" stroke="#2a2f6c" strokeWidth="2" />
                <polygon points="-55,15 -50,5 -45,15" fill="#2a2f6c" />
                <text x="250" y="20" textAnchor="middle" fill="#2a2f6c" fontSize="15" fontWeight="700">
                 Next Frame
                </text>
              </svg>
            </div>
          </div>
        </div>

        {/* Diagram */}
        <div className="diagram-placeholder">
          <img src="/RMOT.png" alt="StarPO Framework" />
          <p style={{ textAlign: 'center', color: '#888' }}>
            The StarPO (State-Thinking-Action-Reward Policy Optimization) framework with two interleaved stages:{' '}
            <strong>rollout stage</strong> and <strong>update stage</strong>.
          </p>
        </div>

        <div className="spacer" style={{ height: '2rem' }}></div>

        {/* Algorithm Components */}
        <div className="algorithm-components">
          {/* MDP Formulation Card */}
          <div className="component-card mdp-card" id="mdp-card">
            <div className="component-header" onClick={toggleMdp} style={{ cursor: 'pointer' }}>
              <h3 className="component-title">MDP Formulation</h3>
              <div className="expand-icon">{mdpExpanded ? '-' : '+'}</div>
            </div>
            <div className={`component-content ${mdpExpanded ? 'active' : ''}`} id="mdp-content">
              <p>
                We formulate agent-environment interactions as Markov Decision Processes (MDPs) where states and
                actions are token sequences, allowing LLMs to reason over environment dynamics.
              </p>

              <div className="interactive-mdp-viz">
                <div className="mdp-state" id="state-0">
                  <span className="state-label">
                    State <span dangerouslySetInnerHTML={{ __html: mathFormulas.state }} />
                  </span>
                  <div className="state-tokens">token sequence</div>
                </div>

                <div className="mdp-arrow">
                  <svg width="50" height="20" viewBox="0 0 50 20">
                    <defs>
                      <marker id="arrowhead1" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#2a2f6c" />
                      </marker>
                    </defs>
                    <path d="M 0 10 L 40 10" stroke="#2a2f6c" strokeWidth="2" markerEnd="url(#arrowhead1)" />
                  </svg>
                  <span className="action-label">
                    Action <span dangerouslySetInnerHTML={{ __html: mathFormulas.action }} />
                  </span>
                </div>

                <div className="mdp-state" id="state-1">
                  <span className="state-label">
                    State <span dangerouslySetInnerHTML={{ __html: mathFormulas.nextState }} />
                  </span>
                  <div className="state-tokens">new token sequence</div>
                </div>
              </div>

              <p>
                At time <span dangerouslySetInnerHTML={{ __html: "\\(t\\)" }} />, state{' '}
                <span dangerouslySetInnerHTML={{ __html: mathFormulas.state }} /> transitions to the next state through
                action <span dangerouslySetInnerHTML={{ __html: mathFormulas.action }} /> following a transition
                function <span dangerouslySetInnerHTML={{ __html: mathFormulas.transitionFunction }} />. The policy{' '}
                <span dangerouslySetInnerHTML={{ __html: mathFormulas.policy }} /> generates actions given the
                trajectory history. The objective is to maximize expected cumulative rewards{' '}
                <span dangerouslySetInnerHTML={{ __html: mathFormulas.expectedRewards }} /> across multiple interaction
                turns.
              </p>
            </div>
          </div>

          {/* StarPO Card */}
          <div className="component-card starpo-card" id="starpo-card">
            <div className="component-header" onClick={toggleStarpo} style={{ cursor: 'pointer' }}>
              <h3 className="component-title">StarPO: Reinforcing Reasoning via Trajectory-Level Optimization</h3>
              <div className="expand-icon">{starpoExpanded ? '-' : '+'}</div>
            </div>
            <div className={`component-content ${starpoExpanded ? 'active' : ''}`} id="starpo-content">
              <p>
                StarPO is a general RL framework for optimizing entire multi-turn interaction trajectories for LLM
                agents. The algorithm alternates between two phases:
              </p>

              <div className="starpo-phases">
                {/* Rollout Phase */}
                <div className="phase-card" id="rollout-phase">
                  <h4>Rollout Stage: Reasoning-Interaction Trajectories</h4>
                  <div className="phase-content">
                    <p>
                      Given an initial Sokoban puzzle state, the LLM generates multiple solving trajectories. At each
                      step, the model receives the puzzle state and generates a reasoning-guided action to push boxes to
                      goal positions:
                    </p>

                    <div className="code-snippet">
                      <code>
                        &lt;think&gt;I need to push the box ($) to the goal (.) which is directly to the
                        right.&lt;/think&gt;&lt;ans&gt; Right &lt;/ans&gt;
                      </code>
                    </div>

                    <p>The environment receives the action and returns the next state with the box pushed to the goal.</p>

                    <div className="interactive-rollout">
                      <div className="rollout-step">
                        <div className="agent-box">
                          <div className="agent-icon">🤖</div>
                          <div className="agent-bubble thinking" id="agent-thinking">
                            I need to push the box ($) to the goal (.) which is directly to the right. To do this, I need
                            to move right and push the box.
                          </div>
                          <div className="agent-bubble action" id="agent-action">
                            Right
                          </div>
                        </div>

                        <div className="env-box">
                          <div className="env-icon">🌐</div>
                          <div className="env-bubble feedback" id="env-feedback">
                            #####
                            <br />
                            #_@*#
                            <br />
                            #####
                            <br />
                            <br />
                            Box pushed to goal position. State updated.
                          </div>
                        </div>
                      </div>

                      <button className="replay-button" id="replay-rollout" onClick={animateRollout}>
                        ▶️ Replay Interaction
                      </button>
                    </div>
                  </div>
                </div>

                {/* Update Phase */}
                <div className="phase-card" id="update-phase">
                  <h4>Update Stage: Multi-turn Trajectory Optimization</h4>
                  <div className="phase-content">
                    <p>
                      After generating trajectories, we train LLMs to optimize expected rewards. Instead of step-by-step
                      optimization, StarPO optimizes entire trajectories using importance sampling. This approach enables
                      long-horizon reasoning while maintaining computational efficiency.
                    </p>

                    <div className="optimization-strategies">
                      <h4 className="strategies-title">StarPO supports multiple optimization strategies:</h4>

                      <div className="strategy-tabs">
                        <div
                          className={`strategy-tab ${activeStrategy === 'ppo' ? 'active' : ''}`}
                          onClick={() => setActiveStrategy('ppo')}
                        >
                          PPO
                        </div>
                        <div
                          className={`strategy-tab ${activeStrategy === 'grpo' ? 'active' : ''}`}
                          onClick={() => setActiveStrategy('grpo')}
                        >
                          GRPO
                        </div>
                      </div>

                      <div className="strategy-content">
                        {/* PPO Content */}
                        <div className={`strategy-details ${activeStrategy === 'ppo' ? 'active' : ''}`} id="ppo-content">
                          <p>
                            <strong>PPO (Proximal Policy Optimization):</strong>
                          </p>
                          <div className="formula">
                            <span dangerouslySetInnerHTML={{ __html: mathFormulas.ppoFormula }} />
                          </div>
                        </div>

                        {/* GRPO Content */}
                        <div className={`strategy-details ${activeStrategy === 'grpo' ? 'active' : ''}`} id="grpo-content">
                          <p>
                            <strong>GRPO (Group Relative Policy Optimization):</strong>
                          </p>
                          <div className="formula">
                            <span dangerouslySetInnerHTML={{ __html: mathFormulas.grpoFormula }} />
                          </div>
                        </div>
                      </div>
                    </div>

                    <p className="conclusion-text">
                      Rollout and update stages interleave in StarPO, enabling both online and offline learning.
                    </p>
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














// import { useEffect, useState } from "react";
// import "../styles/algorithm.css";

// const mathFormulas = {
//   state: "\\(s_t\\)",
//   action: "\\(a_t\\)",
//   nextState: "\\(s_{t+1}\\)",
//   transitionFunction: "\\(P(s_{t+1} | s_t, a_t)\\)",
//   policy: "\\(\\pi(a_t | s_t)\\)",
//   expectedRewards: "\\(\\mathbb{E}_\\pi[\\sum_t \\gamma^t r_t]\\)",
// };

// function Algorithm() {
//   const [activeStrategy, setActiveStrategy] = useState("ppo");
//   const [mdpOpen, setMdpOpen] = useState(true);
//   const [starpoOpen, setStarpoOpen] = useState(true);

//   useEffect(() => {
//     if (window.MathJax) {
//       window.MathJax.typeset();
//     }

//     const flowSteps = document.querySelectorAll(".flow-step");
//     let currentStepIndex = 0;

//     function highlightNextStep() {
//       flowSteps.forEach((step) => step.classList.remove("active"));
//       if (flowSteps[currentStepIndex]) {
//         flowSteps[currentStepIndex].classList.add("active");
//       }
//       currentStepIndex = (currentStepIndex + 1) % flowSteps.length;
//     }

//     highlightNextStep();
//     const interval = setInterval(highlightNextStep, 2000);

//     return () => clearInterval(interval);
//   }, []);

//   return (
//     <section id="starpo" className="content-section alt-background">
//       <div className="section-container">
//         <h2 className="section-title">
//           StarPO (State-Thinking-Action-Reward Policy Optimization)
//         </h2>

//         {/* Flow Steps */}
//         <div className="algorithm-visualization">
//           <div className="visual-container">
//             {["Initial State", "Reasoning", "Action", "Reward"].map(
//               (step, index) => (
//                 <div key={index} className="flow-step">
//                   <div className="step-content">
//                     <h4>{step}</h4>
//                   </div>
//                 </div>
//               )
//             )}
//           </div>
//         </div>

//         {/* Image */}
//         <div className="diagram-placeholder">
//           <img src="/starpo.png" alt="StarPO Framework" />
//         </div>

//         {/* MDP Card */}
//         <div className="component-card">
//           <div
//             className="component-header"
//             onClick={() => setMdpOpen(!mdpOpen)}
//           >
//             <h3>MDP Formulation</h3>
//             <div className="expand-icon">
//               {mdpOpen ? "-" : "+"}
//             </div>
//           </div>

//           {mdpOpen && (
//             <div className="component-content">
//               <p>
//                 We formulate agent-environment interactions as MDPs where
//                 states and actions are token sequences.
//               </p>

//               <p>
//                 State{" "}
//                 <span
//                   dangerouslySetInnerHTML={{
//                     __html: mathFormulas.state,
//                   }}
//                 />{" "}
//                 transitions via action{" "}
//                 <span
//                   dangerouslySetInnerHTML={{
//                     __html: mathFormulas.action,
//                   }}
//                 />
//               </p>
//             </div>
//           )}
//         </div>

//         {/* Strategy Tabs */}
//         <div className="optimization-strategies">
//           <div className="strategy-tabs">
//             <div
//               className={`strategy-tab ${
//                 activeStrategy === "ppo" ? "active" : ""
//               }`}
//               onClick={() => setActiveStrategy("ppo")}
//             >
//               PPO
//             </div>

//             <div
//               className={`strategy-tab ${
//                 activeStrategy === "grpo" ? "active" : ""
//               }`}
//               onClick={() => setActiveStrategy("grpo")}
//             >
//               GRPO
//             </div>
//           </div>

//           <div className="strategy-content">
//             {activeStrategy === "ppo" && (
//               <div>PPO Formula Here</div>
//             )}

//             {activeStrategy === "grpo" && (
//               <div>GRPO Formula Here</div>
//             )}
//           </div>
//         </div>
//       </div>
//     </section>
//   );
// }

// export default Algorithm;