import { useState } from "react";
import "../styles/findingsSection.css";

const findingsData = [
  {
    title:
      "Finding 1: Single-turn RL may not be directly adapted to Multi-turn agent RL",
    content:
      "Vanilla adaptations from single-turn methods like PPO and GRPO achieve early gains in agent settings but often collapse. A critic in PPO may delay instability, but would not prevent reasoning degradation, highlighting the need for specialized stabilization in agent settings.",
  },
  {
    title:
      'Finding 2: Model collapse in agent RL is reflected as "Echo Trap" over training',
    content:
      'We find that early-stage agent respond with diverse symbolic reasoning, but collapse into deterministic, repetitive templates after training. Models converge to fixed phrasing, indicating that RL may reinforce superficial patterns instead of general reasoning and forms an "Echo Trap" that hinders long-term generalization.',
  },
  {
    title:
      "Finding 3: Collapse follows similar dynamics and can be anticipated by indicators",
    content:
      "Reward standard deviation and entropy often fluctuate before performance degrades, while gradient norm spikes typically mark the point of irreversible collapse. These metrics provide early indicators and motivate the need for stabilization strategies.",
  },
  {
    title:
      "Finding 4: Filtering low-variance trajectories improves stability and efficiency",
    content:
      "Training on high-variance prompts delays or eliminates collapse in multi-turn RL. StarPO-S improves performance and reduces update steps by discarding low-information rollouts, especially under PPO.",
  },
  {
    title:
      "Finding 5: Task diversity, action budget, and rollout frequency affect data quality",
    content:
      "Diverse task instances enable better policy contrast and generalization across environments. Moderate action budgets provide enough planning space and avoid noise introduced by overly long sequences.",
  },
  {
    title:
      "Finding 6: Reasoning fails to emerge without meticulous reward design",
    content:
      "While symbolic reasoning can emerge in simple tasks, it fails to persist in multi-turn environments without reward design encouraging intermediate reasoning steps.",
  },
];

function FindingsSection() {
  const [activeIndexes, setActiveIndexes] = useState(
    findingsData.map(() => true) // all open by default
  );

  const toggleCard = (index) => {
    setActiveIndexes((prev) =>
      prev.map((item, i) => (i === index ? !item : item))
    );
  };

  return (
    <section id="findings" className="content-section">
      <div className="section-container">
        <h2 className="section-title">Findings</h2>

        <div className="findings-components">
          {findingsData.map((finding, index) => (
            <div key={index} className="finding-card">
              <div
                className="finding-header"
                onClick={() => toggleCard(index)}
              >
                <h3 className="finding-title">{finding.title}</h3>
                <div className="expand-icon">
                  {activeIndexes[index] ? "-" : "+"}
                </div>
              </div>

              {activeIndexes[index] && (
                <div className="finding-content active">
                  <p>{finding.content}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default FindingsSection;