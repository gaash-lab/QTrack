import { useState } from 'react';
import "../styles/results.css"
function Results() {
  const [activeTab, setActiveTab] = useState('rmot26');

  return (
    <section id="results">
      <div className="section-container">

        {/* Section Title */}
        <h2 className="section-title">Results</h2>

        {/* Tab Navigation */}
        <div className="results-tabs">
          <button
            className={`results-tab ${activeTab === 'rmot26' ? 'active' : ''}`}
            onClick={() => setActiveTab('rmot26')}
          >
            RMOT26 Benchmark
          </button>
          <button
            className={`results-tab ${activeTab === 'mot' ? 'active' : ''}`}
            onClick={() => setActiveTab('mot')}
          >
            MOT17 &amp; DanceTrack
          </button>
        </div>

        {/* ───── Tab 1: RMOT26 ───── */}
        {activeTab === 'rmot26' && (
          <div className="table-section">
            <p className="table-caption">
              <strong>Table 1: Main results on the RMOT26 benchmark.</strong>{' '}
              Higher is better for MCP and MOTP, while lower is better for CLE and NDE.
            </p>
            <div className="table-wrapper">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Model Name</th>
                    <th>Params</th>
                    <th>MCP ↑</th>
                    <th>MOTP ↑</th>
                    <th>CLE (px) ↓</th>
                    <th>NDE ↓</th>
                  </tr>
                </thead>
                <tbody>
                  {/* Open-Source */}
                  <tr className="group-header opensource-header">
                    <td colSpan={6}>Open-Source Models</td>
                  </tr>
                  <tr>
                    <td>Qwen2.5-VL-Instruct</td>
                    <td>7B</td>
                    <td>0.24</td>
                    <td>0.48</td>
                    <td>289.2</td>
                    <td>2.07</td>
                  </tr>
                  <tr>
                    <td>Qwen3-VL-Instruct</td>
                    <td>8B</td>
                    <td className="underline-val">0.25</td>
                    <td>0.64</td>
                    <td>96.0</td>
                    <td>0.97</td>
                  </tr>
                  <tr>
                    <td>Gemma 3</td>
                    <td>27B</td>
                    <td>0.24</td>
                    <td>0.56</td>
                    <td className="underline-val">58.4</td>
                    <td>0.88</td>
                  </tr>
                  <tr>
                    <td>Gemma 3</td>
                    <td>12B</td>
                    <td>0.18</td>
                    <td className="underline-val">0.73</td>
                    <td>172.9</td>
                    <td>0.95</td>
                  </tr>
                  <tr>
                    <td>Llama 3.2 Vision-Instruct</td>
                    <td>11B</td>
                    <td>0.19</td>
                    <td>0.15</td>
                    <td>552.1</td>
                    <td>2.67</td>
                  </tr>
                  <tr>
                    <td>DeepSeek</td>
                    <td>16B</td>
                    <td>0.11</td>
                    <td>0.27</td>
                    <td>989.13</td>
                    <td>4.7</td>
                  </tr>
                  <tr>
                    <td>Mistral-3-Instruct</td>
                    <td>8B</td>
                    <td>0.15</td>
                    <td>0.54</td>
                    <td>225.3</td>
                    <td>0.98</td>
                  </tr>

                  {/* Reasoning */}
                  <tr className="group-header reasoning-header">
                    <td colSpan={6}>Reasoning Models</td>
                  </tr>
                  <tr>
                    <td>VisionReasoner</td>
                    <td>7B</td>
                    <td>0.23</td>
                    <td>0.24</td>
                    <td>428.9</td>
                    <td>2.24</td>
                  </tr>
                  <tr>
                    <td>VisionReasoner</td>
                    <td>3B</td>
                    <td>0.21</td>
                    <td>0.44</td>
                    <td>416.58</td>
                    <td>2.32</td>
                  </tr>
                  <tr>
                    <td>Migician</td>
                    <td>7B</td>
                    <td className="underline-val">0.25</td>
                    <td>0.22</td>
                    <td>658.39</td>
                    <td>3.28</td>
                  </tr>
                  <tr>
                    <td>InternVL</td>
                    <td>8B</td>
                    <td>0.21</td>
                    <td>0.66</td>
                    <td>117.44</td>
                    <td>0.64</td>
                  </tr>

                  {/* Closed-Source */}
                  <tr className="group-header closed-header">
                    <td colSpan={6}>Closed-Source Models</td>
                  </tr>
                  <tr>
                    <td>gpt-4o-mini</td>
                    <td>—</td>
                    <td>0.20</td>
                    <td>0.57</td>
                    <td>130.48</td>
                    <td>0.67</td>
                  </tr>
                  <tr>
                    <td>gpt-5.2</td>
                    <td>—</td>
                    <td className="underline-val">0.25</td>
                    <td>0.61</td>
                    <td>94.2</td>
                    <td className="underline-val">0.55</td>
                  </tr>

                  {/* Ours */}
                  <tr className="ours-row">
                    <td><strong>QTrack (Ours)</strong></td>
                    <td><strong>3B</strong></td>
                    <td><strong>0.30</strong></td>
                    <td><strong>0.75</strong></td>
                    <td><strong>44.61</strong></td>
                    <td><strong>0.39</strong></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ───── Tab 2: MOT17 + DanceTrack ───── */}
        {activeTab === 'mot' && (
          <div className="table-section">
            <p className="table-caption">
              <strong>Table 2: Comparison with traditional MOT methods.</strong>{' '}
              QTrack improves HOTA and MOTP while maintaining competitive MOTA.
            </p>
            <div className="mot-grid">

              {/* MOT17 */}
              <div className="mot-subtable">
                <h4 className="subtable-title">MOT17 Dataset</h4>
                <div className="table-wrapper">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>MOTA</th>
                        <th>MOTP</th>
                        <th>HOTA</th>
                        <th>MCP</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>MOTR</td>
                        <td>0.61</td>
                        <td>0.81</td>
                        <td>0.22</td>
                        <td className="underline-val">0.44</td>
                      </tr>
                      <tr>
                        <td>BoostTrack++</td>
                        <td>0.63</td>
                        <td>0.76</td>
                        <td>0.38</td>
                        <td><strong>0.44</strong></td>
                      </tr>
                      <tr>
                        <td>TrackTrack</td>
                        <td><strong>0.75</strong></td>
                        <td>0.50</td>
                        <td>0.23</td>
                        <td>0.29</td>
                      </tr>
                      <tr>
                        <td>VisionReasoner</td>
                        <td>0.64</td>
                        <td className="underline-val">0.86</td>
                        <td className="underline-val">0.60</td>
                        <td>0.21</td>
                      </tr>
                      <tr className="ours-row">
                        <td><strong>QTrack</strong></td>
                        <td className="underline-val">0.69</td>
                        <td><strong>0.87</strong></td>
                        <td><strong>0.69</strong></td>
                        <td>0.26</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* DanceTrack */}
              <div className="mot-subtable">
                <h4 className="subtable-title">DanceTrack Dataset</h4>
                <div className="table-wrapper">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>MOTA</th>
                        <th>MOTP</th>
                        <th>HOTA</th>
                        <th>MCP</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>MOTR</td>
                        <td>0.42</td>
                        <td>0.70</td>
                        <td>0.35</td>
                        <td>0.51</td>
                      </tr>
                      <tr>
                        <td>MOTRv2</td>
                        <td>0.49</td>
                        <td>0.73</td>
                        <td>0.37</td>
                        <td className="underline-val">0.52</td>
                      </tr>
                      <tr>
                        <td>TrackTrack</td>
                        <td>0.36</td>
                        <td>0.73</td>
                        <td>0.40</td>
                        <td><strong>0.55</strong></td>
                      </tr>
                      <tr>
                        <td>VisionReasoner</td>
                        <td className="underline-val">0.59</td>
                        <td className="underline-val">0.85</td>
                        <td className="underline-val">0.61</td>
                        <td>0.26</td>
                      </tr>
                      <tr className="ours-row">
                        <td><strong>QTrack</strong></td>
                        <td><strong>0.63</strong></td>
                        <td><strong>0.83</strong></td>
                        <td><strong>0.66</strong></td>
                        <td>0.35</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

            </div>
          </div>
        )}

      </div>
    </section>
  );
}

export default Results;