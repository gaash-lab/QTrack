import { useState, useEffect, useCallback } from 'react';
import '../styles/trajectoryViewer.css';


type State = {
  image?: string;
  description: string;
};

type Step = {
  state: State;
  reasoning?: string;
  action?: string;
};

type Trajectory = {
  name: string;
  description: string;
  totalSteps: number;
  success: boolean;
  validActionRate: string;
  effectiveActionRate: string;
  steps: Step[];
};

type TaskData = {
  taskName: string;
  taskDescription: string;
  trajectories: Record<string, Trajectory>;
};

type Task = {
  id: string;
  name: string;
  description: string;
  dataFile: string;
  thumbnail: string;
};

type TaskIndex = {
  version: string;
  lastUpdated: string;
  tasks: Task[];
};

function TrajectoryViewer() {
  // State variables
  const [currentModel, setCurrentModel] = useState('qwen2.5-0.5b');
  const [availableTasks, setAvailableTasks] = useState<Task[]>([]);
  const [taskData, setTaskData] = useState<Record<string, TaskData>>({});
  const [currentTaskId, setCurrentTaskId] = useState('');
  const [currentTrajectoryId, setCurrentTrajectoryId] = useState('');
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playInterval, setPlayInterval] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Get current trajectory
  const getCurrentTrajectory = useCallback((): Trajectory | null => {
    if (!currentTaskId || !currentTrajectoryId || !taskData[currentTaskId]) {
      return null;
    }
    return taskData[currentTaskId].trajectories[currentTrajectoryId] || null;
  }, [currentTaskId, currentTrajectoryId, taskData]);

  // Load task index
  const loadTaskIndex = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const timestamp = new Date().getTime();
      const indexUrl = `/data/trajectories/${currentModel}/index.json?t=${timestamp}`;
      console.log('Loading task index from:', indexUrl);

      const response = await fetch(indexUrl, {
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`Failed to load task index: ${response.status} ${response.statusText}`);
      }

      const data: TaskIndex = await response.json();
      console.log('Available tasks:', data.tasks);
      setAvailableTasks(data.tasks);

      // Load Sokoban by default if available, otherwise load first task
      const sokobanTask = data.tasks.find((task) => task.id === 'sokoban');
      if (sokobanTask) {
        await loadTaskData('sokoban', data.tasks);
      } else if (data.tasks.length > 0) {
        await loadTaskData(data.tasks[0].id, data.tasks);
      } else {
        setError('No tasks available for the selected model.');
        setLoading(false);
      }
    } catch (err) {
      console.error('Error loading task index:', err);
      setError(`Failed to load tasks: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setLoading(false);
    }
  }, [currentModel]);

  // Load task data
  const loadTaskData = useCallback(
    async (taskId: string, tasks: Task[] = availableTasks) => {
      try {
        setLoading(true);
        setError(null);
        console.log('Loading task data for ID:', taskId);

        const taskInfo = tasks.find((task) => task.id === taskId);
        if (!taskInfo) {
          throw new Error(`Task with ID ${taskId} not found`);
        }

        let dataPath = taskInfo.dataFile;
        if (!dataPath.includes(`/data/trajectories/${currentModel}/`)) {
          const fileName = dataPath.split('/').pop();
          dataPath = `/data/trajectories/${currentModel}/${fileName}`;
        }

        const timestamp = new Date().getTime();
        dataPath = `${dataPath}?t=${timestamp}`;

        console.log('Loading data from path:', dataPath);
        const response = await fetch(dataPath, {
          cache: 'no-store',
        });

        if (!response.ok) {
          throw new Error(`Failed to load task data: ${response.status} ${response.statusText}`);
        }

        const data: TaskData = await response.json();
        console.log('Task data loaded:', data.taskName, 'with', Object.keys(data.trajectories).length, 'trajectories');

        setTaskData((prev) => ({ ...prev, [taskId]: data }));
        setCurrentTaskId(taskId);

        // Load first trajectory by default
        const trajectoryIds = Object.keys(data.trajectories);
        if (trajectoryIds.length > 0) {
          setCurrentTrajectoryId(trajectoryIds[0]);
          setCurrentStepIndex(0);
        } else {
          setError(`No trajectories available for task: ${data.taskName}`);
        }

        setLoading(false);
      } catch (err) {
        console.error(`Error loading task data for ${taskId}:`, err);
        setError(`Failed to load data for ${taskId}: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setLoading(false);
      }
    },
    [currentModel, availableTasks]
  );

  // Initialize on mount
  useEffect(() => {
    loadTaskIndex();
  }, [loadTaskIndex]);

  // Navigate to specific step
  const navigateToStep = useCallback((stepIndex: number) => {
    const trajectory = getCurrentTrajectory();
    if (!trajectory) return;

    const totalSteps = trajectory.steps.length;
    if (stepIndex < 0 || stepIndex >= totalSteps) return;

    setCurrentStepIndex(stepIndex);
  }, [getCurrentTrajectory]);

  // Play/Pause functionality
  const togglePlayPause = useCallback(() => {
    const trajectory = getCurrentTrajectory();
    if (!trajectory) return;

    if (isPlaying) {
      if (playInterval !== null) {
        clearInterval(playInterval);
        setPlayInterval(null);
      }
      setIsPlaying(false);
    } else {
      const interval = window.setInterval(() => {
        setCurrentStepIndex((prev) => {
          const trajectory = getCurrentTrajectory();
          if (!trajectory) return prev;

          if (prev < trajectory.steps.length - 1) {
            return prev + 1;
          } else {
            if (playInterval !== null) {
              clearInterval(playInterval);
              setPlayInterval(null);
            }
            setIsPlaying(false);
            return prev;
          }
        });
      }, 2000);

      setPlayInterval(interval);
      setIsPlaying(true);
    }
  }, [isPlaying, playInterval, getCurrentTrajectory]);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (playInterval !== null) {
        clearInterval(playInterval);
      }
    };
  }, [playInterval]);

  // Get current step
  const trajectory = getCurrentTrajectory();
  const currentStep = trajectory?.steps[currentStepIndex];
  const totalSteps = trajectory?.steps.length || 0;

  // Render timeline steps
  const renderTimelineSteps = () => {
    if (!trajectory) return null;

    const steps = [];
    for (let i = 0; i < totalSteps; i++) {
      const isActive = i === currentStepIndex;
      const isCompleted = i < currentStepIndex;
      const leftPosition = totalSteps === 1 ? 100 : (i / (totalSteps - 1)) * 100;

      steps.push(
        <div
          key={i}
          className={`timeline-step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
          style={{ left: `${leftPosition}%` }}
          onClick={() => navigateToStep(i)}
        />
      );
    }
    return steps;
  };

  // Calculate progress bar width
  const progressBarWidth = () => {
    if (totalSteps === 1) return '100%';
    if (totalSteps === 0) return '0%';
    return `${(currentStepIndex / (totalSteps - 1)) * 100}%`;
  };

  // Format description with code blocks
  const formatDescription = (description: string) => {
    if (!description) return 'No description available.';

    if (description.includes('```')) {
      let formatted = description.replace(/```([\w]*)?(\r\n|\n)([\s\S]*?)```/g, (match, lang, newline, code) => {
        const language = lang ? lang.trim() : '';
        const codeContent = code.trim();
        const languageClass = language ? ` language-${language}` : '';

        return `<div class="code-block${languageClass}"><pre>${codeContent}</pre>${
          language ? `<div class="code-language">${language}</div>` : ''
        }</div>`;
      });

      formatted = formatted.replace(/```([\s\S]*?)```/g, (match, code) => {
        return `<div class="code-block"><pre>${code.trim()}</pre></div>`;
      });

      return <div dangerouslySetInnerHTML={{ __html: formatted }} />;
    }

    return <div dangerouslySetInnerHTML={{ __html: description.replace(/\n/g, '<br>') }} />;
  };

  return (
    <section id="data" className="content-section alt-background">
      <div className="section-container">
        <h2 className="section-title">MOT Trajectory Examples</h2>
        <p className="section-description">
          Explore agent trajectories across different tasks. View state transitions, LLM-generated actions, and the
          decision-making process.
        </p>

        <div className="trajectory-viewer">
          {/* Task Selection */}
          <div className="task-selector">
            <label htmlFor="model-select">Model:</label>
            <select
              id="model-select"
              className="model-dropdown"
              value={currentModel}
              onChange={(e) => {
                setCurrentModel(e.target.value);
                loadTaskIndex();
              }}
            >
              <option value="qwen2.5-0.5b">Qwen2.5-0.5B-Instruct</option>
            </select>

            <label htmlFor="task-select">Task:</label>
            <select
              id="task-select"
              className="task-dropdown"
              value={currentTaskId}
              onChange={(e) => loadTaskData(e.target.value)}
            >
              {availableTasks.map((task) => (
                <option key={task.id} value={task.id}>
                  {task.name}
                </option>
              ))}
            </select>

            <label htmlFor="trajectory-select">Trajectory:</label>
            <select
              id="trajectory-select"
              className="trajectory-dropdown"
              value={currentTrajectoryId}
              onChange={(e) => {
                setCurrentTrajectoryId(e.target.value);
                setCurrentStepIndex(0);
              }}
            >
              {trajectory &&
                Object.entries(taskData[currentTaskId]?.trajectories || {}).map(([id, traj]) => (
                  <option key={id} value={id}>
                    {traj.name || id}
                  </option>
                ))}
            </select>
          </div>

          {/* Loading/Error States */}
          {loading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <p>Loading trajectory data...</p>
            </div>
          )}

          {error && <div className="error-message">{error}</div>}

          {/* Timeline */}
          {!loading && !error && trajectory && (
            <>
              <div className="timeline-container">
                <div className="timeline">
                  <div className="timeline-steps" id="timeline-steps">
                    <div className="timeline-progress" style={{ width: progressBarWidth() }} />
                    {renderTimelineSteps()}
                  </div>
                </div>
                <div className="timeline-controls">
                  <button
                    id="prev-step"
                    className="step-button"
                    aria-label="Previous step"
                    disabled={currentStepIndex === 0}
                    onClick={() => navigateToStep(currentStepIndex - 1)}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      width="32"
                      height="32"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="15 18 9 12 15 6"></polyline>
                    </svg>
                  </button>

                  <span id="step-indicator" className="step-indicator">
                    Step <span id="current-step">{currentStepIndex + 1}</span> of{' '}
                    <span id="total-steps">{totalSteps}</span>
                  </span>

                  <button
                    id="next-step"
                    className="step-button"
                    aria-label="Next step"
                    disabled={currentStepIndex === totalSteps - 1}
                    onClick={() => navigateToStep(currentStepIndex + 1)}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      width="32"
                      height="32"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="9 18 15 12 9 6"></polyline>
                    </svg>
                  </button>

                  <button id="play-pause" className="play-button" aria-label="Play trajectory" onClick={togglePlayPause}>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      width="32"
                      height="32"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="play-icon"
                      style={{ display: isPlaying ? 'none' : 'block' }}
                    >
                      <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      width="32"
                      height="32"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="pause-icon"
                      style={{ display: isPlaying ? 'block' : 'none' }}
                    >
                      <rect x="6" y="4" width="4" height="16"></rect>
                      <rect x="14" y="4" width="4" height="16"></rect>
                    </svg>
                  </button>
                </div>
              </div>

              {/* State and Action Display */}
              <div className="trajectory-display">
                <div className="state-container">
                  <h3 className="display-subtitle">State</h3>
                  <div className="state-content">
                    {currentStep?.state.image && (
                      <div className="state-image">
                        <img
                          id="state-image"
                          src={currentStep.state.image}
                          alt="Current state"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = '/placeholder-state.png';
                          }}
                        />
                      </div>
                    )}
                    <div className="state-description">
                      <div id="state-text">{currentStep && formatDescription(currentStep.state.description)}</div>
                    </div>
                  </div>
                </div>

                <div className="action-container">
                  <h3 className="display-subtitle">LLM Response</h3>
                  <div className="action-content">
                    {currentStep?.reasoning && (
                      <div className="llm-reasoning">
                        <h4>Reasoning:</h4>
                        <div id="reasoning-text" className="code-block">
                          <p>{currentStep.reasoning}</p>
                        </div>
                      </div>
                    )}

                    {currentStep?.action && (
                      <div className="llm-action">
                        <h4>Action:</h4>
                        <div id="action-text" className="code-block action-highlight">
                          {currentStep.action}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

export default TrajectoryViewer;


// import { useEffect, useState, useMemo } from "react";
// import "../styles/trajectoryViewer.css";


// function TrajectoryViewer() {
//   const [model, setModel] = useState("qwen2.5-0.5b");

//   const [tasks, setTasks] = useState([]);
//   const [taskData, setTaskData] = useState({});
//   const [currentTaskId, setCurrentTaskId] = useState("");
//   const [currentTrajectoryId, setCurrentTrajectoryId] = useState("");

//   const [currentStepIndex, setCurrentStepIndex] = useState(0);
//   const [isPlaying, setIsPlaying] = useState(false);

//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);

//   /* ===============================
//      Load Task Index
//   =============================== */
//   useEffect(() => {
//     async function loadTaskIndex() {
//       try {
//         setLoading(true);
//         setError(null);

//         const res = await fetch(
//           `/data/trajectories/${model}/index.json`,
//           { cache: "no-store" }
//         );

//         if (!res.ok) {
//           throw new Error("Failed to load task index");
//         }

//         const data = await res.json();
//         setTasks(data.tasks || []);

//         if (data.tasks?.length > 0) {
//           setCurrentTaskId(data.tasks[0].id);
//         }
//       } catch (err) {
//         console.error(err);
//         setError("Failed to load task index.");
//       } finally {
//         setLoading(false);
//       }
//     }

//     loadTaskIndex();
//   }, [model]);

//   /* ===============================
//      Load Task Data
//   =============================== */
//   useEffect(() => {
//     if (!currentTaskId) return;

//     async function loadTaskData() {
//       try {
//         setLoading(true);
//         setError(null);

//         const res = await fetch(
//           `/data/trajectories/${model}/${currentTaskId}.json`,
//           { cache: "no-store" }
//         );

//         if (!res.ok) {
//           throw new Error("Failed to load task data");
//         }

//         const data = await res.json();

//         setTaskData((prev) => ({
//           ...prev,
//           [currentTaskId]: data,
//         }));

//         const firstTrajectory =
//           Object.keys(data.trajectories || {})[0];

//         setCurrentTrajectoryId(firstTrajectory || "");
//         setCurrentStepIndex(0);
//       } catch (err) {
//         console.error(err);
//         setError("Failed to load task data.");
//       } finally {
//         setLoading(false);
//       }
//     }

//     loadTaskData();
//   }, [currentTaskId, model]);

//   /* ===============================
//      Reset step + stop play when trajectory changes
//   =============================== */
//   useEffect(() => {
//     setCurrentStepIndex(0);
//     setIsPlaying(false);
//   }, [currentTrajectoryId]);

//   /* ===============================
//      Derived Data
//   =============================== */
//   const currentTask = taskData[currentTaskId];

//   const trajectory =
//     currentTask?.trajectories?.[currentTrajectoryId];

//   const steps = useMemo(
//     () => trajectory?.steps || [],
//     [trajectory]
//   );

//   const currentStep = steps[currentStepIndex];

//   /* ===============================
//      Auto Play
//   =============================== */
//   useEffect(() => {
//     if (!isPlaying || steps.length === 0) return;

//     const interval = setInterval(() => {
//       setCurrentStepIndex((prev) => {
//         if (prev < steps.length - 1) {
//           return prev + 1;
//         }
//         setIsPlaying(false);
//         return prev;
//       });
//     }, 2000);

//     return () => clearInterval(interval);
//   }, [isPlaying, steps]);

//   /* ===============================
//      Handlers
//   =============================== */
//   const handlePrev = () => {
//     setCurrentStepIndex((prev) =>
//       Math.max(prev - 1, 0)
//     );
//   };

//   const handleNext = () => {
//     setCurrentStepIndex((prev) =>
//       Math.min(prev + 1, steps.length - 1)
//     );
//   };

//   const handlePlayPause = () => {
//     if (steps.length > 0) {
//       setIsPlaying((prev) => !prev);
//     }
//   };

//   /* ===============================
//      Render
//   =============================== */
//   return (
//     <section className="content-section alt-background">
//       <div className="section-container">
//         <h2 className="section-title">
//           RAGEN Trajectory Examples
//         </h2>

//         {error && (
//           <div className="error-message">{error}</div>
//         )}

//         {loading && (
//           <div className="loading-indicator">
//             Loading...
//           </div>
//         )}

//         {/* ================= Selectors ================= */}
//         <div className="task-selector">
//           <select
//             value={model}
//             onChange={(e) =>
//               setModel(e.target.value)
//             }
//           >
//             <option value="qwen2.5-0.5b">
//               Qwen2.5-0.5B-Instruct
//             </option>
//           </select>

//           <select
//             value={currentTaskId}
//             onChange={(e) =>
//               setCurrentTaskId(e.target.value)
//             }
//           >
//             {tasks.map((task) => (
//               <option
//                 key={task.id}
//                 value={task.id}
//               >
//                 {task.name}
//               </option>
//             ))}
//           </select>

//           <select
//             value={currentTrajectoryId}
//             onChange={(e) =>
//               setCurrentTrajectoryId(e.target.value)
//             }
//           >
//             {currentTask &&
//               Object.entries(
//                 currentTask.trajectories || {}
//               ).map(([id, traj]) => (
//                 <option
//                   key={id}
//                   value={id}
//                 >
//                   {traj.name || id}
//                 </option>
//               ))}
//           </select>
//         </div>

//         {/* ================= Timeline ================= */}
//         {steps.length > 0 && (
//           <>
//             <div className="timeline">
//               {steps.map((_, index) => (
//                 <div
//                   key={index}
//                   className={`timeline-step ${
//                     index === currentStepIndex
//                       ? "active"
//                       : index < currentStepIndex
//                       ? "completed"
//                       : ""
//                   }`}
//                   onClick={() =>
//                     setCurrentStepIndex(index)
//                   }
//                 />
//               ))}
//             </div>

//             {/* ================= Controls ================= */}
//             <div className="timeline-controls">
//               <button
//                 onClick={handlePrev}
//                 disabled={currentStepIndex === 0}
//               >
//                 Prev
//               </button>

//               <span>
//                 Step {currentStepIndex + 1} of{" "}
//                 {steps.length}
//               </span>

//               <button
//                 onClick={handleNext}
//                 disabled={
//                   currentStepIndex ===
//                   steps.length - 1
//                 }
//               >
//                 Next
//               </button>

//               <button onClick={handlePlayPause}>
//                 {isPlaying ? "Pause" : "Play"}
//               </button>
//             </div>

//             {/* ================= Display ================= */}
//             {currentStep && (
//               <div className="trajectory-display">
//                 <div className="state-container">
//                   <h3>State</h3>
//                   {currentStep.state?.image && (
//                     <img
//                       src={
//                         currentStep.state.image
//                       }
//                       alt="state"
//                     />
//                   )}
//                   <p>
//                     {
//                       currentStep.state
//                         ?.description
//                     }
//                   </p>
//                 </div>

//                 <div className="action-container">
//                   <h3>LLM Response</h3>

//                   {currentStep.reasoning && (
//                     <div>
//                       <h4>Reasoning</h4>
//                       <pre>
//                         {
//                           currentStep.reasoning
//                         }
//                       </pre>
//                     </div>
//                   )}

//                   {currentStep.action && (
//                     <div>
//                       <h4>Action</h4>
//                       <pre>
//                         {currentStep.action}
//                       </pre>
//                     </div>
//                   )}
//                 </div>
//               </div>
//             )}
//           </>
//         )}
//       </div>
//     </section>
//   );
// }

// export default TrajectoryViewer;















// // import { useEffect, useState } from "react";
// // import "../styles/trajectoryViewer.css";

// // function TrajectoryViewer() {
// //   const [model, setModel] = useState("qwen2.5-0.5b");
// //   const [tasks, setTasks] = useState([]);
// //   const [taskData, setTaskData] = useState({});
// //   const [currentTaskId, setCurrentTaskId] = useState("");
// //   const [currentTrajectoryId, setCurrentTrajectoryId] = useState("");
// //   const [currentStepIndex, setCurrentStepIndex] = useState(0);
// //   const [isPlaying, setIsPlaying] = useState(false);

// //   // Load Task Index
// //   useEffect(() => {
// //     async function loadTaskIndex() {
// //       try {
// //         const res = await fetch(
// //           `/data/trajectories/${model}/index.json`,
// //           { cache: "no-store" }
// //         );
// //         const data = await res.json();
// //         setTasks(data.tasks);

// //         if (data.tasks.length > 0) {
// //           setCurrentTaskId(data.tasks[0].id);
// //         }
// //       } catch (err) {
// //         console.error("Error loading index:", err);
// //       }
// //     }

// //     loadTaskIndex();
// //   }, [model]);

// //   // Load Task Data
// //   useEffect(() => {
// //     if (!currentTaskId) return;

// //     async function loadTaskData() {
// //       try {
// //         const res = await fetch(
// //           `/data/trajectories/${model}/${currentTaskId}.json`,
// //           { cache: "no-store" }
// //         );
// //         const data = await res.json();
// //         setTaskData((prev) => ({ ...prev, [currentTaskId]: data }));

// //         const firstTrajectory = Object.keys(data.trajectories)[0];
// //         setCurrentTrajectoryId(firstTrajectory);
// //         setCurrentStepIndex(0);
// //       } catch (err) {
// //         console.error("Error loading task data:", err);
// //       }
// //     }

// //     loadTaskData();
// //   }, [currentTaskId, model]);

// //   const currentTask = taskData[currentTaskId];
// //   const trajectory =
// //     currentTask?.trajectories?.[currentTrajectoryId];

// //   const steps = trajectory?.steps || [];
// //   const currentStep = steps[currentStepIndex];

// //   // Auto Play
// //   useEffect(() => {
// //     if (!isPlaying) return;

// //     const interval = setInterval(() => {
// //       setCurrentStepIndex((prev) => {
// //         if (prev < steps.length - 1) return prev + 1;
// //         setIsPlaying(false);
// //         return prev;
// //       });
// //     }, 2000);

// //     return () => clearInterval(interval);
// //   }, [isPlaying, steps.length]);

// //   return (
// //     <section className="content-section alt-background">
// //       <div className="section-container">
// //         <h2 className="section-title">
// //           RAGEN Trajectory Examples
// //         </h2>

// //         {/* Selectors */}
// //         <div className="task-selector">
// //           <select
// //             value={model}
// //             onChange={(e) => setModel(e.target.value)}
// //           >
// //             <option value="qwen2.5-0.5b">
// //               Qwen2.5-0.5B-Instruct
// //             </option>
// //           </select>

// //           <select
// //             value={currentTaskId}
// //             onChange={(e) =>
// //               setCurrentTaskId(e.target.value)
// //             }
// //           >
// //             {tasks.map((task) => (
// //               <option key={task.id} value={task.id}>
// //                 {task.name}
// //               </option>
// //             ))}
// //           </select>

// //           <select
// //             value={currentTrajectoryId}
// //             onChange={(e) =>
// //               setCurrentTrajectoryId(e.target.value)
// //             }
// //           >
// //             {currentTask &&
// //               Object.entries(
// //                 currentTask.trajectories || {}
// //               ).map(([id, traj]) => (
// //                 <option key={id} value={id}>
// //                   {traj.name || id}
// //                 </option>
// //               ))}
// //           </select>
// //         </div>

// //         {/* Timeline */}
// //         <div className="timeline">
// //           {steps.map((_, index) => (
// //             <div
// //               key={index}
// //               className={`timeline-step ${
// //                 index === currentStepIndex
// //                   ? "active"
// //                   : index < currentStepIndex
// //                   ? "completed"
// //                   : ""
// //               }`}
// //               onClick={() =>
// //                 setCurrentStepIndex(index)
// //               }
// //             />
// //           ))}
// //         </div>

// //         {/* Controls */}
// //         <div className="timeline-controls">
// //           <button
// //             disabled={currentStepIndex === 0}
// //             onClick={() =>
// //               setCurrentStepIndex((prev) => prev - 1)
// //             }
// //           >
// //             Prev
// //           </button>

// //           <span>
// //             Step {currentStepIndex + 1} of{" "}
// //             {steps.length}
// //           </span>

// //           <button
// //             disabled={
// //               currentStepIndex === steps.length - 1
// //             }
// //             onClick={() =>
// //               setCurrentStepIndex((prev) => prev + 1)
// //             }
// //           >
// //             Next
// //           </button>

// //           <button
// //             onClick={() =>
// //               setIsPlaying(!isPlaying)
// //             }
// //           >
// //             {isPlaying ? "Pause" : "Play"}
// //           </button>
// //         </div>

// //         {/* State + Action */}
// //         {currentStep && (
// //           <div className="trajectory-display">
// //             <div className="state-container">
// //               <h3>State</h3>
// //               {currentStep.state.image && (
// //                 <img
// //                   src={currentStep.state.image}
// //                   alt="state"
// //                 />
// //               )}
// //               <p>{currentStep.state.description}</p>
// //             </div>

// //             <div className="action-container">
// //               <h3>LLM Response</h3>

// //               {currentStep.reasoning && (
// //                 <div>
// //                   <h4>Reasoning</h4>
// //                   <pre>{currentStep.reasoning}</pre>
// //                 </div>
// //               )}

// //               {currentStep.action && (
// //                 <div>
// //                   <h4>Action</h4>
// //                   <pre>
// //                     {currentStep.action}
// //                   </pre>
// //                 </div>
// //               )}
// //             </div>
// //           </div>
// //         )}
// //       </div>
// //     </section>
// //   );
// // }

// // export default TrajectoryViewer;