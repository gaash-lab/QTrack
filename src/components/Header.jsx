import { useEffect, useState } from "react";
import "../styles/header.css";

function formatNumber(num) {
  return num > 999 ? (num / 1000).toFixed(1) + "k" : num;
}

function Header() {
  const [stars, setStars] = useState(null);
  const [forks, setForks] = useState(null);

  useEffect(() => {
    async function fetchGitHubStats() {
      try {
        const response = await fetch(
          "https://api.github.com/repos/RAGEN-AI/RAGEN"
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
    <header>
      <div className="container">
        <a href="#hero" className="info-badge">
          <img src="/icon.png" alt="Icon" className="badge-icon" />
          <span className="badge-text">
            <strong>Gaash Lab</strong>
          </span>
        </a>

        <div className="nav-wrapper">
          <nav className="main-nav">
            <ul>
              <li className="dropdown">
                <a href="#" className="dropdown-toggle">
                  More Research
                  <svg
                    className="dropdown-icon"
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <polyline points="6 9 12 15 18 9"></polyline>
                  </svg>
                </a>

                <div className="dropdown-menu">
                  <a
                    href="https://vagen-ai.github.io/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    VAGEN
                  </a>

                  <a
                    href="https://mind-cube.github.io/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    MindCube
                  </a>

                  <a
                    href="https://embodied-agent-interface.github.io/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Embodied Agent Interface
                  </a>
                </div>
              </li>

              <li>
                <a
                  href="https://www.mll.lab.northwestern.edu/"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  MLL Lab
                </a>
              </li>

              <li>
                <a
                  href="https://github.com/RAGEN-AI/RAGEN"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="github-repo"
                >
                  <div className="repo-icon">
                    <svg
                          height="22"
                          viewBox="0 0 16 16"
                          fill="white"
                          aria-hidden="true"
                        >
                          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 
                          0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                          -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 
                          2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 
                          0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 
                          0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 
                          1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82 
                          .44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 
                          0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 
                          0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 
                          8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                  </div>

                  {/* <div className="repo-stats">
                    ⭐ {stars !== null ? formatNumber(stars) : "-"}
                    {"  "}
                    🍴 {forks !== null ? formatNumber(forks) : "-"}
                  </div> */}
                </a>
              </li>
            </ul>
          </nav>
        </div>
      </div>
    </header>
  );
}

export default Header;



// import "../styles/header.css";

// function Header() {
//   return (
//     <header>
//       <div className="container">
//         <a href="#hero" className="info-badge">
//           <img src="/icon.png" alt="Icon" className="badge-icon" />
//           <span className="badge-text">
//             <strong>RAGEN</strong>
//           </span>
//         </a>

//         <div className="nav-wrapper">
//           <nav className="main-nav">
//             <ul>
//               <li className="dropdown">
//                 <a href="#" className="dropdown-toggle">
//                   More Research
//                   <svg
//                     xmlns="http://www.w3.org/2000/svg"
//                     width="12"
//                     height="12"
//                     viewBox="0 0 24 24"
//                     fill="none"
//                     stroke="currentColor"
//                     strokeWidth="2"
//                     strokeLinecap="round"
//                     strokeLinejoin="round"
//                     className="dropdown-icon"
//                   >
//                     <polyline points="6 9 12 15 18 9"></polyline>
//                   </svg>
//                 </a>

//                 <div className="dropdown-menu">
//                   <a
//                     href="https://vagen-ai.github.io/"
//                     target="_blank"
//                     rel="noopener noreferrer"
//                     className="dropdown-item"
//                   >
//                     VAGEN
//                   </a>

//                   <a
//                     href="https://mind-cube.github.io/"
//                     target="_blank"
//                     rel="noopener noreferrer"
//                     className="dropdown-item"
//                   >
//                     MindCube
//                   </a>

//                   <a
//                     href="https://embodied-agent-interface.github.io/"
//                     target="_blank"
//                     rel="noopener noreferrer"
//                     className="dropdown-item"
//                   >
//                     Embodied Agent Interface
//                   </a>
//                 </div>
//               </li>

//               <li>
//                 <a
//                   href="https://www.mll.lab.northwestern.edu/"
//                   target="_blank"
//                   rel="noopener noreferrer"
//                 >
//                   MLL Lab
//                 </a>
//               </li>

//               <li>
//                 <a
//                   href="https://github.com/RAGEN-AI/RAGEN"
//                   target="_blank"
//                   rel="noopener noreferrer"
//                   className="github-repo"
//                 >
//                   <div className="repo-icon">
//                     <svg
//                       aria-hidden="true"
//                       height="24"
//                       viewBox="0 0 16 16"
//                       width="24"
//                       fill="currentColor"
//                     >
//                       <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59..."></path>
//                     </svg>
//                   </div>
//                 </a>
//               </li>
//             </ul>
//           </nav>
//         </div>
//       </div>
//     </header>
//   );
// }

// export default Header;