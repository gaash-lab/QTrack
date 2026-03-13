// import { useEffect } from "react";
// import Layout from "../layouts/Layout";
// import Hero from "../components/Hero";
// import Algorithm from "../components/Algorithm";
// import FindingsSection from "../components/FindingsSection";
// import TrajectoryViewer from "../components/TrajectoryViewer";
// import AuthorsCitation from "../components/AuthorsCitation";

// // Import styles
// import "../styles/section.css";
// import "../styles/hero.css";
// import "../styles/authors.css";
// import "../styles/resources.css";
// import "../styles/trajectoryViewer.css";

// // Import JS functions
// import { initTrajectoryViewer } from "../scripts/trajectoryViewer";
// import { initCitationCopy } from "../scripts/citation";

// function Home() {
//   const pageTitle = "Sonia  Tawheed MOT";

//   useEffect(() => {
//     // Initialize external scripts
//     initCitationCopy();
//     initTrajectoryViewer();

//     // Text animation for hero section
//     const animateTitle = () => {
//       const titleElement = document.getElementById("animated-title");
//       const subtitleElement = document.getElementById("animated-subtitle");

//       if (titleElement && subtitleElement) {
//         // Animate title
//         const titleText = titleElement.innerText;
//         titleElement.innerHTML = "";

//         for (let i = 0; i < titleText.length; i++) {
//           const charSpan = document.createElement("span");
//           charSpan.className = "reveal-char";
//           charSpan.style.animationDelay = `${i * 0.1}s`;

//           charSpan.innerHTML =
//             titleText[i] === " " ? "&nbsp;" : titleText[i];

//           titleElement.appendChild(charSpan);
//         }

//         // Animate subtitle
//         const subtitleText = subtitleElement.innerText;
//         subtitleElement.innerHTML = "";

//         for (let i = 0; i < subtitleText.length; i++) {
//           const charSpan = document.createElement("span");
//           charSpan.className = "reveal-char-subtitle";
//           charSpan.style.animationDelay = `${i * 0.05 + 0.5}s`;

//           charSpan.innerHTML =
//             subtitleText[i] === " " ? "&nbsp;" : subtitleText[i];

//           subtitleElement.appendChild(charSpan);
//         }
//       }
//     };

//     animateTitle();
//   }, []);

//   return (
//     <Layout title={pageTitle}>
//       <Hero />
//       <Algorithm />
//       <FindingsSection />
//       <TrajectoryViewer />
//       <AuthorsCitation />
//     </Layout>
//   );
// }

// export default Home;