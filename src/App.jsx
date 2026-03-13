import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'
import Header from './components/Header';
// import AuthorsCitation from "./components/AuthorsCitation";
// import Algorithm from './components/Algorithm';
// import ChartsSection from './components/ChartsSection';
// import FindingsSection from './components/FindingsSection';
import Footer from './components/Footer';
// import TrajectoryViewer from "./components/TrajectoryViewer.tsx";
import Hero from './components/Hero';



function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Header />
        <main>
                <Hero />
                {/* <Algorithm/>
                <ChartsSection />
                <FindingsSection />
                <TrajectoryViewer />
                <AuthorsCitation /> */}
        
        </main>
        
      <Footer />
    </>
  );
}

export default App;
