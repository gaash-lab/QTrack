import './App.css'
import Header from './components/Header';
import AuthorsCitation from "./components/AuthorsCitation";
import Algorithm from './components/Algorithm';
import ChartsSection from './components/ChartsSection';
import Footer from './components/Footer';
import Hero from './components/Hero';
import Results from './components/Results.jsx';
import Abstract from './components/Abstract.jsx';



function App() {
  return (
    <>
      <Header />
        <main>
                <Hero />
                <Abstract />
                <Algorithm/>
                <Results />
                <ChartsSection />
                <AuthorsCitation />
        
        </main>
        
      <Footer />
    </>
  );
}

export default App;
