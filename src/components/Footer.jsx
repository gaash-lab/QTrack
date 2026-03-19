import "../styles/footer.css";

function Footer() {
  const currentYear = new Date().getFullYear(); 

  return (
    <footer className="simple-footer">
      <p>
        Copyright © {currentYear} - All right reserved by{" "}
        <a
          href="https://gaash.nitsri.ac.in"
          target="_blank"
          rel="noopener noreferrer"
        >
          Gaash Lab
        </a>
      </p>

      <p>
        Website maintained by{" "}
        <a
          href="https://gaash.nitsri.ac.in"
          target="_blank"
          rel="noopener noreferrer"
        >
          Gaash Lab
        </a>
      </p>
    </footer>
  );
}

export default Footer;