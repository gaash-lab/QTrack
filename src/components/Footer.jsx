import "../styles/footer.css";

function Footer() {
  const currentYear = new Date().getFullYear(); // dynamic year

  return (
    <footer className="simple-footer">
      <p>
        Copyright © {currentYear} - All right reserved by{" "}
        <a
          href="https://github.com/ya-sonia/MOT"
          target="_blank"
          rel="noopener noreferrer"
        >
          Gaash Lab
        </a>
      </p>

      <p>
        Website maintained by{" "}
        <a
          href="https://github.com/ya-sonia/MOT"
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