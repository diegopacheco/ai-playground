export default function Header() {
  return (
    <header style={{
      borderBottom: "1px solid #38444d",
      padding: "16px 20px",
      display: "flex",
      alignItems: "center",
      gap: "10px",
      backgroundColor: "#15202b",
      position: "sticky",
      top: 0,
      zIndex: 10,
    }}>
      <span style={{ fontSize: "28px" }}>🐦</span>
      <h1 style={{
        margin: 0,
        fontSize: "20px",
        fontWeight: 700,
        color: "#e7e9ea",
      }}>Chirper</h1>
    </header>
  );
}
