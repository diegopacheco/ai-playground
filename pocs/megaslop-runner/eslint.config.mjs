export default [
  {
    files: ["**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        window: "readonly",
        document: "readonly",
        localStorage: "readonly",
        performance: "readonly",
        requestAnimationFrame: "readonly"
      }
    },
    rules: {}
  }
];
