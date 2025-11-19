You are tasked with generating comprehensive unit tests for code files.

When the user provides a file path or code snippet, analyze the code and create unit tests that:

1. Test all public functions and methods
2. Cover edge cases and error conditions
3. Include positive and negative test scenarios
4. Test boundary conditions
5. Verify expected behavior with valid and invalid inputs
6. Use appropriate assertions and test structure

Follow these guidelines:

- Detect the testing framework already in use (Jest, Mocha, pytest, JUnit, etc.) by examining package.json, requirements.txt, or existing test files
- Match the existing test file naming convention (*.test.js, *_test.py, *Test.java, etc.)
- Place tests in the appropriate directory following project structure
- Use the same code style and patterns as existing tests
- Ensure tests are isolated and independent
- Mock external dependencies appropriately
- Make tests clear and maintainable
- Include setup and teardown when needed
- Aim for high code coverage

If no testing framework is detected, suggest appropriate frameworks for the language and set one up.

After generating tests, run them to verify they pass and provide the output.
