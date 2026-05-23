export interface EvalCase {
  id: string;
  prompt: string;
  url: string;
  expectInScript: string[];
}

export const EVAL_CASES: ReadonlyArray<EvalCase> = [
  {
    id: "saucedemo-login-standard",
    prompt:
      "Log in with standard_user / secret_sauce, see the inventory page with the heading 'Products'",
    url: "https://www.saucedemo.com",
    expectInScript: ["standard_user", "secret_sauce", "Products"],
  },
  {
    id: "saucedemo-login-wrong-password",
    prompt:
      "Try to log in with standard_user and a wrong password 'nope', see the error message containing 'Epic sadface'",
    url: "https://www.saucedemo.com",
    expectInScript: ["Epic sadface"],
  },
  {
    id: "saucedemo-add-to-cart",
    prompt:
      "Log in as standard_user / secret_sauce, add the Sauce Labs Backpack to the cart, verify the cart badge shows 1",
    url: "https://www.saucedemo.com",
    expectInScript: ["Backpack", "1"],
  },
  {
    id: "the-internet-login-success",
    prompt:
      "Log in at /login with username 'tomsmith' and password 'SuperSecretPassword!', see the 'You logged into a secure area!' confirmation",
    url: "https://the-internet.herokuapp.com/login",
    expectInScript: ["tomsmith", "SuperSecretPassword", "secure area"],
  },
  {
    id: "the-internet-checkbox",
    prompt:
      "Visit /checkboxes, check both checkboxes, verify both are now checked",
    url: "https://the-internet.herokuapp.com/checkboxes",
    expectInScript: ["check"],
  },
  {
    id: "the-internet-dropdown",
    prompt:
      "Visit /dropdown, select 'Option 2' from the dropdown, verify Option 2 is selected",
    url: "https://the-internet.herokuapp.com/dropdown",
    expectInScript: ["Option 2"],
  },
  {
    id: "the-internet-broken-images",
    prompt:
      "Visit /broken_images, count how many images are present (3 expected) and assert the page title contains 'Broken Images'",
    url: "https://the-internet.herokuapp.com/broken_images",
    expectInScript: ["Broken Images"],
  },
  {
    id: "the-internet-form-auth",
    prompt:
      "Visit /login. Submit the form WITHOUT typing anything. Verify an error flash appears containing 'invalid'",
    url: "https://the-internet.herokuapp.com/login",
    expectInScript: ["invalid"],
  },
  {
    id: "todomvc-add-todo",
    prompt:
      "Add a todo 'buy milk', then verify it appears in the list",
    url: "https://demo.playwright.dev/todomvc",
    expectInScript: ["buy milk"],
  },
  {
    id: "todomvc-complete-todo",
    prompt:
      "Add two todos: 'buy milk' and 'walk dog'. Mark 'buy milk' as completed. Verify 1 item left.",
    url: "https://demo.playwright.dev/todomvc",
    expectInScript: ["buy milk", "walk dog"],
  },
];
