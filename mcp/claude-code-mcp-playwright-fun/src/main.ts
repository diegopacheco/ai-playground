function greet(name: string): string {
  return `Hello, ${name}!`;
}

function main(): void {
  const message = greet("World");
  console.log(message);
}

export { greet, main };

main();
