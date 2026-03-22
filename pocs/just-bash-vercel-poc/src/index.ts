import { Bash } from "just-bash";

async function main() {
  const bash = new Bash({
    env: { GREETING: "Hello from just-bash!" },
  });

  console.log("=== Writing and reading files ===");
  await bash.exec('echo "just-bash is awesome" > hello.txt');
  let result = await bash.exec("cat hello.txt");
  console.log(result.stdout);

  console.log("=== Environment variables ===");
  result = await bash.exec("echo $GREETING");
  console.log(result.stdout);

  console.log("=== Pipes and text processing ===");
  await bash.exec('echo "banana\napple\ncherry\ndate" > fruits.txt');
  result = await bash.exec("cat fruits.txt | sort | head -n 2");
  console.log(result.stdout);

  console.log("=== Loops ===");
  result = await bash.exec('for i in 1 2 3; do echo "item-$i"; done');
  console.log(result.stdout);

  console.log("=== Grep ===");
  await bash.exec('echo "error: something failed\ninfo: all good\nerror: another issue" > log.txt');
  result = await bash.exec("grep error log.txt");
  console.log(result.stdout);

  console.log("=== Directory operations ===");
  await bash.exec("mkdir -p /app/data");
  await bash.exec('echo "config=true" > /app/data/config.txt');
  result = await bash.exec("ls /app/data");
  console.log(result.stdout);
  result = await bash.exec("cat /app/data/config.txt");
  console.log(result.stdout);

  console.log("=== Heredoc ===");
  result = await bash.exec(`cat <<EOF
name: just-bash
version: 2.14
type: virtual-shell
EOF`);
  console.log(result.stdout);

  console.log("Done!");
}

main().catch(console.error);
