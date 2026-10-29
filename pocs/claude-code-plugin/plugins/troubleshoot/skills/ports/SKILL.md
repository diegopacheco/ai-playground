---
name: ports
description: Find which process holds a port and whether it answers. Use when the user says a port is already in use, a service does not respond, or runs /troubleshoot:ports.
---

# Ports

Check the ports given in the arguments: `$ARGUMENTS`.

1. For each port run `lsof -nP -iTCP:<port> -sTCP:LISTEN`.
2. When a process listens, report its pid and command.
3. When it is an HTTP port, check that it answers with a short request and report the status code.
4. When nothing listens, say the port is free.

Never kill a process. Report one line per port with the port, FREE or BUSY, the pid, the command and the answer.
