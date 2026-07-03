#!/bin/bash

SCRIPT_DIR=/Users/diegopacheco/git/diegopacheco/ai-playground/pocs/claude-window-starter
SCRIPT_PATH=$SCRIPT_DIR/claude-window-starter.sh
PLIST_PATH=~/Library/LaunchAgents/com.claude-window-starter.plist
LOG_FILE=/Users/diegopacheco/claude-window-starter.log

chmod +x "$SCRIPT_PATH"

if [ -f "$PLIST_PATH" ]; then
    echo "Unloading existing background job..."
    launchctl unload "$PLIST_PATH" 2>/dev/null
fi

echo "Creating LaunchAgent plist at: $(eval echo $PLIST_PATH)"
cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.claude-window-starter</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SCRIPT_PATH</string>
    </array>
    <key>StartInterval</key>
    <integer>14400</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>
    <key>StandardErrorPath</key>
    <string>$LOG_FILE</string>
</dict>
</plist>
EOF

echo "Loading LaunchAgent..."
launchctl load "$PLIST_PATH"

echo ""
echo "Background job installed."
echo "Schedule: every 4 hours (14400s)"
echo "Log file: $LOG_FILE"
echo ""

echo "Background process status:"
launchctl list | grep claude-window-starter
echo ""

if [ -f "$LOG_FILE" ]; then
    echo "Current logs from $LOG_FILE:"
    cat "$LOG_FILE"
else
    echo "No logs yet in $LOG_FILE"
fi
