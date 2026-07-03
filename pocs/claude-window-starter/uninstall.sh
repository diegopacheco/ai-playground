#!/bin/bash

PLIST_PATH=~/Library/LaunchAgents/com.claude-window-starter.plist

if [ -f "$PLIST_PATH" ]; then
    echo "Unloading background job..."
    launchctl unload "$PLIST_PATH" 2>/dev/null
    rm "$PLIST_PATH"
    echo "Removed $(eval echo $PLIST_PATH)"
else
    echo "No LaunchAgent found."
fi
