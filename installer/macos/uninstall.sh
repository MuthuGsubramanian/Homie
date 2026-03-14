#!/bin/bash
echo "Uninstalling Homie AI..."
launchctl unload ~/Library/LaunchAgents/com.heyhomie.daemon.plist 2>/dev/null
rm -f ~/Library/LaunchAgents/com.heyhomie.daemon.plist
if [ -d "/Applications/Homie AI.app" ]; then
    osascript -e 'tell application "Finder" to delete POSIX file "/Applications/Homie AI.app"' 2>/dev/null || rm -rf "/Applications/Homie AI.app"
fi
echo "Homie AI uninstalled. Your data in ~/.homie/ was preserved."
echo "To remove all data: rm -rf ~/.homie/"
