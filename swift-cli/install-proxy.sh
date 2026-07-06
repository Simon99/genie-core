#!/bin/bash
# Install genie-speech-cli as a LaunchAgent (runs in GUI session, has TCC permissions)
# Usage: ./install-proxy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$HOME/bin"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_NAME="com.genie.speech-proxy"

# Build
echo "Building genie-speech-cli..."
cd "$SCRIPT_DIR"
swift build -c release

# Install binary atomically (cp over a running/executed binary corrupts the
# kernel's cached code signature -> launchd kills it with OS_REASON_CODESIGNING)
mkdir -p "$BIN_DIR"
cp .build/release/genie-speech-cli "$BIN_DIR/genie-speech-cli.new"
mv -f "$BIN_DIR/genie-speech-cli.new" "$BIN_DIR/genie-speech-cli"
echo "Installed to $BIN_DIR/genie-speech-cli"

# Create LaunchAgent
mkdir -p "$PLIST_DIR"
mkdir -p "$HOME/Library/Logs"
cat > "$PLIST_DIR/$PLIST_NAME.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_NAME</string>
    <key>ProgramArguments</key>
    <array>
        <string>$BIN_DIR/genie-speech-cli</string>
        <string>--server</string>
        <string>--port</string>
        <string>5300</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/genie-speech-proxy.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/genie-speech-proxy.log</string>
</dict>
</plist>
EOF

# Load
launchctl unload "$PLIST_DIR/$PLIST_NAME.plist" 2>/dev/null || true
launchctl load "$PLIST_DIR/$PLIST_NAME.plist"

echo "Speech proxy installed and running on port 5300"
echo "Test: curl http://localhost:5300/health"
