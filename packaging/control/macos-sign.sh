#!/bin/bash
# Sign + notarize + staple the macOS .app (control-build.yml, only when the Developer ID secrets exist).
# Env: MAC_CERT_P12_B64 MAC_CERT_PASS (Developer ID Application .p12), MAC_NOTARY_KEY_B64 MAC_NOTARY_KEY_ID
# MAC_NOTARY_ISSUER (App Store Connect API key). Secrets are written to the runner's temp dir and removed; never echoed.
set -euo pipefail
APP="$1"
KC="$RUNNER_TEMP/pxa.keychain-db"; P12="$RUNNER_TEMP/pxa.p12"; KEY="$RUNNER_TEMP/notary.p8"
KCPASS="$(uuidgen)"
trap 'rm -f "$P12" "$KEY"; security delete-keychain "$KC" 2>/dev/null || true' EXIT
echo "$MAC_CERT_P12_B64" | base64 --decode > "$P12"
echo "$MAC_NOTARY_KEY_B64" | base64 --decode > "$KEY"
security create-keychain -p "$KCPASS" "$KC"
security set-keychain-settings -lut 21600 "$KC"
security unlock-keychain -p "$KCPASS" "$KC"
security import "$P12" -P "$MAC_CERT_PASS" -A -t cert -f pkcs12 -k "$KC"
security list-keychain -d user -s "$KC"
ID="$(security find-identity -v -p codesigning "$KC" | sed -n 's/.*"\(Developer ID Application:.*\)"/\1/p' | head -1)"
codesign --force --deep --options runtime --timestamp --sign "$ID" "$APP"
codesign --verify --deep --strict "$APP"
ditto -c -k --keepParent "$APP" "$RUNNER_TEMP/app.zip"
xcrun notarytool submit "$RUNNER_TEMP/app.zip" --key "$KEY" --key-id "$MAC_NOTARY_KEY_ID" --issuer "$MAC_NOTARY_ISSUER" --wait
xcrun stapler staple "$APP"
