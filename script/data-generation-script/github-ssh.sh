#!/bin/bash

# === GitHub SSH Setup Script ===
# Run this on a new computer to set up SSH authentication with GitHub.

EMAIL="sami.haq99@gmail.com"   # <-- replace with your GitHub email

echo "🔑 Generating a new SSH key for $EMAIL ..."
ssh-keygen -t ed25519 -C "$EMAIL" -f ~/.ssh/id_ed25519 -N ""

echo "🚀 Starting ssh-agent ..."
eval "$(ssh-agent -s)"

echo "➕ Adding SSH key to agent ..."
ssh-add ~/.ssh/id_ed25519

echo "📋 Your SSH public key (copy this and add it to GitHub → Settings → SSH and GPG keys):"
echo "------------------------------------------------------------------"
cat ~/.ssh/id_ed25519.pub
echo "------------------------------------------------------------------"

echo "✅ Now go to: https://github.com/settings/keys and add this key."
echo "Then test with: ssh -T git@github.com"
