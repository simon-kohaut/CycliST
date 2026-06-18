#!/usr/bin/env bash
set -euo pipefail

BLENDER_VERSION="4.0.1"
BLENDER_INSTALL_DIR="/opt/blender"
BLENDER_URL="https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.0/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

# --- System dependencies ---
apt-get update -qq
apt-get install -y --no-install-recommends software-properties-common
add-apt-repository -y ppa:savoury1/blender
apt-get update -qq
apt-get install -y --no-install-recommends \
    wget \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    libgl1-mesa-glx \
    libopenal1 \
    libsndfile1 \
    libfontconfig1 \
    libfreetype6 \
    xz-utils \
    libxkbcommon-x11-0 \
    libsm6 \
    libegl1 \
    libegl-mesa0
apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Download and install Blender ---
echo "Downloading Blender ${BLENDER_VERSION}..."
wget -q "${BLENDER_URL}" -O /tmp/blender.tar.xz
mkdir -p "${BLENDER_INSTALL_DIR}"
tar -xf /tmp/blender.tar.xz -C "${BLENDER_INSTALL_DIR}" --strip-components=1
rm /tmp/blender.tar.xz
echo "Blender installed to ${BLENDER_INSTALL_DIR}"

# Add Blender to PATH for the current session and persistently
export PATH="${BLENDER_INSTALL_DIR}:${PATH}"
if ! grep -q "${BLENDER_INSTALL_DIR}" "${HOME}/.bashrc" 2>/dev/null; then
    echo "export PATH=\"${BLENDER_INSTALL_DIR}:\${PATH}\"" >> "${HOME}/.bashrc"
    echo "Added Blender to PATH in ~/.bashrc — run 'source ~/.bashrc' or open a new shell"
fi

# --- Register cyclist package with Blender's bundled Python ---
# This allows renderer.py to import cyclist when called from within Blender
PTH_FILE="${BLENDER_INSTALL_DIR}/4.0/python/lib/python3.10/site-packages/cyclist.pth"
echo "${PWD}" >> "${PTH_FILE}"
echo "Registered ${PWD} in Blender's Python path at ${PTH_FILE}"

echo "Done. Verify with: blender --version"
