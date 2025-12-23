apt-get update

apt install software-properties-common
apt-get update


add-apt-repository ppa:savoury1/blender
apt-get update

apt-get update && apt-get install -y --no-install-recommends \
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
    libegl-mesa0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


wget https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.0/blender-4.0.1-linux-x64.tar.xz -O /tmp/blender.tar.xz
mkdir -p /opt/blender && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 && rm /tmp/blender.tar.xz
PATH="/opt/blender:$PATH"


echo $PWD >> /opt/blender/4.0/python/lib/python3.10/site-packages/cyclist.pth
