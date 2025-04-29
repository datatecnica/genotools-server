#!/bin/sh

echo "[INFO] Running docker-setup.sh"

sudo apt update
sudo apt install -y unzip

# Make iaap-cli executable if it exists
IAAP_CLI="/tmp/genotools-server/bin/iaap-cli-linux-x64-1.1.0/iaap-cli/iaap-cli"
if [ -f "$IAAP_CLI" ]; then
    echo "[INFO] Making iaap-cli executable"
    chmod +x "$IAAP_CLI"
fi

# Set the PLINK version
PLINK_VERSION="1.9"
# Set the URL for the latest PLINK 1.9 release
PLINK_URL="https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20220402.zip"
# Set the VM directory where PLINK will be installed
PLINK_INSTALL_DIR="/tmp/genotools-server/bin/plink1.9"
# Create the PLINK installation directory
mkdir -p $PLINK_INSTALL_DIR
# Download and unzip the latest PLINK 1.9 release
curl -L $PLINK_URL -o plink1.9.zip
unzip plink1.9.zip -d $PLINK_INSTALL_DIR
# Create the module file
mkdir -p /etc/modulefiles/plink1.9
cat <<EOF > /etc/modulefiles/plink1.9/${PLINK_VERSION}
#%Module
set plink_root ${PLINK_INSTALL_DIR}
prepend-path PATH \$plink_root
prepend-path LD_LIBRARY_PATH \$plink_root
EOF

# Set the PLINK version
PLINK_VERSION="2.3"
# Set the URL for the latest PLINK 1.9 release
PLINK_URL="https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_x86_64_20241114.zip"
# Set the VM directory where PLINK will be installed
PLINK_INSTALL_DIR="/tmp/genotools-server/bin/plink2"
# Create the PLINK installation directory
mkdir -p $PLINK_INSTALL_DIR
# Download and unzip the latest PLINK release
curl -L $PLINK_URL -o plink2.zip
unzip plink2.zip -d $PLINK_INSTALL_DIR
# Create the module file
mkdir -p /etc/modulefiles/plink2
cat <<EOF > /etc/modulefiles/plink2/${PLINK_VERSION}
#%Module
set plink_root ${PLINK_INSTALL_DIR}/plink2
prepend-path PATH $plink_root
prepend-path LD_LIBRARY_PATH $plink_root
EOF

# Make PLINK executable
PLINK="/tmp/genotools-server/bin/plink1.9/plink"
if [ -f "$PLINK" ]; then
    echo "[INFO] Making PLINK1.9 executable"
    chmod +x "$PLINK"
fi
PLINK2="/tmp/genotools-server/bin/plink2/plink2"
if [ -f "$PLINK2" ]; then
    echo "[INFO] Making PLINK2 executable"
    chmod +x "$PLINK2"
fi


# Start shell
exec /bin/sh