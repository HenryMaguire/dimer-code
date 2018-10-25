sudo apt-get update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt-get -y upgrade
sudo apt install g++-7 -y
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
sudo apt install python-minimal python-pip
sudo apt-get install python-dev

sudo -H pip2 install --upgrade pip
sudo -H pip2 install numpy==1.15.1 scipy==1.1.0 matplotlib==2.2.3 cython==0.28.5
sudo -H pip2 install qutip==4.3.1
sudo -H pip2 install jupyter==1.0.0
