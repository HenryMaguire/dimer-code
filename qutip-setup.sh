sudo apt-get update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt-get -y upgrade
sudo apt install g++-7 -y
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
sudo apt install python2.7 python-pip
sudo apt-get install python-dev

sudo -H pip2 install --upgrade pip
sudo -H pip2 install numpy scipy matplotlib cython
sudo -H pip2 install qutip
