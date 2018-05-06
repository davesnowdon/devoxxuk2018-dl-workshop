# -*- mode: ruby -*-
# vi: set ft=ruby :

$script = <<SCRIPT
sudo apt-get install tree -y
sudo apt-get install software-properties-common
sudo apt-add-repository ppa:ansible/ansible
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install -y oracle-java8-installer \
                        oracle-java8-set-default \
                        git \
                        maven \
                        gradle \
                        ansible
sudo mkdir /ansible
sudo cp /etc/ansible/ansible.cfg /ansible
sudo touch /ansible/hosts
sudo sed -i 's?#inventory      = /etc/ansible/hosts?inventory = /ansible/hosts?g' /ansible/ansible.cfg
echo "---------------------------------------------------------------------------------------"
echo "*** Creating public-private keypair for ansible-playbook... for user '$USER' *** "
yes y | ssh-keygen -t rsa -b 4096 -C "devoxx-dl-workshop@example.com" -q -P "" -f $HOME/.ssh/id_rsa
echo "---------------------------------------------------------------------------------------"
echo "*** Public-private keypair for ansible-playbook... for user '$USER' has been created in *** "
ls -lash $HOME/.ssh
echo "---------------------------------------------------------------------------------------"
export VAGRANT_MOUNT=/vagrant
echo "*** Creating 'hosts' file for ansible-playbook... ***"
echo "localhost ansible_connection=local" > $VAGRANT_MOUNT/hosts
echo "---------------------------------------------------------------------------------------"
echo "*** '$VAGRANT_MOUNT/hosts' file for ansible-playbook contains ***"
cat $VAGRANT_MOUNT/hosts
echo "*** Checking the version of Java installed ***"
java -version
echo "*** Cloning the devoxxuk2018-dl-workshop library ***"
cd $HOME && git clone http://github.com/davesnowdon/devoxxuk2018-dl-workshop.git
cd $HOME/devoxxuk2018-dl-workshop
ls
chmod +x gradlew
./gradlew
./gradlew updateOfflineRepository -PofflineRepositoryRoot=./offline-repository
./gradlew -PofflineRepositoryRoot=./offline-repository :ex0-setup:ex0run --offline
echo "---------------------------------------------------------------------------------------"
SCRIPT

Vagrant.configure("2") do |config|

  config.vm.define :dlworkshop do |dlworkshop|
    dlworkshop.vm.box = "ubuntu/trusty64"
    dlworkshop.vm.hostname = "dlworkshop"
    dlworkshop.vm.network :private_network, type: "dhcp"
    dlworkshop.vm.provision "shell", inline: $script, privileged: false
  end
end
