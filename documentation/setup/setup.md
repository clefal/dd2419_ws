## Connect from computer

1. To be able to connect to the robot and get the information from there we have to be connected to the same network of the robot
2. Make sure you have the right ROS_DOMAIN_ID. It should be 3. I had to change it in my bashrc file. Check it by running

```bash
echo $ROS_DOMAIN_ID
```

1. Make sure you don’t have a configuration that only allows access to localhost. I had to detele something from my bashrc file (a variable set to =localhost).
2. Connect with ssh to robot. Make sure it is the right ip

run without pixi run:
```bash
ssh group3@{ip}
ssh group3@192.168.1.63
sshpass -p 'group3' ssh group3@192.168.1.63
sshpass -p 'group3' ssh group3@10.94.192.242
```

1. Run any launch. For example

```bash
pixi run lidar
```

1. Open rviz or rqt to send commands or visualize data