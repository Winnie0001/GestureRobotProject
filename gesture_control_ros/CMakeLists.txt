cmake_minimum_required(VERSION 3.0.2)
project(gesture_control_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/gesture_publisher.py
  scripts/robot_controller.py
  scripts/tcp_command_bridge.py
  scripts/nav_evaluate.py
  scripts/nav_plot_results.py
  scripts/session_recorder.py
  scripts/session_analyze.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
