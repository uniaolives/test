# modules/robotics/sim/gazebo/arkhen_ros2_bridge.py
import rclpy
from rclpy.node import Node as ROSNode
# from geometry_msgs.msg import PoseStamped
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../core/python')))
# from anl import Node as ArkhenNode

class ArkhenROS2Bridge(ROSNode):
    def __init__(self):
        super().__init__('arkhen_bridge')
        print("Arkhe(n) ROS2 Bridge started.")
        # self.arkhen_node = ArkhenNode('sim_drone')
        # self.publisher = self.create_publisher(PoseStamped, '/command/pose', 10)
        # self.arkhen_node.add_handover('goto', self.goto_callback)

    def goto_callback(self, target):
        # msg = PoseStamped()
        # msg.pose.position.x = target[0]
        # msg.pose.position.y = target[1]
        # msg.pose.position.z = target[2]
        # self.publisher.publish(msg)
        return {"status": "sent"}

def main(args=None):
    rclpy.init(args=args)
    node = ArkhenROS2Bridge()
    # rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
