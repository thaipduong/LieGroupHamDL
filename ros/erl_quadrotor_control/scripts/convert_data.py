#!/usr/bin/env python
  

import rospy
from std_msgs.msg import Bool
from mavros_msgs.msg import ERLQuadStates
from tf.transformations import quaternion_matrix
import numpy as np

 
collecting_data = False
prev_collecting_data = False
first_msg = True
start_time = None
prev_time = None
dataset = []
time_span = []
raw_time = []
def collectdatacallback(msg):
    global collecting_data, prev_collecting_data
    prev_collecting_data = collecting_data
    collecting_data = msg.data
    # print the actual message in its raw format
    rospy.loginfo("Data collecting: %s", collecting_data)
    # otherwise simply print a convenient message on the terminal
    # print('Data from /topic_name received')
  
 
def quadstatecallback(msg):
    global first_msg, collecting_data, dataset, time_span, start_time, prev_time, raw_time
    if collecting_data:
        rospy.loginfo("Receiving ERLQuadStates data from mavros")
        if first_msg:
            first_msg = False
            start_time = msg.timestamp
            prev_time = msg.timestamp
        rotmat = quaternion_matrix([msg.orientation[1], msg.orientation[2], msg.orientation[3], msg.orientation[0]])
        #rospy.loginfo("Will process the message here!!!!")
        #print(rotmat)
        collected_state = np.concatenate((msg.position[0:3], rotmat[0:3, 0:3].flatten(), msg.velocity, msg.angular_velocity, msg.controls))
        dataset.append(collected_state)
        time_span.append(np.array([msg.timestamp - start_time, msg.timestamp - prev_time]))
        raw_time.append(msg.timestamp)
        prev_time = msg.timestamp
    if prev_collecting_data and not collecting_data:
        print("################################################################SAVING DATASET....")
        np.savez("~/dataset.npz", dataset = dataset, time_span = time_span, raw_time=raw_time)

  
 
def main():
      
    rospy.init_node('convert_data', anonymous=True)
    rospy.Subscriber("/collect_data", Bool, collectdatacallback)
    rospy.Subscriber("/mavros/erl/erl_quad_states", ERLQuadStates, quadstatecallback)
      
    # spin() simply keeps python from
    # exiting until this node is stopped
    rospy.spin()
  
if __name__ == '__main__':
      
    # you could name this function
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    print("################################################################SAVING DATASET....")
    np.savez("~/dataset.npz", dataset = dataset, time_span = time_span, raw_time=raw_time)
