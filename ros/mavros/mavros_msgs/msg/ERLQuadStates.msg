# ERL Quad States 
uint64 timestamp		        # time since system start (microseconds)
float32[3] position             	# position x, y, z in world-frame 
float32[4] orientation          	# orientation in quaternion [w, x, y, z]
float32[3] velocity             	# velocity in body-frame 
float32[3] angular_velocity     	# angular velocity in body-frame 
float32[4] controls	            	# thrust, torque x, torque y, torque z 
float32[4] controls_scaled	    	# thrust, torque x, torque y, torque z (after battery scaling) 