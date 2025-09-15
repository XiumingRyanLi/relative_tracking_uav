#!/usr/bin/env python3
import math
import numpy as np
from geometry_msgs.msg import TwistStamped


def MD_Controller(node):
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'base_link'
        
    yaw_rate = 0.0
           
    if node._tracking_enabled:
        # Update integral error for yaw controller

            
        node.yaw_integral_error += node.person_err
        yaw_rate = -node.yaw_kp * node.person_err - node.yaw_ki * node.yaw_integral_error
        yaw_rate = max(min(yaw_rate, node.max_yaw_rate), -node.max_yaw_rate)
        tangential_angle = node.bearing + math.pi/2

        bearing_vector = np.array([math.sin(node.bearing), math.cos(node.bearing)])
        tangential_vector = np.array([math.sin(tangential_angle), math.cos(tangential_angle)])

        Projection_matrix = np.outer(bearing_vector, bearing_vector)
        I = np.eye(2)
        Q = I - Projection_matrix
        position =  [node.pose.pose.position.x, node.pose.pose.position.y]
        estimator_state_update = node.estimator_gain * Q @ (position - node.estimated_state)
        node.estimated_state += estimator_state_update / 20
        estimated_distance_to_target = np.linalg.norm(position - node.estimated_state[:2])
        node.estimated_error = estimated_distance_to_target - node.desired_radius
            
        msg.twist.linear.x = float(node.tangential_speed * tangential_vector[0] + node.parallel_speed * node.estimated_error * bearing_vector[0])
        msg.twist.linear.y = float(node.tangential_speed * tangential_vector[1] + node.parallel_speed * node.estimated_error * bearing_vector[1])
        msg.twist.angular.z = float(yaw_rate)
            
        node.vel_pub.publish(msg)

    return msg




def MDV_Controller(node):
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'base_link'
        
    yaw_rate = 0.0
           
    if node._tracking_enabled:
        # Calculate bearing vector and yaw command
        node.yaw_integral_error += node.person_err
        yaw_rate = -node.yaw_kp * node.person_err - node.yaw_ki * node.yaw_integral_error
        yaw_rate = max(min(yaw_rate, node.max_yaw_rate), -node.max_yaw_rate)
        tangential_angle = node.bearing + math.pi/2
        bearing_vector = np.array([math.sin(node.bearing), math.cos(node.bearing)])
        tangential_vector = np.array([math.sin(tangential_angle), math.cos(tangential_angle)])


        # Udate the estimator
        Projection_matrix = np.outer(bearing_vector, bearing_vector)
        I = np.eye(2)
        Q = I - Projection_matrix
        position =  [node.pose.pose.position.x, node.pose.pose.position.y]
        estimator_state_update = node.estimator_gain * Q @ (position - node.estimated_state)
        node.estimated_state += estimator_state_update / 20
        estimated_distance_to_target = np.linalg.norm(position - node.estimated_state[:2])
        node.estimated_error = estimated_distance_to_target - node.desired_radius

        # Calculate bearing vector from position to target
        position = np.array([node.pose.pose.position.x, node.pose.pose.position.y])
        target_vector = node.estimated_state[:2] - position
        bearing_to_target = math.atan2(target_vector[0], target_vector[1])
        bearing_vector_to_target = np.array([math.sin(bearing_to_target), math.cos(bearing_to_target)])
        tangential_vector_to_target = np.array([bearing_vector_to_target[1], -bearing_vector_to_target[0]])

        
        # Generate the control message
        msg.twist.linear.x = float(node.tangential_speed * tangential_vector_to_target[0] + node.parallel_speed * node.estimated_error * bearing_vector_to_target[0])
        msg.twist.linear.y = float(node.tangential_speed * tangential_vector_to_target[1] + node.parallel_speed * node.estimated_error * bearing_vector_to_target[1])
        msg.twist.angular.z = float(yaw_rate)
            
        node.vel_pub.publish(msg)

    return msg



def DKR_Controller(node):
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'base_link'
        
    yaw_rate = 0.0
           
    if node._tracking_enabled:
        # Calculate bearing vector and yaw command
        node.yaw_integral_error += node.person_err
        yaw_rate = -node.yaw_kp * node.person_err - node.yaw_ki * node.yaw_integral_error
        yaw_rate = max(min(yaw_rate, node.max_yaw_rate), -node.max_yaw_rate)
        tangential_angle = node.bearing + math.pi/2
        bearing_vector = np.array([math.sin(node.bearing), math.cos(node.bearing)])
        tangential_vector = np.array([math.sin(tangential_angle), math.cos(tangential_angle)])


        # Update the estimator
        Projection_matrix = np.outer(bearing_vector, bearing_vector)
        updated_lambda = 1/8**2
        position = [node.pose.pose.position.x, node.pose.pose.position.y]

        node.get_logger().info(f"Determinant of P: {np.linalg.det(node.P)}")
        
        if np.linalg.det(node.P) > 1e-5:
            P_inv = np.linalg.pinv(node.P)
            estimator_state_update = - updated_lambda * (node.estimated_state - (P_inv @ node.Q).flatten())
            node.estimated_state += estimator_state_update 
        
            #Calculate Distance error
            estimated_distance_to_target = np.linalg.norm(position - node.estimated_state[:2])
            node.estimated_error = np.linalg.norm(estimated_distance_to_target - node.desired_radius)


            target_pos = [0,5]
            actual_error = np.linalg.norm(np.array(position) - np.array(target_pos))

            estimated_error = (node.estimated_state - (P_inv @ node.Q).flatten())

            node.get_logger().info(f"Actual error: {actual_error} Estimated error: {estimated_error} Difference: {actual_error - estimated_error}")

        else:
            node.estimated_error = 0.0


        # Calculate bearing vector from position to target
        position = np.array([node.pose.pose.position.x, node.pose.pose.position.y])
        target_vector = node.estimated_state[:2] - position
        bearing_to_target = math.atan2(target_vector[0], target_vector[1])
        bearing_vector_to_target = np.array([math.sin(bearing_to_target), math.cos(bearing_to_target)])
        tangential_vector_to_target = np.array([bearing_vector_to_target[1], -bearing_vector_to_target[0]])

        # Update P and Q matrix 
        dT = 0.125
        exp_neg_dT = math.exp(-dT)
        node.P = exp_neg_dT * node.P + (1 - exp_neg_dT) * Projection_matrix
        position_col = np.array(position).reshape(-1, 1)
        node.Q = exp_neg_dT * node.Q + (1 - exp_neg_dT) * Projection_matrix @ position_col




        # Generate the control message
        msg.twist.linear.x = float(node.tangential_speed * tangential_vector_to_target[0] + node.parallel_speed * node.estimated_error * bearing_vector_to_target[0])
        msg.twist.linear.y = float(node.tangential_speed * tangential_vector_to_target[1] + node.parallel_speed * node.estimated_error * bearing_vector_to_target[1])
        msg.twist.angular.z = float(yaw_rate)
            
        node.vel_pub.publish(msg)

        node.get_logger().info(f"Estimated error: {node.estimated_error}")

    return msg




def KRV_Controller(node):
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'base_link'
        
    yaw_rate = 0.0
           
    if node._tracking_enabled:
        # Calculate bearing vector and yaw command
        node.yaw_integral_error += node.person_err
        yaw_rate = -node.yaw_kp * node.person_err - node.yaw_ki * node.yaw_integral_error
        yaw_rate = max(min(yaw_rate, node.max_yaw_rate), -node.max_yaw_rate)
        tangential_angle = node.bearing + math.pi/2
        bearing_vector = np.array([math.sin(node.bearing), math.cos(node.bearing)])
        tangential_vector = np.array([math.sin(tangential_angle), math.cos(tangential_angle)])


        # Update the estimator
        Projection_matrix = np.outer(bearing_vector, bearing_vector)
        updated_lambda = 1/30**2
        position = [node.pose.pose.position.x, node.pose.pose.position.y]
        P_inv = np.linalg.pinv(node.P)
        
        # MATLAB: obj.x_hat_dot(:,t) = - obj.k_x .* inv(obj.P{t}) * (obj.P{t} * obj.x_hat(:,t) - obj.q{t});
        # Convert estimated_state to column vector for matrix operations
        x_hat_col = node.estimated_state.reshape(-1, 1)
        estimator_state_update = - updated_lambda * (P_inv @ (node.P @ x_hat_col - node.Q)).flatten()
        node.estimated_state += estimator_state_update 
        
        #Calculate Distance error
        estimated_distance_to_target = np.linalg.norm(position - node.estimated_state[:2])
        node.estimated_error = estimated_distance_to_target - node.desired_radius

        # Calculate bearing vector from position to target
        position = np.array([node.pose.pose.position.x, node.pose.pose.position.y])
        target_vector = node.estimated_state[:2] - position
        bearing_to_target = math.atan2(target_vector[0], target_vector[1])
        bearing_vector_to_target = np.array([math.sin(bearing_to_target), math.cos(bearing_to_target)])
        tangential_vector_to_target = np.array([bearing_vector_to_target[1], -bearing_vector_to_target[0]])

        # Update P and Q matrix 
        # MATLAB: obj.P{t+1} = (1-exp(-t*dT)) * obj.P{t} + 1 * obj.bar_varphi * obj.bar_varphi.' ;
        # MATLAB: obj.q{t+1} = (1-exp(-t*dT)) * obj.q{t} + 1 * obj.bar_varphi * obj.bar_varphi.' * obj.p;
        dT = 0.125
        t = node.counter  # Time step counter
        decay_factor = 1 - math.exp(-t * dT)
        
        node.P = decay_factor * node.P + decay_factor  * Projection_matrix
        position_col = np.array(position).reshape(-1, 1)
        node.Q = decay_factor * node.Q + decay_factor * Projection_matrix @ position_col

        node.counter += 1

        # Generate the control message
        msg.twist.linear.x = float(node.tangential_speed * tangential_vector_to_target[0] + node.parallel_speed * node.estimated_error * bearing_vector_to_target[0])
        msg.twist.linear.y = float(node.tangential_speed * tangential_vector_to_target[1] + node.parallel_speed * node.estimated_error * bearing_vector_to_target[1])
        msg.twist.angular.z = float(yaw_rate)
            
        node.vel_pub.publish(msg)

        node.get_logger().info(f"Estimated error: {node.estimated_error}")

    return msg
