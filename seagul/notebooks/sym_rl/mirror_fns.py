# Mirroring functions for  HumanoidBulletEnv-v0 and Walker2DBulletEnv-v0
# Thomas Ibbetson

import tensorflow as tf

# "more" comes from here:https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/robot_locomotors.py
def mirror_more(more):
    #    z,sin_target,cos_target,vx,vy,vz,roll,pitch = more  # This line requires an iteration over more, when more is a tf.Tensor with eager=False that is not allowed
    #    return [z, -sin_target, cos_target, vx, -vy, vz, -roll, -pitch]

    mask = tf.constant([1,-1,-1,1,-1,1,-1,-1],dtype=more.dtype)
    return tf.multiply(mask, more)
    

def mirror_walker_obs(obs):
    m_more = mirror_more(obs[:,0:8])
    m_feet_contact  = tf.concat([obs[:,21], obs[:,20]],axis=0)    
    m_feet_contact = tf.reshape(m_feet_contact,(-1,2))
    m_rgr = obs[:,8:14]
    m_grg = obs[:,14:20]
    return tf.concat([m_more,m_grg,m_rgr,m_feet_contact],axis=1)

def mirror_walker_act(act):
    m_act = tf.concat([act[:,3:6],act[:,0:3]],axis=1)
    return m_act

def mirror_human_obs(obs):
    m_more = mirror_more(obs[:,0:8])
    m_feet_contact = [obs[:,43],obs[:,42]]
    # all joints have [pos, velocity], so abs = 4 entries for 2 joints
    # abs   [ twist left, arch back ]  *relative to main torso, negate twist, arch stays same
    m_abd  = np.hstack( [-obs[:,8:10],obs[:,10:12] ]) 
    # center hip  [  roll left ]       *relative to abs, negate
    m_hip_c  = -obs[:,12:14] 
    # right hip   [ slide left,twist left,swing back ]    *relative to hip center
    # left hip    [ slide right, twist left, swing back ] *relative to hip center
    hip_r = obs[:,14:20]; m_hip_l  = hip_r
    hip_l = obs[:,20:26]; m_hip_r  = hip_l
    # right knee  [ bend forward ]                        *relative to right hip
    # left knee   [ bend forward ]                        *relative to left hip  
    # just swap them
    knee_r = obs[:,26:28]; m_knee_l = knee_r    
    knee_l = obs[:,28:30]; m_knee_r = knee_l       
    # right shldr [ pull in, rotate forward ]             *relative to main torso
    # left shldr  [ pull in, rotate backward ]            *relative to main torso
    # swap them, negate rotation.
    shld_r = obs[:,30:34]; m_shld_l = tf.concat([shld_r[:,0:2],-shld_r[:,2:4]], axis=1)
    shld_l = obs[:,34:38]; m_shld_r = tf.concat([shld_l[:,0:2],-shld_l[:,2:4]], axis=1)
    # right elbow [ bend in ]                             *relative to right shoulder
    # left elbow  [ bend in ]                             *relative to left should
    # just swap them
    elb_r  = obs[:,38:40]; m_elb_l = elb_r
    elb_l  = obs[:,40:42]; m_elb_r = elb_l  
    
    m_obs = tf.concat([m_more,m_abd,m_hip_c,m_hip_r,m_knee_r,m_hip_l,m_knee_l,m_shld_r,m_elb_r,m_shld_l,m_elb_l,m_feet_contact],axis=1)
    return m_obs

def mirror_human_act(act):
    # abs   [ twist left, arch back ]  *relative to main torso, negate twist, arch stays same
    m_abd = tf.concat([-act[:,0:1],act[:,1:2]],axis=1)
    # center hip  [  roll left ]       *relative to abs, negate
    m_hip_c = -act[:,2:3]
    # right hip   [ slide left,twist left,swing back ]    *relative to hip center
    # left hip    [ slide right, twist left, swing back ] *relative to hip center
    # just swap them
    hip_r = act[:,3:6]; m_hip_l  = hip_r
    hip_l = act[:,6:9]; m_hip_r  = hip_l
    # right knee  [ bend forward ]                        *relative to right hip
    # left knee   [ bend forward ]                        *relative to left hip  
    # just swap them
    knee_r = obs[:,9:10];  m_knee_l = knee_r    
    knee_l = obs[:,10:11]; m_knee_r = knee_l       
    # right shldr [ pull in, rotate forward ]             *relative to main torso
    # left shldr  [ pull in, rotate backward ]            *relative to main torso
    # swap them, negate rotation.
    shld_r = obs[:,11:13]; m_shld_l = tf.concat([shld_r[0],-shld_r[1]],axis=1) 
    shld_l = obs[:,13:15]; m_shld_r = tf.concat([shld_l[0],-shld_l[1]],axis=1)
    # right elbow [ bend in ]                             *relative to right shoulder
    # left elbow  [ bend in ]                             *relative to left should
    # just swap them
    elb_r  = obs[:,15:16]; m_elb_l = elb_r
    elb_l  = obs[:,16:17]; m_elb_r = elb_l
    m_act = tf.concat([m_abd,m_hip_c,m_hip_r,m_knee_r,m_hip_l,m_knee_l,m_shld_r,m_elb_r,m_shld_l,m_elb_l],axis=1)
    return m_act

