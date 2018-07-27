try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import numpy as np
# from tqdm import tqdm
from numpy import deg2rad, rad2deg

buffer_size = 1000
run = 9
port = 19998

print ("run {}, port {}".format(run, port))

ball_min_window = 3
z_ball_threshold = 0.17499

def handleErr(status, msg=''):
    if status != 0:
        print ("error:", status)
        if msg != '':
            print ("error encountered while", msg)
        quit()


# start with
# vrep.exe -h -gREMOTEAPISERVERSERVICE_19998_FALSE_FALSE D:\flo\documents\dev\ergo-experiments\vrep-video-recorder\poppy_ergo_jr_vanilla_ball.ttt

print ('Program started')
vrep.simxFinish(-1)  # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1', port, True, True, 5000, 5)  # Connect to V-REP
if clientID == -1:
    print ('couldn\'t connect to remote API server')
    quit()

print ('Connected to remote API server')

# positions = np.loadtxt('throws.csv', dtype=np.int16, delimiter=',')

start = run*buffer_size
end = (run+1)*buffer_size

height = np.zeros((buffer_size, 1), dtype=np.float32)

# initiate the joints
joint_handles = []
for i in range(6):
    motorName = 'm' + str(i + 1)
    err, joint_n = vrep.simxGetObjectHandle(clientID, motorName, vrep.simx_opmode_blocking)
    handleErr(err, 'getting the motor handle: ' + motorName)
    joint_handles.append(joint_n)

    vrep.simxGetJointPosition(clientID, joint_n, vrep.simx_opmode_streaming)
    time.sleep(0.2)
    err, joint_n_pos = vrep.simxGetJointPosition(clientID, joint_n, vrep.simx_opmode_buffer)
    handleErr(err, 'getting the motor initial position: ' + motorName)

    print (err, "motor {} pos: {}".format(i, rad2deg(joint_n_pos)))


def getBaseLink():
    err, baseHandle = vrep.simxGetObjectHandle(clientID, 'base_link_respondable', vrep.simx_opmode_blocking)
    handleErr(err, 'getting base link')
    return baseHandle


def getCurrentPos():
    out = []
    for i in range(6):
        err, joint_n_pos = vrep.simxGetJointPosition(clientID, joint_handles[i], vrep.simx_opmode_buffer)
        out.append(rad2deg(joint_n_pos))
    return out


def issueNextMotorCmd(nextCmdIdx):
    for i in range(6):
        motorPos = positions[nextCmdIdx, i]
        # print "motor {}: {}".format(i, motorPos)
        vrep.simxSetJointTargetPosition(clientID, joint_handles[i], deg2rad(motorPos), vrep.simx_opmode_streaming)


restMotors = [0, -90, 35, 0, 55, -90]


def gotoPos(pos, sleep=1.0):
    for i in range(6):
        vrep.simxSetJointTargetPosition(clientID, joint_handles[i], deg2rad(-pos[i]), vrep.simx_opmode_streaming)
    time.sleep(sleep)


def restPos():
    # print "resting"
    gotoPos(restMotors, 0.5)


# parameters:
# 0 - cube, 1 - sphere, 2 -..
# bit-encoded (1) - back-face culling, (2) - edges visible, (4) - smooth surface, (8) - responsive shape, (16) - static
# size triplet

# LUA CODE FOR V-REP
#
# spawnBall=function(inInts,inFloats,inStrings,inBuffer)
#     local size={inFloats[1],inFloats[2],inFloats[3]}
#     local position={inFloats[4],inFloats[5],inFloats[6]}
#     local ballHandle=simCreatePureShape(1, 12, size, inFloats[7], nil)
#     simSetObjectName(ballHandle,'ball')
#     simSetObjectPosition(ballHandle,-1,position)
#     return {ballHandle},{},{},''
# end

emptyBuff = bytearray()


def addBall():
    err, ret_ints, ret_floats, ret_bytes, ret_string = vrep.simxCallScriptFunction(
        clientID,
        "remoteApiCommandServer",
        vrep.sim_scripttype_childscript,
        'spawnBall',
        [],
        [.035, .035, .035,  # size
         0, 0.05, .3,  # position
         .01],  # weight
        [],
        emptyBuff,
        vrep.simx_opmode_blocking
    )

    ballID = ret_ints[0]
    # init position streaming for ball - don't need anymore, cause synchronous recording
    # vrep.simxGetObjectPosition(clientID, ballID, -1, vrep.simx_opmode_streaming)

    return ballID


def removeBall(ballID):
    err = vrep.simxRemoveObject(clientID, ballID, vrep.simx_opmode_blocking)
    handleErr(err, 'removing ball')


def throwBall(run):
    gotoPos(positions[run], 0)


def getBallPos(ballID, base):
    err, pos = vrep.simxGetObjectPosition(clientID, ballID, base, vrep.simx_opmode_blocking)
    handleErr(err, 'getting ball position')
    return pos


def recordThrow(ballID, i):
    base = getBaseLink()
    vrep.simxSynchronous(clientID, True)
    ballPos = getBallPos(ballID, base)

    ballHeights = [ballPos[2]]  # * ball_min_window

    throwBall(i)
    vrep.simxSynchronousTrigger(clientID)

    j = 0
    while True:
        # check if ballPos z is lower than minimum
        ballPos = getBallPos(ballID, -1) # base
        ballHeights.append(ballPos[2])

        # if yes, exist, return max height
        # else  simxSynchronousTrigger(clientID)
        vrep.simxSynchronousTrigger(clientID)
        j += 1
        if j > 5:
            if ballPos[2] <= z_ball_threshold or j == 50:
                break

    vrep.simxSynchronous(clientID, False)
    return max(ballHeights)


print ("recording throws")
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
time.sleep(1)
idx = 0
for i in tqdm(range(start, end)):
    restPos()
    ballID = addBall()
    time.sleep(.5)

    # move to start pos
    gotoPos([0, 40, 40, 0, -90, -55])

    # loop, monitor max height
    maxHeight = recordThrow(ballID, i)
    height[idx] = maxHeight
    idx += 1

    removeBall(ballID)

vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
vrep.simxGetPingTime(clientID)

# Now close the connection to V-REP:
vrep.simxFinish(clientID)

np.savetxt('heights-{}.csv'.format(run), height, fmt='%f', delimiter=',', newline='\n')

print ('DONE')
