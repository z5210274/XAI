import math

radius = 160

def toRadian(theta):
    return theta * math.pi / 180

def toDegrees(theta):
    return theta * 180 / math.pi

def getGradient(p1, p2):
    if p1[0] == p2[0]:
        m = toRadian(90)
    else:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    return m

def getAngleFromGradient(gradient):
    return math.atan(gradient)

def getAngle(pos, origin):
    '''m = getGradient(pos, origin)
    thetaRad = getAngleFromGradient(m)
    theta = round(toDegrees(thetaRad), 2)
    return theta'''

    try:
        angle = math.atan((origin[0] - pos[0]) / (origin[1] - pos[1]))
    except:
        angle = math.pi / 2

    if pos[0] < origin[0] and pos[1] > origin[1]:
        angle = abs(angle)
    elif pos[0] < origin[0] and pos[1] < origin[1]:
        angle = math.pi - angle
    elif pos[0] > origin[0] and pos[1] < origin[1]:
        angle = math.pi + abs(angle)
    elif pos[0] > origin[0] and pos[1] > origin[1]:
        angle = (math.pi * 2) - angle

    return round(toDegrees(angle), 2) + 90

def getPosOnCircumference(theta, origin):
    theta = toRadian(theta)
    x = origin[0] + radius * math.cos(theta)
    y = origin[1] + radius * math.sin(theta)
    return (x, y)
