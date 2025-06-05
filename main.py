import matplotlib.pyplot as plt
#from basic_image_process import BasicImageProcess
import numpy as np
import cv2
import math
import imageio

def perp(v):
    return np.array([v[1], -v[0]])

def lineLength(l):
    v = l[1] - l[0]
    return np.sqrt(np.sum(v * v))

def distPointLine(x, l):
    P = l[0]
    Q = l[1]
    PQ = Q - P
    PX = x - P
    u = PX.dot(PQ) / PQ.dot(PQ)
    if u < 0:
        return np.sqrt(PX.dot(PX))
    elif u > 1:
        QX = x - Q
        return np.sqrt(QX.dot(QX))
    else:
        v = PX.dot(perp(PQ)) / np.sqrt(PQ.dot(PQ))
        return abs(v)

def backTransform(x, l1, l2):
    Pp = l1[0]
    Qp = l1[1]
    P = l2[0]
    Q = l2[1]
    PX = x - P
    PQ = Q - P
    PpQp = Qp - Pp
    u = PX.dot(PQ) / PQ.dot(PQ)
    v = PX.dot(perp(PQ)) / np.sqrt(PQ.dot(PQ))
    return np.int16(Pp + u * PpQp + v * perp(PpQp) / np.sqrt(PpQp.dot(PpQp)))

def warp(image, L1, L2, A, B, P):
    h = image.shape[0]
    w = image.shape[1]
    #print("warp")
    #h, w = image.shape[:2]
    line_num = L1.shape[0] # number of lines
    
    I_trans = np.zeros(image.shape, dtype='uint8')
    for i in range(h):
        print("warp i:", i, "/", h)
        for j in range(w):
            x = np.array([i, j])
            disp_sum = np.zeros((2))
            weight_sum = 0
            for k in range(line_num):
                displacement = backTransform(x, L1[k], L2[k]) - x
                dist = distPointLine(x, L2[k])
                # weight = pow(pow(lineLength(L2[k]), P) / (A + dist), B)
                weight = pow(1 / (A + dist), B)
                disp_sum += displacement * weight
                weight_sum += weight
            pos = np.clip(x + np.int16(disp_sum / weight_sum), [0, 0], [h-1, w-1])
            I_trans[i][j] = image[pos[0]][pos[1]]
    return I_trans

def GetLineSet(t, L0, L1):
    source_center = (L0[1] + L0[0]) / 2
    target_center = (L1[1] + L1[0]) / 2
    center = (1-t) * source_center + t * target_center
    source_vector = L0[1] - L0[0]
    target_vector = L1[1] - L1[0]
    vector = (1-t) * source_vector + t * target_vector
    
    newLineSet = np.zeros((2,2))
    newLineSet[0] = np.array([int(center[0]-vector[0]/2), int(center[1]-vector[1]/2)])
    newLineSet[1] = np.array([int(center[0]+vector[0]/2), int(center[1]+vector[1]/2)])
    
    # print(newLineSet)
    return newLineSet

def GenerateAnimation(Image0, L0, Image1, L1):
    A = 1
    B = 2
    P = 0
    times = 20
    gif_frames = []
    video_frames = []
    
    L = np.zeros((len(L0), 2, 2))
    for t in range(1, times+1):
        print("generated frame:", t, "/", times)
        for i in range(len(L0)):            
            L[i] = GetLineSet(t/times, L0[i], L1[i])
        warp_image0 = warp(Image0, L0, L, A, B, P)
        warp_image1 = warp(Image1, L1, L, A, B, P)
        
        final_frame = ((1 - (t / times)) * warp_image0 + (t / times) * warp_image1).astype('uint8')
        FinalImage = (1-(t/times)) * warp_image0 + (t/times) * warp_image1
        
        gif_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(gif_frame)
        
        video_frames.append(final_frame)

        #cv2.imwrite('tmp'+str(t)+'.png', FinalImage)
    imageio.mimsave('morph_animation.gif', gif_frames, duration=0.1)
    print("GIF has been saved as morph_animation.gif")
    
    height, width, _ = video_frames[0].shape
    video = cv2.VideoWriter('morph_animation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for frame in video_frames:
        video.write(frame)
    video.release()
    print("MP4 video has been saved as morph_animation.mp4")

def main():
    I_source = cv2.imread("./sample.png") # 3 channel
    I_target = cv2.imread("./sketch.png")
    #BasicImageProcess.save_image("I_source.png", I_source)
    #BasicImageProcess.save_image("I_target.png", I_target)

    L1 = np.array([[[144, 17], [258, 16]],
                [[69, 103], [330, 110]],
                [[102, 165],[130, 158]],
                [[169, 164], [205, 160]], 
                [[245, 165], [285, 167]], 
                [[99, 210], [142, 212]], 
                [[260, 216], [345, 221]], 
                #[[208, 245], [208, 245]],
                [[67, 296], [115, 302]],
                [[176, 290], [230, 285]],                
                [[205, 301], [204, 333]],
                [[280, 303], [323, 298]], 
                [[149,371], [254, 367]]
                ])
    L2 = np.array([[[139, 14], [264, 13]], 
                [[78, 98], [322, 102]], 
                [[125, 140], [160, 142]], 
                [[195, 147], [228, 147]], 
                [[260, 144], [283, 144]], 
                [[117, 178], [160, 180]], 
                [[256, 182], [300, 182]], 
                #[[209,235], [209,235]],                
                [[53, 262], [124, 269]],                 
                [[168, 273], [248, 270]], 
                [[283, 272], [326, 274]], 
                [[203, 278], [204, 322]], 
                [[108, 347], [296, 355]]
                ])

    GenerateAnimation(I_source, L1, I_target, L2)


if __name__ == '__main__':
    main()