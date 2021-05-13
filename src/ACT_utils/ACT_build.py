from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pickle

import numpy as np

from progress.bar import Bar

from datasets.init_dataset import get_dataset

from .ACT_utils import nms2d, nms_tubelets, iou2d


def load_frame_detections(opt, dataset, K, vlist, inference_dir):
    alldets = []  # list of numpy array with <video_index> <frame_index> <ilabel> <score> <x1> <y1> <x2> <y2>
    bar = Bar('{}'.format('FrameAP'), max=len(vlist))
    
    #v3_in = [10, 11, 16, 17, 27, 36, 37, 38, 39, 44, 45, 46, 49, 50, 51, 52, 54, 55, 58, 69, 70, 71, 72, 76, 77, 78, 79, 80, 81, 82, 86, 87, 88, 89, 90, 95, 96, 97, 98, 99, 100, 103, 104, 105, 106, 107, 108, 109, 110, 114, 115, 116, 120, 121, 122, 123, 124, 125, 126, 127, 128, 132, 133, 134, 135, 136, 137, 138, 141, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 163, 164, 165, 166, 167, 168, 176, 177, 178, 179, 180, 181, 195, 196, 197, 199, 200, 202, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 244, 245, 250, 251, 254, 255, 257, 258, 260]
    #v3_out = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 47, 48, 53, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 73, 74, 75, 83, 84, 85, 91, 92, 93, 94, 101, 102, 111, 112, 113, 117, 118, 119, 129, 130, 131, 139, 140, 142, 160, 161, 162, 169, 170, 171, 172, 173, 174, 175, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 198, 201, 203, 204, 205, 206, 207, 208, 209, 210, 243, 246, 247, 248, 249, 252, 253, 256, 259, 261, 262, 263, 264]
    for iv, v in enumerate(vlist):
        
        
        h, w = dataset._resolution[v]

        # aggregate the results for each frame
        vdets = {i: np.empty((0, 6), dtype=np.float32) for i in range(1, 1 + dataset._nframes[v])} # 1 # x1, y1, x2, y2, score, ilabel

        # ORIG: MOC clip: load results for each starting frame
        #for i in range(1, 1 + dataset._nframes[v] - K + 1):
        
        # debug: target short clip to inspect
        #if dataset._nframes[v] <= K  * opt.ninput:
        #    print('v: {}| frames: {}.'.format(v, dataset._nframes[v]))
        
        if opt.pa_model != '':
            opt_ninput = opt.ninput
        elif opt.rgb_model != '':
            opt_ninput = opt.ninputrgb
        
        for i in range(min(K  * opt_ninput , dataset._nframes[v]) - opt_ninput + 1  , 1 + dataset._nframes[v]): # short-range mem: K, 1 + dataset._nframes[v] # + opt.ninput - 1
            
            
            pkl = os.path.join(inference_dir, v, "{:0>5}.pkl".format(i))
            if not os.path.isfile(pkl):
                print("ERROR: Missing extracted tubelets " + pkl)
                sys.exit()

            with open(pkl, 'rb') as fid:
                dets = pickle.load(fid)

            for label in dets:
                # dets  : {label:  N, 4K+1}
                # 4*K+1 : (x1, y1, x2, y2) * K, score
                tubelets = dets[label]
                labels = np.empty((tubelets.shape[0], 1), dtype=np.int32)
                labels[:, 0] = label - 1
                '''
                # ORIG: MOC clip
                for k in range(K):
                    vdets[i + k] = np.concatenate((vdets[i + k], np.concatenate((tubelets[:, np.array([4 * k, 1 + 4 * k, 2 + 4 * k, 3 + 4 * k, -1])], labels), axis=1)), axis=0)
                '''
                
                '''
                # dense reverse
                n_mem = K - 1
                for k in range(K):
                   vdets[i - k] = np.concatenate((vdets[i - k], np.concatenate((tubelets[:, np.array([4 * (n_mem) , 1 + 4 * (n_mem), 2 + 4 * (n_mem), 3 + 4 * (n_mem), -1])], labels), axis=1)), axis=0)
                   n_mem = n_mem - 1
                '''
                
                '''
                # MOD2: simply enlarge gap from 1 to n (e.g., fixed 2, 3, 4?)
                n_mem = 4
                
                ff = i
                for j in reversed(range(1, n_mem+1)):
                    ff = np.maximum(1, ff - 2)
                    if vdets[ff].shape[0] <= 10000:
                        vdets[ff] = np.concatenate((vdets[ff], np.concatenate((tubelets[:, np.array([4*(j-1), 1+4*(j-1), 2+4*(j-1), 3+4*(j-1), -1])], labels), axis=1)), axis=0)
                        
                for k in range(K-n_mem): # K = clip length + 2(?)
                    #if k != K-n_mem - 1: # to validate pure frame base detection
                    #    continue
                    if vdets[i - (K - n_mem) + k + 1].shape[0] <= 10000:
                        vdets[i - (K - n_mem) + k + 1] = np.concatenate((vdets[i - (K - n_mem) + k + 1], np.concatenate((tubelets[:, np.array([4 * (k+n_mem), 1 + 4 * (k+n_mem), 2 + 4 * (k+n_mem), 3 + 4 * (k+n_mem), -1])], labels), axis=1)), axis=0)
                '''
                
                
                
                # final working SPARSE version: MOD: clip + mem (reverse)
                n_mem = K - 1
                
                # TODO: This was not implemented correctly! (even for the K3+2 training)
                
                clip_lo = i 
                clip_mem = clip_lo // (n_mem + 1)
                
                list_update = []
                list_update.append(i)
                
                # linked clip
                add_ele = 0 # when at cur frame, don't add; otherwise add 1
                for j in range(0, K):
                    if vdets[max(i-opt_ninput*j + add_ele*j, 1)].shape[0] <= 10000:
                        vdets[max(i-opt_ninput*j + add_ele*j, 1)] = np.concatenate((vdets[max(i-opt_ninput*j + add_ele*j, 1)], np.concatenate((tubelets[:, np.array([4*(n_mem), 1+4*(n_mem), 2+4*(n_mem), 3+4*(n_mem), -1])], labels), axis=1)), axis=0)
                        n_mem = n_mem - 1
                        #add_ele = 1
                '''
                # sparse clip; TODO: need to verify if it is implemented correctly!!!
                # memory frames
                if clip_mem != 0:
                    for j in reversed(range(1, n_mem+1)): # n_mem+1
                        if vdets[clip_mem*j].shape[0] < 10000: # continue to add detection when current number of detetoin < N
                            vdets[clip_mem*j] = np.concatenate((vdets[clip_mem*j], np.concatenate((tubelets[:, np.array([4*(j-1), 1+4*(j-1), 2+4*(j-1), 3+4*(j-1), -1])], labels), axis=1)), axis=0)
                            list_update.append(clip_mem*j)
                else:
                    for j in reversed(range(1, n_mem+1)): # n_mem+1
                        if vdets[clip_mem*j + i].shape[0] < 10000: # continue to add detection when current number of detetoin < N
                            vdets[clip_mem*j + i] = np.concatenate((vdets[clip_mem*j + i], np.concatenate((tubelets[:, np.array([4*(j-1), 1+4*(j-1), 2+4*(j-1), 3+4*(j-1), -1])], labels), axis=1)), axis=0)
                            list_update.append(clip_mem*j + i)
                
                
                # target (final) frame
                for _ in range(1): # K = clip length + 2(?)
                    #if k != K-n_mem - 1: # to validate pure frame base detection
                    #    continue
                    if vdets[i].shape[0] < 10000:
                        vdets[i] = np.concatenate((vdets[i], np.concatenate((tubelets[:, np.array([4 * n_mem, 1 + 4 * n_mem, 2 + 4 * n_mem, 3 + 4 * n_mem, -1])], labels), axis=1)), axis=0)
                '''
                
                
                '''
                # ADDED: extrapolation?
                list_update.reverse()
                #print(list_update)
                interval = len(list_update) - 1
                for ii in range(interval):
                    lo_bound = tubelets[:,ii*4:ii*4+4]
                    hi_bound = tubelets[:,ii*4+4:ii*4+8]
                    scores = tubelets[:,-1]
                    assert len(lo_bound) == len(hi_bound)
                    
                    for d in range(len(lo_bound)): 
                        lo_box = lo_bound[d]
                        hi_box = hi_bound[d]
                        sc = scores[d]
                        
                        #assert lo_box[-1] == hi_box[-1] and lo_box[-2] == hi_box[-2] # score and label should be equavelent
                        
                        diff_box = (hi_box - lo_box) / (list_update[ii+1] - list_update[ii])
                        
                        for iii in range(list_update[ii]+1, list_update[ii+1]):
                            if vdets[iii].shape[0] < 10000:
                                vdets[iii] = np.concatenate((vdets[iii], np.array([np.concatenate([lo_box + (iii-list_update[ii])*diff_box, [sc], [label-1]])])))
                '''
        '''
        # does excessive repeats on certain frames compromise acc? --> did not matter much (neglegible)
        for vkey in vdets:
            if len(vdets[vkey]) > 500:
                vdets[vkey] = vdets[vkey][-500:]
        '''
        # Perform NMS in each frame
        # vdets : {frame_num:  K*N, 6} ---- x1, x2, y1, y2, score, label
        for i in vdets:
            num_objs = vdets[i].shape[0]
            for ilabel in range(len(dataset.labels)):
                vdets[i] = vdets[i].astype(np.float32)
                a = np.where(vdets[i][:, 5] == ilabel)[0]
                if a.size == 0:
                    continue
                vdets[i][vdets[i][:, 5] == ilabel, :5] = nms2d(vdets[i][vdets[i][:, 5] == ilabel, :5], 0.6)
            # alldets: N,8 --------> ith_video, ith_frame, label, score, x1, x2, y1, y2
            alldets.append(np.concatenate((iv * np.ones((num_objs, 1), dtype=np.float32), i * np.ones((num_objs, 1),
                                                                                                      dtype=np.float32), vdets[i][:, np.array([5, 4, 0, 1, 2, 3], dtype=np.int32)]), axis=1))
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(iv + 1, len(vlist), v, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()
    return np.concatenate(alldets, axis=0)

def BuildTubes(opt):
    redo = opt.redo
    if not redo:
        print('load previous linking results...')
        print('if you want to reproduce it, please add --redo')
    Dataset = get_dataset(opt.dataset)
    inference_dirname = opt.inference_dir
    K = opt.K
    split = 'val'
    dataset = Dataset(opt, split)

    print('inference_dirname is ', inference_dirname)
    vlist = dataset._test_videos[opt.split - 1]
    bar = Bar('{}'.format('BuildTubes'), max=len(vlist))
    
    # DEBUG: target cerain video / class to build tube
    #'shoot_ball/ImprovingBasketballSkills-BasketballTurnaroundFadeAway_shoot_ball_f_nm_np1_ri_med_0', 'shoot_ball/KELVIN_shoot_ball_u_cm_np1_ba_med_0', 'shoot_ball/KELVIN_shoot_ball_u_cm_np1_ba_med_2', 
    #target_video = ['jump/Gregoire_Airman_showreel_2008_jump_f_cm_np1_le_bad_2']
    
    for iv, v in enumerate(vlist):
       
        #if v not in target_video:
        #    continue
        
        outfile = os.path.join(inference_dirname, v + "_tubes.pkl")
        
        
        if os.path.isfile(outfile) and not redo:
            continue

        RES = {}
        nframes = dataset._nframes[v]
  
        
        # TODO: hardcoded for jhmdb for now 
        if dataset._nframes[v] >= K  * opt.ninput:
            ok_frame_inds = [16, 21, 26, 31, 36, dataset._nframes[v]]  
        
        else:
            print('video: {}; Number of frames: {}'.format(v, dataset._nframes[v]))
            ok_frame_inds = [dataset._nframes[v] - opt.ninput + 1, dataset._nframes[v]]
            #ok_frame_inds = []
            #for kk in range(opt.K):
            #    ok_frame_inds.append(max(dataset._nframes[v] - kk*opt.ninput, 1))
            #ok_frame_inds.reverse()
        
        if opt.inference_mode == 'stream':
            # record the latest four frame index (for the final frame allocate detection)
            last_k_ind = []
            last_k_ind_init = min(K  * opt.ninput, dataset._nframes[v]) - opt.ninput + 1
            last_k_ind.append(last_k_ind_init)
            for _ in range(opt.K - 1):
                last_k_ind_init = max(1, last_k_ind_init - opt.ninput)
                last_k_ind.append(last_k_ind_init)
            last_k_ind.reverse()
        
        # load detected tubelets
        VDets = {}
        # orig: for startframe in range(1, nframes + 2 - K):
        for startframe in range(min(K  * opt.ninput , dataset._nframes[v]) - opt.ninput + 1  , 1 + dataset._nframes[v]):
            
            if opt.inference_mode == 'stream': # otherwise ignore
                if startframe not in ok_frame_inds:
                    continue
            
            if startframe != min(K  * opt.ninput, dataset._nframes[v]) - opt.ninput + 1: # not initial frame (ex: 16)
                
                if opt.inference_mode == 'stream': # otherwise ignore
                    last_k_ind.append(startframe)
                    if len(last_k_ind) > opt.K: # only keep the last K index
                        del last_k_ind[0]
            
            resname = os.path.join(inference_dirname, v, "{:0>5}.pkl".format(startframe))
            if not os.path.isfile(resname):
                print("ERROR: Missing extracted tubelets " + resname)
                sys.exit()

            with open(resname, 'rb') as fid:
                VDets[startframe] = pickle.load(fid)
        
        # added: may not be correct but proceed with learning tube building
        first_endframe = list(VDets.keys())[0]
               
        for ilabel in range(len(dataset.labels)):
            
            FINISHED_TUBES = []
            CURRENT_TUBES = []  # tubes is a list of tuple (frame, lstubelets)
            # calculate average scores of tubelets in tubes

            def tubescore(tt):
                return np.mean(np.array([tt[i][1][-1] for i in range(len(tt))])) # a tube could contain multiple linked mini-tubes (linked over time); hence the for loop

            #orig: for frame in range(1, dataset._nframes[v] + 2 - K):
            
            
            
            for frame in range(min(K  * opt.ninput , dataset._nframes[v]) - opt.ninput + 1  , 1 + dataset._nframes[v]):
                # load boxes of the new frame and do nms while keeping Nkeep highest scored
                
                
                if opt.inference_mode == 'stream':
                    if frame not in ok_frame_inds:
                        continue
                
                
                ltubelets = VDets[frame][ilabel + 1]  # [:,range(4*K) + [4*K + 1 + ilabel]]  Nx(4K+1) with (x1 y1 x2 y2)*K ilabel-score

                ltubelets = nms_tubelets(ltubelets, 0.6, top_k=10)

                # just start new tubes
                if frame == first_endframe: # orig: 1 
                    for i in range(ltubelets.shape[0]):
                        CURRENT_TUBES.append([(first_endframe, ltubelets[i, :])]) # orig: 1
                    continue

                # sort current tubes according to average score
                avgscore = [tubescore(t) for t in CURRENT_TUBES]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]
                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets # mine interpretation: for each tube in the memory, associates with any possible current tubelet 
                    last_endframe, last_tubelet = t[-1] # confusing -1? -> # a tube could contain multiple linked mini-tubes (linked over time)
                    ious = []
                    offset =  round((frame - last_endframe) / opt.ninput) # orig: frame - last_endframe
                    if offset < K:
                        nov = K - offset # number of overlaps
                        ious = sum([iou2d(ltubelets[:, 4 * iov:4 * iov + 4], last_tubelet[4 * (iov + offset):4 * (iov + offset + 1)]) for iov in range(nov)]) / float(nov)
                    else:
                        ious = iou2d(ltubelets[:, :4], last_tubelet[4 * K - 4:4 * K])# head and tail matching

                    valid = np.where(ious >= 0.5)[0]  # 0.5

                    if valid.size > 0: # ONLY match the best QUERY tube to the database, then delete this query
                        # take the one with maximum score
                        idx = valid[np.argmax(ltubelets[valid, -1])]
                        CURRENT_TUBES[it].append((frame, ltubelets[idx, :]))
                        ltubelets = np.delete(ltubelets, idx, axis=0)
                    else:
                        if offset >= opt.K:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::-1]:  # process in reverse order to delete them with the right index why --++--
                    FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]

                # start new tubes
                for i in range(ltubelets.shape[0]):
                    CURRENT_TUBES.append([(frame, ltubelets[i, :])])

            # all tubes are not finished
            FINISHED_TUBES += CURRENT_TUBES

            # build real tubes
            output = []
            
            for t_i, t in enumerate(FINISHED_TUBES):
                # DEBUG
                #print(t_i) 
                score = tubescore(t)

                # just start new tubes
                if score < 0.005:
                    continue

                beginframe = max(t[0][0] - opt.ninput*(K-1), 1) #t[0][0] # TODO: needs to be taken care of (forward vs backward)
                endframe = t[-1][0] #t[-1][0] + K - 1 # TODO
                
                length = endframe + 1 - beginframe

                # delete tubes with short duraton (contibuting to many fp?)
                #if length < min(dataset._nframes[v], (K *opt.ninput - opt.ninput + 1) + opt.ninput*3): # 15
                #    continue
                
                if length < dataset._nframes[v] // 2: # 15
                    continue
                # build final tubes by average the tubelets
                out = np.zeros((length, 6), dtype=np.float32)
                out[:, 0] = np.arange(beginframe, endframe + 1)
                n_per_frame = np.zeros((length, 1), dtype=np.int32) # orig: zeros
                
                for i in range(len(t)):
                    frame, box = t[i] # frame: end frame of a tube
                    n_mem = K - 1
                    
                    if opt.inference_mode == 'stream':
                        # for stream detection
                        if frame != nframes:
                            for k in range(K):
                                out[max(frame - k*opt.ninput, 1) - beginframe, 1:5] += box[4 * n_mem:4 * n_mem + 4]
                                out[max(frame - k*opt.ninput, 1) - beginframe, -1] += box[-1] 
                                n_per_frame[max(frame - k*opt.ninput, 1) - beginframe, 0] += 1
                                n_mem -= 1
                            
                            # for the last frame
                        else: 
                            for k in reversed(range(0, K)):
                                out[last_k_ind[k] - beginframe, 1:5] += box[4 * n_mem:4 * n_mem + 4]
                                out[last_k_ind[k] - beginframe, -1] += box[-1] 
                                n_per_frame[last_k_ind[k] - beginframe, 0] += 1
                                n_mem -= 1
                    
                    else:
                        for k in range(K):
                            out[max(frame - k*opt.ninput, 1) - beginframe, 1:5] += box[4 * n_mem:4 * n_mem + 4]
                            out[max(frame - k*opt.ninput, 1) - beginframe, -1] += box[-1] 
                            n_per_frame[max(frame - k*opt.ninput, 1) - beginframe, 0] += 1
                            n_mem -= 1
                        
                ''' orig
                for i in range(len(t)):
                    frame, box = t[i]
                    for k in range(K):
                        out[frame - beginframe + k, 1:5] += box[4 * k:4 * k + 4] # avg effect on the box coord; more stable?
                        out[frame - beginframe + k, -1] += box[-1]  # single frame confidence
                        n_per_frame[frame - beginframe + k, 0] += 1
                '''
                
                nonzero_ind =n_per_frame!=0 # sparse! would be dividing a lot of zeros
                out[nonzero_ind[:,0], 1:] /= n_per_frame[nonzero_ind[:,0]]
                # orig
                #out[:, 1:] /= n_per_frame
                
                if 0 in out[:,-1]: # if any frame index contains 0 (meaning not filled! This line was creating issues!)
                    #print ('Frame detection interpolation takes place!')
                    # ADDED: extrapolation?
                    nonzero_ind = np.where(nonzero_ind==True)[0]
                    nz_v_prev = -5
                    nonzero_nonlink = []
                    for nz_i, nz_v in enumerate(nonzero_ind):
                        nz_offset = nz_v - nz_v_prev
                        
                        
                        if nz_offset == 1: 
                            nz_v_prev = nz_v
                            continue
                        
                        if nz_i > 0:
                            nonzero_nonlink.append((nz_v_prev, nz_v))
                        nz_v_prev = nz_v
                    
                    for idx, lo_hi in enumerate(nonzero_nonlink):
                        lo_hi_dist = lo_hi[1] - lo_hi[0]
                        lo_box = out[lo_hi[0], 1:]
                        hi_box = out[lo_hi[1], 1:]
                        score = (out[lo_hi[0], -1] + out[lo_hi[1], -1]) / 2.
                        diff_box = (hi_box - lo_box) / lo_hi_dist
                        
                        for offset in range(1, lo_hi_dist):
                            if out[lo_hi[0] + offset, -1] == 0: # if the cell was not filled before
                                out[lo_hi[0] + offset, 1:] = lo_box + offset*diff_box
                                out[lo_hi[0] + offset, -1] = score
                            
                
                output.append([out, score])
                
                # out: [num_frames, (frame idx, x1, y1, x2, y2, score)]

            RES[ilabel] = output
        # RES{ilabel:[(out[length,6],score)]}ilabel[0,...]
        with open(outfile, 'wb') as fid:
            pickle.dump(RES, fid)
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(
            iv + 1, len(vlist), v, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()
