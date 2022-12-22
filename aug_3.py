# -*- coding: utf-8 -*-

"""
Created on Mon Dec 12 23:23:01 2022

@author: klwonder
"""
import numpy as np
import pandas as pd
import  os
import pickle
import random
import shutil
import time
def split_video(max_seq_len):
    result_csv=pd.read_csv("./data/wsb/result.csv")
    info_csv=pd.read_csv("./data/wsb/info.csv")
    videos=set(result_csv['txtname'])
    
    begin_list=[]
    end_list=[]
    dur_list=[]
    videoname_list=[]
    fps_list=[]
    new_feat_folder="./data/wsb/feature3"
    time0=time.time()
    try:
      shutil.rmtree(new_feat_folder)
    except:
      print("Not Exist", new_feat_folder)
    os.mkdir(new_feat_folder)
    time1=time.time()
    print("deleting files cost", time1-time0)
    
    
    for video_txt in videos:
        sub_video=result_csv[result_csv['txtname']==video_txt]
        video_name=video_txt.replace("_annotated.txt","")
        video_name=video_name.replace(".txt","")
        
        fps=info_csv[info_csv['video_name']==video_name]["frames"].iloc[0]
        duration=info_csv[info_csv['video_name']==video_name]["durations"].iloc[0]
        segments,labels=  [],[]
        if(len(sub_video)>0):
            
            for loc_index in sub_video.index:
                segments.append([sub_video.loc[loc_index]['begin_second'],
                                 sub_video.loc[loc_index]['end_second']])
                labels.append([0])
            segments = np.asarray(segments, dtype=np.float32)
            labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
        else:
            segments = None
            labels = None
   
      

        # load features
        #change features loaction
        feat_folder="./data/wsb/feature_words"
        filename = os.path.join(feat_folder,video_name+"_feat.p" )
        with open(filename,'rb') as f:
            feats = np.array(pickle.load(f)).astype(np.float32)    
        

        n_00=0
        r_list=[]    
        new_seg_list=[]
        max_sec=duration # how many samples we need to add
        
        max_sec=max_seq_len/fps
        
        time1=time.time()
        
        for i in range(len(segments)):
            n_01=segments[i][0]
            
            
            for j in range(i,min(i+3,len(segments))):
                n_10=segments[j][1]    
                if(j==len(segments)-1):
                    
                    n_11=len(feats)/fps
                else:
                    n_11=segments[j+1][0]

                r_0=random.randint(int(n_00),int(n_01))
                r_1=random.randint(int(n_10),int(n_11))
                   
                if((r_1-r_0)>max_sec): #too large
                  continue
                elif((r_1-r_0)<=5): # too small
                  continue
                else:
                    r_list.append([r_0,r_1])
                    j_seg_list=[]
                    for j2 in range(i,j+1):
                        j_seg_list.append([segments[j2][0],segments[j2][1]])
                    new_seg_list.append(j_seg_list)
            n_00=segments[i][1]
        
        time2=time.time()
        # Restore features and annotations
        print("time2-time1",time2-time1)
        def get_j_feature(j,i):
            s=np.random.normal(0,2,1)
            feats_i=feats[int(r_list[i][0]*fps):int(fps*r_list[i][1])]
            #feats_i=[[y+s[0] for y in x] for x in feats_i]
            feats_i=[[y+s[0]*(y!=0) for y in x] for x in feats_i]
            seg_i=[x-r_list[i][0] for x in new_seg_list[i][j]]
            #print(seg_i)
            new_feat_video=video_name+"_{:03d}".format(i)+"_{:03d}".format(j)
            new_filename=os.path.join(new_feat_folder,new_feat_video+"_feat.p" )
            duration=r_list[i][1]-r_list[i][0]            
            with open(new_filename,"wb") as fp:
                    pickle.dump(feats_i,fp, protocol=pickle.HIGHEST_PROTOCOL)
                
            begin_list.append(seg_i[0])
            end_list.append(seg_i[1])
            dur_list.append(duration)
            videoname_list.append(new_feat_video)
            fps_list.append(fps)
            return(s)
        
        def store_feature(i):
            [get_j_feature(j,i)  for j in range(len(new_seg_list[i]))]
            return i
        [store_feature(i) for i in range(len(r_list))]
        time3=time.time()

        print("time3-time2",time3-time2)
    result_csv2=pd.DataFrame({"begin_second":begin_list,"end_second":end_list,"fps":fps_list,
                              "duration":dur_list,"video_name":videoname_list})

    return(result_csv2)

result_csv2=split_video(4608)
result_csv2.to_csv("./data/wsb/result2.csv")
