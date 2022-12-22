# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 22:28:39 2022

@author: klwonder
"""

# Read all the transcripts
import numpy as np
import pandas as pd
import  os
import pickle
import random
import shutil
import json

from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

import gensim
from gensim.models import Word2Vec
import gensim.downloader
from num2words import num2words
from collections import OrderedDict

result_csv=pd.read_csv("./data/wsb/result.csv")
info_csv=pd.read_csv("./data/wsb/info.csv")
videos=set(result_csv['txtname'])

def replace_txt(x):
    x=x.replace("_annotated.txt","")
    x=x.replace(".txt","")
    return(x)

video_names=[replace_txt(x) for x in videos]

video_name=video_names[0]

trans_loc="./data/wsb/transcript/"


g_news = gensim.downloader.load('glove-twitter-200')


  

def get_item_attr(x):
    if("start_time" in x.keys()):
        begin_frame=int(fps*float(x['start_time']) )
        end_frame=int(fps*float(x['end_time']))
        word=x['alternatives'][0]['content'] 
        # check the length of the word
        #word_vector=[g_news.get_vector(x.lower()) for x in word]
        word_x=word_tokenize(word)
        if(len(word_x)<=1):
            
                try:
                    word_vec=g_news.get_vector(word_x[0].lower())
                    begin_frame_list.append(begin_frame)
                    end_frame_list.append(end_frame)
                    word_vec_list.append(word_vec)
                except KeyError:
                    try:
                        # y is the number
                        y=num2words(int(word_x[0]))
                        if(" " in y):
                            word_vec=g_news.get_vector(y.split(" ")[0])
                        elif("-" in y):
                            word_vec=g_news.get_vector(y.split("-")[0])
                        else:
                            word_vec=g_news.get_vector(y)
                            
                        begin_frame_list.append(begin_frame)
                        end_frame_list.append(end_frame)
                        word_vec_list.append(word_vec)
                        
                    except:
                        begin_frame_list.append(begin_frame)
                        end_frame_list.append(end_frame)
                        word_vec_list.append(np.array([0.5]*200))


def write_trans_features(idx):
    b_frame=begin_frame_list[idx]
    e_frame=end_frame_list[idx]
    word_vec=word_vec_list[idx]
    for frame in range(b_frame,e_frame):
        audio_dict[frame]=word_vec

def complete_dict(frame):
    if(frame not in audio_dict.keys()):
        audio_dict[frame]=np.array([0.0]*200)
    
for video_name in video_names:

    with open(trans_loc+video_name+".json","r") as trans_file:
        data=json.load(trans_file)
    fps=info_csv[info_csv['video_name']==video_name]["frames"].iloc[0]
    items=data['results']['items']
    begin_frame_list=[]
    end_frame_list=[]
    word_vec_list=[]
    error_list=[]
    audio_dict={}
    [get_item_attr(x) for x in items]
    [write_trans_features(idx) for idx in range(len(begin_frame_list))]
    
    feat_folder="./data/wsb/feature"
    filename = os.path.join(feat_folder,video_name+"_feat.p" )
    with open(filename,'rb') as f:
        feats = np.array(pickle.load(f)).astype(np.float32) 
    
    [complete_dict(frame) for frame in range(len(feats))]
    
    # Dump pure audio features
    feat_folder="./data/wsb/onlywords"
    filename=os.path.join(feat_folder,video_name+"_feat.p" )
    dict1 = OrderedDict(sorted(audio_dict.items()))
    audio_feature=list(dict1.values())
    with open(filename,'wb') as fp:
        pickle.dump(audio_feature,fp, protocol=pickle.HIGHEST_PROTOCOL)   
        
    # Another approach is to dump video+word
    feat_folder="./data/wsb/feature_words"
    filename=os.path.join(feat_folder,video_name+"_feat.p" )
    dict1 = OrderedDict(sorted(audio_dict.items()))
    audio_feature=list(dict1.values())
    all_feature=[np.concatenate((feats[i],audio_feature[i])) for i in range(len(feats))]
    with open(filename,'wb') as fp:
        pickle.dump(all_feature,fp, protocol=pickle.HIGHEST_PROTOCOL)      
    
# 200-dimension vectors

