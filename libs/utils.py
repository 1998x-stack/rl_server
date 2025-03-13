import random
import numpy as np
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import traceback,sys,time
import os,glob

start_time = time.time()

#运行退出
def exit_run(): 
    
    #if time.time()-start_time>4*60*60: return True
    
    path = os.path.abspath(os.path.dirname(__file__) + '/' + '../') 
                                   
    if os.path.exists(path+"/exit.cmd"):
        return True
    else:
        return False
    
#从models文件夹加载模型 prefix 为前缀 local、sample、grad  filename: prefix_version_y_m_d_h_m_s.td(torch.state_dict)
#路径为 models/prefix/prefix_version_y_m_d_h_m_s.td
def get_model_from_file(model,prefix:str,version=None,model_info:dict=None):
    
    dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)),'../models/',prefix) +"/"
    #拼接文件名称
    #最新文件，或者模糊查找
    if version is not None:
        files = glob.glob(dir_name + prefix+'_'+str(version)+'_'+"*.td")
        if files:
            if len(files) > 1:
                print("model has too many same version: " + str(version))
            file_name = max(files, key=os.path.getmtime)
        else:
            return None
    #查找最新文件
    else:
        files = glob.glob(dir_name+"*.td")
        if files: #files is not an empty list
            file_name = max(files, key=os.path.getmtime)
        else:
            return None
        
    if file_name is not None:
        if model_info is None:
            model_info = dict() 
        model_info.update(torch.load(file_name))
        if ('state_dict' in model_info.keys()) and ('version' in model_info.keys()):
            model.load_state_dict(model_info['state_dict'])
            del model_info['state_dict']
            return model_info['version']
        else:
            return None    
    return None
    
#将模型保存到本地 prefix 为前缀 local、sample、grad filename: prefix_version_y_m_d_h_m_s
def save_model_to_file(model,prefix:str,version,other_info:dict= None):
    
    if version is None:
        return
    
    dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)),'../models/',prefix) +"/"
    
    #判断是否存在路径
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    #删除相同版本    
    #file_name = search_file(dir_name,prefix+'_'+str(version)+'_')
    #if file_name is not None:
    #    os.unlink(file_name)
    
    file_name = dir_name + prefix + '_' + str(version) + '_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".td"
    
    model_info = dict()
    model_info['version'] = version
    model_info['state_dict'] = model.state_dict()
    
    if other_info is not None:
        model_info.update(other_info)
        
    torch.save(model_info, file_name)
    
#杀掉进程
def kill_process(pid):
    # 本函数用于中止传入pid所对应的进程
    if os.name == 'nt':
        # Windows系统
        cmd = 'taskkill /pid ' + str(pid) + ' /f'
        try:
            os.system(cmd)
            print(pid, 'killed')
        except Exception as e:
            print(e)
    elif os.name == 'posix':
        # Linux系统
        cmd = 'kill ' + str(pid)
        try:
            os.system(cmd)
            print(pid, 'killed')
        except Exception as e:
            print(e)
    else:
        print('Undefined os.name')

#设置随机种子      
def setup_seed(seed = None):
    if seed is None:
        seed = 1970010101

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
