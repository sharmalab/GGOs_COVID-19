import os
import img2ply
#import pdb

if __name__ == "__main__": 
    # get path
    image_path = "/home/ms/Desktop/Point_Cloud_Monjoy/data_19Jan2021/1_image_data/"
    save_path = "/home/ms/Desktop/Point_Cloud_Monjoy/data_19Jan2021/2_ply_data/"
    source_folders = os.listdir(image_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Created ouput directory: " + save_path)
    for single_folder in source_folders:
        fname_image_folder = os.path.basename(single_folder)
        #print(single_file)
        input = (image_path + single_folder)
        ply = (save_path +fname_image_folder +".ply")
        
        img2ply.convert(
        input, 
        ply, 
        [15.0, 10.0, 15.0],
        direction="y", 
        inverse=True,
        ignoreAlpha=True,
        wSamples=0, 
        hSamples=0, 
        maintainAspectRatio=True
    #get list of png source files
    
    #pdb.set_trace()
    # get input and output
    #input = os.path.join(path, "images")
    #ply = os.path.join(path, "new_modified.ply")
    
    # convert
#    img2ply.convert(
#        input, 
#        ply, 
#        [15.0, 10.0, 15.0],
#        direction="y", 
#        inverse=True,
#        ignoreAlpha=True,
#        wSamples=0, 
#        hSamples=0, 
#        maintainAspectRatio=True
    )
