1) HPC: Run evaluation (yolo too!) for ALL 7 models
2) yolo is actually wrong in terms of code since I now get 69%. Just look at Adele's code and try
4) Research "online" augmentations (with "online" as in "real time while running the training process")
    - to see if I can make the occlusions on the fly
    - or if I should be creating a shit ton of images
    > https://chatgpt.com/c/6900e9cf-960c-8328-83aa-645d2a999140
    > Try the various approaches (only augs, % of augs, growing % of augs)
    > Take into account storing maybe just the landmarks in case it's an overhead, although it shouldn't be. Based on my experience the patching was much heavier, so maybe optimize that code.



make tf work locally on my gpu: uninstall tf and reinstall and check if this time buda build is true, as cudatoolkit and cudnn are there finanlly