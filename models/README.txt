The code in the cvt/registry file comes from : https://github.com/microsoft/CvT/blob/main/lib/models/cls_cvt.py
Meanwhile, the code in the cmt, registry1 and various decoder files comes from : https://github.com/CIRS-Girona/s3Tseg/blob/main/utils/utils.py


All credit for these architectures go to their respective authors
For cvt, we utilize the core functionality of the original code, but modify the model structure to be applied to an
image-to-image task as opposed to a classification task.