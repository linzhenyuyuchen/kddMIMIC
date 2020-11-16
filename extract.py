import os, yaml
from glob import glob
from preprocessing.featureExtractor import featureExtractor

####################################
def getFeatureAllAxis(dir_name, type_name):
    origin_dir = dataset_dir + dir_name + "/"
    print(origin_dir)
    imagePaths = glob(origin_dir + "CT_origin/*.nii")
    print(len(imagePaths))
    maskPaths = [s.replace("CT_origin/", f"ROI_{type_name}/") for s in imagePaths]
    outputPath = os.path.join(feature_dir, f"{dir_name}_{type_name}.csv")
    feor = featureExtractor(imagePaths, maskPaths, paramPath, outputPath)
    feor.extract()

####################################
def getFeatureShift(dir_name, type_name):
    origin_dir = dataset_dir + dir_name + "/"
    imagePaths = glob(origin_dir + "CT_origin/*.nii")
    maskPaths = [s.replace("CT_origin/", f"ROI_{type_name}/") for s in imagePaths]
    outputPath = os.path.join(feature_dir, f"{dir_name}_{type_name}_origin.csv")
    feor = featureExtractor(imagePaths, maskPaths, paramPath, outputPath)
    feor.extract()
    
    maskPaths = [s.replace("CT_origin/", f"ROI_{type_name}/") for s in imagePaths]
    outputPath = os.path.join(feature_dir, f"{dir_name}_{type_name}_expand1.csv")
    feor = featureExtractor(imagePaths, maskPaths, paramPath, outputPath)
    feor.extract()
    
    maskPaths = [s.replace("CT_origin/", f"ROI_{type_name}/") for s in imagePaths]
    outputPath = os.path.join(feature_dir, f"{dir_name}_{type_name}_shrink1.csv")
    feor = featureExtractor(imagePaths, maskPaths, paramPath, outputPath)
    feor.extract()

####################################
def getFeatureMultiAxis(dir_name, type_name):
    origin_dir = dataset_dir + dir_name + "/"
    imagePaths = glob(origin_dir + "CT_origin/*.nii")
    maskPaths = [s.replace("CT_origin/", f"ROI_{type_name}/") for s in imagePaths]
    outputPath = os.path.join(feature_dir, f"{dir_name}_{type_name}_origin.csv")
    feor = featureExtractor(imagePaths, maskPaths, paramPath, outputPath)
    feor.extract()
    
    maskPaths = [s.replace("CT_origin/", f"ROI_{type_name}_exp1/") for s in imagePaths]
    outputPath = os.path.join(feature_dir, f"{dir_name}_{type_name}_expand1.csv")
    feor = featureExtractor(imagePaths, maskPaths, paramPath, outputPath)
    feor.extract()
    
    maskPaths = [s.replace("CT_origin/", f"ROI_{type_name}_shr1/") for s in imagePaths]
    outputPath = os.path.join(feature_dir, f"{dir_name}_{type_name}_shrink1.csv")
    feor = featureExtractor(imagePaths, maskPaths, paramPath, outputPath)
    feor.extract()

if __name__ == "__main__":
    # Hyper-parameters
    cfg = yaml.load(open("./cfg.yaml", "r"), Loader=yaml.FullLoader)
    dataset_dir = cfg["origin_dir"]
    feature_dir = cfg["feature_dir"]
    paramPath = cfg["params"]
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    if not os.path.exists(paramPath):
        print("paramPath not exists: ", paramPath)
        exit(0)
    ####################################
    # expand-or-shrink-all-axis
    """
    getFeatureAllAxis("C058_train", "shr4.5")
    getFeatureAllAxis("C058_train", "shr3.5")
    getFeatureAllAxis("C058_train", "shr2.5")
    getFeatureAllAxis("C058_train", "shr1.5")
    getFeatureAllAxis("C058_train", "shr0.5")
    getFeatureAllAxis("C058_train", "exp0.5")
    getFeatureAllAxis("C058_train", "exp1.5")
    getFeatureAllAxis("C058_train", "exp2.5")
    getFeatureAllAxis("C058_train", "exp3.5")
    getFeatureAllAxis("C058_train", "exp4.5")
    """


    """
    getFeatureAllAxis("C058_test", "shr6")
    getFeatureAllAxis("C058_test", "shr5")
    getFeatureAllAxis("C058_test", "shr4")
    getFeatureAllAxis("C058_test", "shr3")
    getFeatureAllAxis("C058_test", "shr2")
    getFeatureAllAxis("C058_test", "shr1")
    getFeatureAllAxis("C058_test", "origin")
    getFeatureAllAxis("C058_test", "exp1")
    getFeatureAllAxis("C058_test", "exp2")
    getFeatureAllAxis("C058_test", "exp3")
    getFeatureAllAxis("C058_test", "exp4")
    getFeatureAllAxis("C058_test", "exp5")
    getFeatureAllAxis("C058_test", "exp6")


    getFeatureAllAxis("C058_train", "shr6")
    getFeatureAllAxis("C058_train", "shr5")
    getFeatureAllAxis("C058_train", "shr4")
    getFeatureAllAxis("C058_train", "shr3")
    getFeatureAllAxis("C058_train", "shr2")
    getFeatureAllAxis("C058_train", "shr1")
    getFeatureAllAxis("C058_train", "origin")
    getFeatureAllAxis("C058_train", "exp1")
    getFeatureAllAxis("C058_train", "exp2")
    getFeatureAllAxis("C058_train", "exp3")
    getFeatureAllAxis("C058_train", "exp4")
    getFeatureAllAxis("C058_train", "exp5")
    getFeatureAllAxis("C058_train", "exp6")
    """


    """
    ####################################
    # shift-10mm CT-ROI
    getFeatureShift("C058_train", "shift10")
    getFeatureShift("C058_valid1", "shift10")
    getFeatureShift("C058_valid2", "shift10")
    getFeatureShift("C058_valid3", "shift10")
    ####################################
    """

    ####################################
    # Axis x y z-123mm ROI
    getFeatureMultiAxis("C058_test", "expX1")
    getFeatureMultiAxis("C058_test", "expX2")
    getFeatureMultiAxis("C058_test", "expX3")
    
    getFeatureMultiAxis("C058_test", "expY1")
    getFeatureMultiAxis("C058_test", "expY2")
    getFeatureMultiAxis("C058_test", "expY3")
    exit(0)
    getFeatureMultiAxis("C058_test", "expZ1")
    getFeatureMultiAxis("C058_test", "expZ2")
    getFeatureMultiAxis("C058_test", "expZ3")
    ####################################
    # Axis xy xz yz-123mm ROI
    getFeatureMultiAxis("C058_test", "expX1Y1")
    getFeatureMultiAxis("C058_test", "expX1Y2")
    getFeatureMultiAxis("C058_test", "expX1Y3")
    getFeatureMultiAxis("C058_test", "expX1Z1")
    getFeatureMultiAxis("C058_test", "expX1Z2")
    getFeatureMultiAxis("C058_test", "expX1Z3")
    
    getFeatureMultiAxis("C058_test", "expX2Y1")
    getFeatureMultiAxis("C058_test", "expX2Y2")
    getFeatureMultiAxis("C058_test", "expX2Y3")
    getFeatureMultiAxis("C058_test", "expX2Z1")
    getFeatureMultiAxis("C058_test", "expX2Z2")
    getFeatureMultiAxis("C058_test", "expX2Z3")
    
    getFeatureMultiAxis("C058_test", "expX3Y1")
    getFeatureMultiAxis("C058_test", "expX3Y2")
    getFeatureMultiAxis("C058_test", "expX3Y3")

    # getFeatureMultiAxis("C058_test", "expX3Z1")
    # getFeatureMultiAxis("C058_test", "expX3Z2")
    # getFeatureMultiAxis("C058_test", "expX3Z3")
    # getFeatureMultiAxis("C058_test", "expY1Z1")
    # getFeatureMultiAxis("C058_test", "expY1Z2")
    # getFeatureMultiAxis("C058_test", "expY1Z3")
    # getFeatureMultiAxis("C058_test", "expY2Z1")
    # getFeatureMultiAxis("C058_test", "expY2Z2")
    # getFeatureMultiAxis("C058_test", "expY2Z3")


    getFeatureMultiAxis("C058_test", "expY3Z1")
    getFeatureMultiAxis("C058_test", "expY3Z2")
    getFeatureMultiAxis("C058_test", "expY3Z3")

    ####################################
    # Axis xyz-123mm ROI
    getFeatureMultiAxis("C058_test", "expX1Y1Z2")
    # getFeatureMultiAxis("C058_test", "expX1Y1Z3")
    # getFeatureMultiAxis("C058_test", "expX1Y2Z2")
    # getFeatureMultiAxis("C058_test", "expX1Y2Z3")
    # getFeatureMultiAxis("C058_test", "expX1Y3Z2")
    # getFeatureMultiAxis("C058_test", "expX1Y3Z3")

    getFeatureMultiAxis("C058_test", "expX2Y1Z1")
    # getFeatureMultiAxis("C058_test", "expX2Y1Z2")
    # getFeatureMultiAxis("C058_test", "expX2Y1Z3")
    # getFeatureMultiAxis("C058_test", "expX2Y2Z1")
    # getFeatureMultiAxis("C058_test", "expX2Y2Z3")
    # getFeatureMultiAxis("C058_test", "expX2Y3Z1")
    getFeatureMultiAxis("C058_test", "expX2Y3Z2")
    # getFeatureMultiAxis("C058_test", "expX2Y3Z3")







