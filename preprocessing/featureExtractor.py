import os, csv, six
from tqdm import tqdm
from radiomics import featureextractor

class featureExtractor():
    def __init__(self, imagePaths, maskPaths, paramPath, outputPath):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.paramPath = paramPath
        self.outputPath = outputPath
        assert len(self.imagePaths) == len(self.maskPaths), "# is not consistent!"

    def save2csv(self, features):
        if len(features) == 0:
            print("No features!")
        else:
            obs_header = features[0].keys()
            if os.path.exists(self.outputPath) or os.path.isfile(self.outputPath):
                os.remove(self.outputPath)
            f = open(self.outputPath, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
            w = csv.DictWriter(open(self.outputPath, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
            for fea in tqdm(features, total=len(features)):
                w.writerows([fea])
            print("features saved to ", self.outputPath)

    def singleExtract(self, imageName, maskName, params):

        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(params)
            result = extractor.execute(imageName, maskName)
            feature = {key: val for key, val in six.iteritems(result)}
        except Exception as e:
            print(e)
            print("error when extacting ", imageName)
            feature = None
        return feature

    def extract(self):
        features = []
        lens = len(self.imagePaths)
        for i in tqdm(range(lens), total=lens):
            imageName = self.imagePaths[i]
            maskName = self.maskPaths[i]
            feature = self.singleExtract(imageName, maskName, self.paramPath)
            if feature is not None:
                features.append(feature)
        self.save2csv(features)
