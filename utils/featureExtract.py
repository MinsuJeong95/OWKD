class saveFeature:
    def __init__(self):
        self.oriFeature = []
        self.attFeature = []
        self.CBAMSaptialFeature = []

    def saveOriFeature(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.oriFeature.append(grad_output)

    def saveAttFeature(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.attFeature.append(grad_output)

    def saveCBAMSpatialFeature(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.CBAMSaptialFeature.append(grad_output)

    def returnFeature(self):
        return self.oriFeature, self.attFeature, self.CBAMSaptialFeature

    def initFeature(self):
        self.oriFeature.clear()
        self.attFeature.clear()
        self.CBAMSaptialFeature.clear()


class saveFeatureForKd:
    def __init__(self):
        self.oriFeature1 = []
        self.oriFeature2 = []
        self.oriFeature3 = []
        self.oriFeature4 = []

        self.attFeature1 = []
        self.attFeature2 = []
        self.attFeature3 = []
        self.attFeature4 = []

        self.gceFeature1 = []
        self.gceFeature2 = []
        self.gceFeature3 = []

    def saveOriFeature1(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.oriFeature1.append(grad_output)

    def saveOriFeature2(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.oriFeature2.append(grad_output)

    def saveOriFeature3(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.oriFeature3.append(grad_output)

    def saveOriFeature4(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.oriFeature4.append(grad_output)

    def saveAttFeature1(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.attFeature1.append(grad_output)

    def saveAttFeature2(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.attFeature2.append(grad_output)

    def saveAttFeature3(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.attFeature3.append(grad_output)

    def saveAttFeature4(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.attFeature4.append(grad_output)

    def saveGceFeature1(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.gceFeature1.append(grad_output)

    def saveGceFeature2(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.gceFeature2.append(grad_output)

    def saveGceFeature3(self, *args):
        # grad_input = args[1]
        grad_output = args[2]
        self.gceFeature3.append(grad_output)

    def returnOriFeature(self):
        return self.oriFeature1, self.oriFeature2, self.oriFeature3, self.oriFeature4

    def returnAttFeature(self):
        return self.attFeature1, self.attFeature2, self.attFeature3, self.attFeature4

    def returnGceFeature(self):
        return self.gceFeature1, self.gceFeature2, self.gceFeature3

    def initFeature(self):
        self.oriFeature1.clear()
        self.oriFeature2.clear()
        self.oriFeature3.clear()
        self.oriFeature4.clear()
        self.attFeature1.clear()
        self.attFeature2.clear()
        self.attFeature3.clear()
        self.attFeature4.clear()
        self.gceFeature1.clear()
        self.gceFeature2.clear()
        self.gceFeature3.clear()

    def returnFeature(self):
        return [self.oriFeature1, self.oriFeature2, self.oriFeature3, self.oriFeature4], \
               [self.attFeature1, self.attFeature2, self.attFeature3, self.attFeature4],\
               [self.gceFeature1, self.gceFeature2, self.gceFeature3]
