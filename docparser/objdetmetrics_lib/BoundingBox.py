from docparser.objdetmetrics_lib.utils import *


class BoundingBox:
    def __init__(self,
                 imageName,
                 classId,
                 x,
                 y,
                 w,
                 h,
                 typeCoordinates=CoordinatesType.Absolute,
                 imgSize=None,
                 bbType=BBType.GroundTruth,
                 classConfidence=None,
                 format=BBFormat.XYWH,
                 bbox_id=None,
                 column=None):
        """Constructor.
        Args:
            imageName: String representing the image name.
            classId: String value representing class id.
            x: Float value representing the X upper-left coordinate of the bounding box.
            y: Float value representing the Y upper-left coordinate of the bounding box.
            w: Float value representing the width bounding box.
            h: Float value representing the height bounding box.
            typeCoordinates: (optional) Enum (Relative or Absolute) represents if the bounding box
            coordinates (x,y,w,h) are absolute or relative to size of the image. Default:'Absolute'.
            imgSize: (optional) 2D vector (width, height)=>(int, int) represents the size of the
            image of the bounding box. If typeCoordinates is 'Relative', imgSize is required.
            bbType: (optional) Enum (Groundtruth or Detection) identifies if the bounding box
            represents a ground truth or a detection. If it is a detection, the classConfidence has
            to be informed.
            classConfidence: (optional) Float value representing the confidence of the detected
            class. If detectionType is Detection, classConfidence needs to be informed.
            format: (optional) Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the
            coordinates of the bounding boxes. BBFormat.XYWH: <left> <top> <width> <height>
            BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
            bbox_id: (optional) A unique ID (per image) to show which ground truth bbox a detection 
            was matched with
        """
        self._imageName = imageName
        self._typeCoordinates = typeCoordinates
        if typeCoordinates == CoordinatesType.Relative and imgSize is None:
            raise IOError(
                'Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if bbType == BBType.Detected and classConfidence is None:
            raise IOError(
                'For bbType=\'Detection\', it is necessary to inform the classConfidence value.')

        self._classConfidence = classConfidence
        self._bbType = bbType
        self._classId = classId
        self._format = format
        self._column = column

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (typeCoordinates == CoordinatesType.Relative):
            (self._x, self._y, self._w, self._h) = convertToAbsoluteValues(imgSize, (x, y, w, h))
            self._width_img = imgSize[0]
            self._height_img = imgSize[1]
            if format == BBFormat.XYWH:
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            else:
                raise IOError(
                    'For relative coordinates, the format must be XYWH (x,y,width,height)')
        # For absolute coords: (x,y,w,h)=real bb coords
        else:
            self._x = x
            self._y = y
            if format == BBFormat.XYWH:
                self._w = w
                self._h = h
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            else:  # format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                self._x2 = w
                self._y2 = h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
        if imgSize is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = imgSize[0]
            self._height_img = imgSize[1]

        self._bbox_id = bbox_id

    def setAbsoluteBoundingBox(self, x, y, w, h):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._x2 = self._x + self._w
        self._y2 = self._y + self._h

    def getAbsoluteBoundingBox(self, format=BBFormat.XYWH):
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)

    def getRelativeBoundingBox(self, imgSize=None):
        if imgSize is None and self._width_img is None and self._height_img is None:
            raise IOError(
                'Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if imgSize is None:
            return convertToRelativeValues((imgSize[0], imgSize[1]),
                                           (self._x, self._y, self._w, self._h))
        else:
            return convertToRelativeValues((self._width_img, self._height_img),
                                           (self._x, self._y, self._w, self._h))

    def getImageName(self):
        return self._imageName

    def getBboxID(self):
        return self._bbox_id

    def getColumn(self):
        return self._column

    def setColumn(self, column):
        self._column = column

    def setBboxID(self, bbox_id):
        self._bbox_id = bbox_id

    def getConfidence(self):
        return self._classConfidence

    def getFormat(self):
        return self._format

    def getClassId(self):
        return self._classId

    def setClassId(self, new_class_id):
        self._classId = new_class_id

    def getImageSize(self):
        return (self._width_img, self._height_img)

    def getCoordinatesType(self):
        return self._typeCoordinates

    def getBBType(self):
        return self._bbType

    @staticmethod
    def compare(det1, det2):
        det1BB = det1.getAbsoluteBoundingBox()
        det1ImgSize = det1.getImageSize()
        det2BB = det2.getAbsoluteBoundingBox()
        det2ImgSize = det2.getImageSize()

        if det1.getClassId() == det2.getClassId() and \
                det1.classConfidence == det2.classConfidenc() and \
                det1BB[0] == det2BB[0] and \
                det1BB[1] == det2BB[1] and \
                det1BB[2] == det2BB[2] and \
                det1BB[3] == det2BB[3] and \
                det1ImgSize[0] == det1ImgSize[0] and \
                det2ImgSize[1] == det2ImgSize[1]:
            return True
        return False

    @staticmethod
    def clone(boundingBox):
        absBB = boundingBox.getAbsoluteBoundingBox(format=BBFormat.XYWH)
        newBoundingBox = BoundingBox(
            boundingBox.getImageName(),
            boundingBox.getClassId(),
            absBB[0],
            absBB[1],
            absBB[2],
            absBB[3],
            typeCoordinates=boundingBox.getCoordinatesType(),
            imgSize=boundingBox.getImageSize(),
            bbType=boundingBox.getBBType(),
            classConfidence=boundingBox.getConfidence(),
            format=BBFormat.XYWH)
        return newBoundingBox

    def get_union_bbox_xywh(self, other_bbox):
        [x0, y0, x1, y1] = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        [other_x0, other_y0, other_x1, other_y1] = other_bbox.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        union_x0 = min(x0, other_x0)
        union_y0 = min(y0, other_y0)
        union_x1 = max(x1, other_x1)
        union_y1 = max(y1, other_y1)
        union_w = union_x1 - union_x0
        union_h = union_y1 - union_y0

        return [union_x0, union_y0, union_w, union_h]

    def intersects(self, other_bbox):
        boxA = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        boxB = other_bbox.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)

        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    def intersectionArea(self, other_bbox):
        boxA = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        boxB = other_bbox.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA) * (yB - yA)

    def getArea(self):
        w, h, _, _ = self.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        return w * h
