import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy
class ROCMetric():
    """【完成解析】
        bins是确定ROC的分段数，ROC点的个数是`bins + 1`
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):

        # 在ROC曲线上有bins个段，也就是说有bins+1个坐标点
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)

            # output=preds target=labels nclass=nclass score_thresh=score_thress
            # 这一个阈值下算出来的tp、正样本、虚警、负样本、预测为正的像素点个数
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)

            # 将四个值接到数组里，索引为ROC点号
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        # NOTE: 分母+0.001，转换为float的同时避免除0错误
        # x坐标
        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        # y坐标
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])



class PD_FA():
    """
        【解析完成】，
        PD是按照目标个数计算的召回率，
        FA是按照像素个数计算的虚警率
    """
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins) # 这里分数阈值的变化是从0~255，分bins+1个点
            # TODO: pred的取值范围是多少？
            predits  = np.array((preds > score_thresh).cpu()).astype('int64') # 这里不sigmoid，直接跟取值为0~255的score_thresh比
            predits  = np.reshape (predits,  (256,256)) # 把前面两个没用的维度解掉
            labelss = np.array((labels).cpu()).astype('int64') # label也做同样处理
            labelss = np.reshape (labelss , (256,256))

            # 使用八连通的方法来将01predits转换成image连通标注图，其中每个相同的数字代表一个连通块
            image = measure.label(predits, connectivity=2)
            # regionprops是一个连通标注图的计算器，可以计算所有连通块的各种属性，以列表形式返回
            coord_image = measure.regionprops(image)

            # 对于label也做相同操作
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            # 核心代码
            # 遍历所有预测的连通块，从连通属性表coord_image中拿到他们的面积，存储在image_area_total中
            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            # 如果欧氏距离小于3，则将预测块与label块匹配，添加到match中，并删除这个块
            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break
            # 将没匹配上的预测块认为是虚警，放到dismatch中
            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch) # 虚警像素占比，图像分辨率是256*256
            self.PD[iBin]+=len(self.distance_match) # 在ibin下匹配上的预测块的个数

    def get(self,img_num):

        Final_FA =  self.FA / ((256 * 256) * img_num) # 虚警像素占比，图像分辨率是256*256
        Final_PD =  self.PD /self.target # 以目标个数来计算的召回率

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

class mIoU():
    """
        【完成解析】
        就是计算IoU的，每次update的时候会累加，最后算一个平均值
    """
    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    # TP部分：是目标的像素点，而且也正确预测了
    intersection = predict * ((predict == target).float())

    # NOTE: 混淆矩阵 tp fp tn fn的计算方法
    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """计算交集面积和并集面积"""
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

