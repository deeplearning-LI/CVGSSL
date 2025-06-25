#modified from https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
from torchvision import transforms
from data_aug.loader import GaussianBlur


class Multi_Transform(object):
    def __init__(
            self,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,normalize,init_size=224):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans=[]
        #image_k
        weak = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=1),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=1),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        trans.append(weak)


        trans_weak=[]
        #image_q
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )

            weak=transforms.Compose([
            randomresizedcrop,
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=1),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
            trans_weak.extend([weak]*nmb_crops[i])

        trans.extend(trans_weak)
        self.trans=trans
        print("in total we have %d transforms"%(len(self.trans)))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops


class Last_transform(object):
    def __init__(
            self,
            num_crops,transform_train):

        trans=[]
        for i in range(num_crops):
            trans.append(transform_train)
        self.trans=trans
        print("In total we have %d transformations"%len(self.trans))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops
